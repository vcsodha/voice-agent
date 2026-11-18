// The browser side of the voice agent.
//   mic -> 16kHz PCM frames -> websocket -> server
//   server -> text + one WAV per sentence -> captions + audio queue
// The captions show which sentences you actually heard, which is the same
// thing the server keeps track of for "continue".

const config = window.APP_CONFIG || {};
const accessCode = new URLSearchParams(location.search).get('code') || '';

const pageEl = document.getElementById('page');
const startBtn = document.getElementById('startBtn');
const startLabel = document.getElementById('startLabel');
const statusEl = document.getElementById('status');
const captionsEl = document.getElementById('captions');
const noticeEl = document.getElementById('notice');
const canvas = document.getElementById('wave');
const ctx2d = canvas.getContext('2d');

const WHITE = '255, 255, 255';   // the assistant's voice
const USER = '198, 205, 255';    // your voice (a paler blue-white)

let ws = null;
let audioCtx = null;      // for the microphone
let playCtx = null;       // for playing the answer
let micStream = null;
let workletNode = null;
let micAnalyser = null;
let playAnalyser = null;
let running = false;
let mode = 'idle';        // idle | listening | speaking

// playback queue
let playQueue = [];
let currentSource = null;
let playing = false;
let playEpoch = 0;        // goes up on interrupt so half-decoded audio is thrown away

// captions
let writing = null;       // the answer that text is currently being added to
let liveAnswer = null;    // the answer whose audio is playing
let sawError = false;
let rafId = null;

// ---------- small ui helpers ----------

const wordsFor = {
  idle: 'Ready when you are.',
  connecting: 'Connecting...',
  listening: 'Listening.',
  speaking: 'Speaking.',
};

function setMode(newMode) {
  mode = newMode;
  statusEl.textContent = wordsFor[newMode] || '';
  startBtn.dataset.state = newMode === 'idle' ? 'idle' : (newMode === 'connecting' ? 'connecting' : 'live');
  startLabel.textContent = newMode === 'idle' ? 'Start talking' : (newMode === 'connecting' ? 'Connecting' : 'Stop');
  startBtn.setAttribute('aria-label', newMode === 'idle' ? 'Start talking' : (newMode === 'connecting' ? 'Connecting' : 'Stop the conversation'));
  drawSoon();
}

function showNotice(text) {
  noticeEl.textContent = text;
  noticeEl.hidden = !text;
}

function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function scrollToLatest(el) {
  el.scrollIntoView({ block: 'nearest', behavior: reducedMotion() ? 'auto' : 'smooth' });
}

// ---------- captions ----------

function addUserLine(text) {
  const p = document.createElement('p');
  p.className = 'you';
  p.textContent = text;
  captionsEl.appendChild(p);
  scrollToLatest(p);
}

function addErrorLine(text) {
  const p = document.createElement('p');
  p.className = 'answer-error';
  p.textContent = text;
  captionsEl.appendChild(p);
  scrollToLatest(p);
}

function newAnswer() {
  const el = document.createElement('p');
  el.className = 'answer';
  captionsEl.appendChild(el);
  return { el, text: '', played: 0, playingNow: false, cut: false, done: false };
}

// same rule the server uses to cut the answer into spoken sentences
function sentencesOf(text) {
  return text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
}

function classFor(answer, i) {
  if (i < answer.played) return 'heard';
  if (i === answer.played && answer.playingNow) return 'now';
  return 'pending';
}

// when you cut the assistant off, the sentences you did not hear are not
// shown any more, so the captions only ever show what you actually heard
function render(answer) {
  const sentences = sentencesOf(answer.text);
  const pieces = [];
  sentences.forEach((s, i) => {
    if (answer.cut && i >= answer.played) return;
    const span = document.createElement('span');
    span.className = 'sent ' + classFor(answer, i);
    span.textContent = s + ' ';
    pieces.push(span);
  });
  answer.el.replaceChildren(...pieces);
  answer.el.hidden = pieces.length === 0;   // nothing heard at all: no empty gap
}

function isFinished(answer) {
  return answer.done && answer.played >= sentencesOf(answer.text).length;
}

// ---------- messages from the server ----------

function handleControlMessage(msg) {
  switch (msg.type) {
    case 'ready':
      break;

    case 'transcript':
      writing = null;
      addUserLine(msg.text);
      break;

    case 'response_delta':
      if (!writing) {
        writing = newAnswer();
        liveAnswer = writing;
      }
      writing.text += msg.text;
      render(writing);
      scrollToLatest(writing.el);
      setMode('speaking');
      break;

    case 'response_done':
      if (writing) {
        writing.done = true;
        render(writing);
      }
      writing = null;
      // the audio usually keeps playing for a while after the text is done
      if (!playing && playQueue.length === 0) setMode('listening');
      break;

    case 'interrupted': {
      stopPlayback();
      if (liveAnswer && !isFinished(liveAnswer)) {
        liveAnswer.cut = true;
        liveAnswer.playingNow = false;
        render(liveAnswer);
      }
      writing = null;
      setMode('listening');
      break;
    }

    case 'error':
      sawError = true;
      console.error('server error:', msg.message);
      writing = null;
      addErrorLine(msg.message);
      if (mode !== 'idle') setMode('listening');
      break;
  }
}

// ---------- start / stop ----------

function wsUrl() {
  let base = config.backendUrl
    ? config.backendUrl.replace(/^http/, 'ws').replace(/\/$/, '')
    : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
  base += '/ws';
  return accessCode ? `${base}?code=${encodeURIComponent(accessCode)}` : base;
}

async function start() {
  showNotice('');
  sawError = false;
  captionsEl.replaceChildren();
  writing = null;
  liveAnswer = null;
  pageEl.classList.add('talking');
  setMode('connecting');

  // make the audio contexts right here, inside the click. Safari (and
  // sometimes Chrome) leaves audio frozen if it wasn't started by a click.
  audioCtx = new AudioContext();
  playCtx = new AudioContext();
  audioCtx.resume();
  playCtx.resume();
  playAnalyser = playCtx.createAnalyser();
  playAnalyser.fftSize = 1024;
  playAnalyser.connect(playCtx.destination);

  const mine = new WebSocket(wsUrl());
  ws = mine;
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      handleControlMessage(JSON.parse(event.data));
    } else {
      enqueueAudio(event.data);
    }
  };

  ws.onopen = async () => {
    try {
      await startMic();
    } catch (e) {
      if (ws !== mine) { releaseMic(); return; }  // we were shut down while waiting
      console.error(e);
      showNotice('The microphone is blocked. Click the lock icon in the address bar, allow the microphone, then start again.');
      stop();
      return;
    }
    if (ws !== mine) { releaseMic(); return; }    // the server hung up while we waited
    running = true;
    setMode('listening');
  };

  ws.onclose = () => {
    const wasRunning = running;
    stop();
    // the free host cuts each session after about 5 minutes
    if (wasRunning && !sawError) {
      showNotice('The session ended. Start again to keep talking. (The free host limits each session to about 5 minutes.)');
    }
  };
}

function releaseMic() {
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  micStream = null;
}

function stop() {
  running = false;
  stopPlayback();
  releaseMic();
  if (workletNode) workletNode.disconnect();
  if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
  if (playCtx && playCtx.state !== 'closed') playCtx.close();
  micStream = workletNode = micAnalyser = playAnalyser = null;
  if (ws) {
    const old = ws;
    ws = null;
    old.onclose = null;
    old.close();
  }
  // nothing was said yet (mic blocked, refused): go back to the landing look
  if (!captionsEl.children.length) pageEl.classList.remove('talking');
  setMode('idle');
}

startBtn.addEventListener('click', () => {
  if (mode === 'idle') start(); else if (mode === 'listening' || mode === 'speaking') stop();
});

// ---------- microphone ----------

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const source = audioCtx.createMediaStreamSource(micStream);

  micAnalyser = audioCtx.createAnalyser();
  micAnalyser.fftSize = 1024;
  source.connect(micAnalyser);

  await audioCtx.audioWorklet.addModule('pcm-worklet.js');
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-downsampler', {
    processorOptions: { targetSampleRate: 16000, inputSampleRate: audioCtx.sampleRate },
  });
  workletNode.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  source.connect(workletNode);
}

// ---------- playback ----------

function enqueueAudio(arrayBuffer) {
  playQueue.push(arrayBuffer);
  if (!playing) playNext();
}

// tell the server one sentence finished playing. the server uses this to
// know what you actually heard (needed for interrupt and resume)
function sendChunkPlayed() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'chunk_played' }));
  }
}

function chunkFinished() {
  sendChunkPlayed();
  if (liveAnswer) {
    liveAnswer.played += 1;
    liveAnswer.playingNow = false;
    render(liveAnswer);
  }
  playNext();
}

async function playNext() {
  if (playQueue.length === 0) {
    playing = false;
    if (mode === 'speaking' && !writing) setMode('listening');
    return;
  }
  playing = true;
  const data = playQueue.shift();
  const epoch = playEpoch;
  try {
    const audioBuffer = await playCtx.decodeAudioData(data.slice(0));
    if (epoch !== playEpoch) return;  // you interrupted while this was decoding
    const src = playCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(playAnalyser);
    currentSource = src;
    src.onended = chunkFinished;
    if (liveAnswer) {
      liveAnswer.playingNow = true;
      render(liveAnswer);
    }
    drawSoon();
    src.start();
  } catch (e) {
    if (epoch !== playEpoch) return;
    chunkFinished();  // still count it so the server doesn't wait forever
  }
}

function stopPlayback() {
  playEpoch++;
  playQueue = [];
  if (currentSource) {
    try { currentSource.onended = null; currentSource.stop(); } catch (e) {}
    currentSource = null;
  }
  playing = false;
}

// ---------- the waveform strand ----------
// One bright line that is your real audio (or the assistant's), with a few
// fainter strands that swell and settle with how loud it is. When nothing is
// happening it just breathes.

let level = 0;                    // smoothed loudness, 0 to 1
let samples = null;               // reused buffer for the analyser
const pageOpenedAt = performance.now();
const REVEAL_MS = 1600;           // on load the strand runs in from the left

function sizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx2d.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawSoon() {
  if (!rafId) rafId = requestAnimationFrame(drawWave);
}

// whose voice to draw right now
function currentVoice() {
  if (mode === 'speaking' && playing && playAnalyser) return { analyser: playAnalyser, rgb: WHITE };
  if (mode === 'listening' && micAnalyser) return { analyser: micAnalyser, rgb: USER };
  return { analyser: null, rgb: WHITE };
}

function drawWave(now) {
  rafId = null;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx2d.clearRect(0, 0, w, h);
  const mid = h / 2;
  const t = (now - pageOpenedAt) / 1000;
  const calm = reducedMotion();
  const { analyser, rgb } = currentVoice();

  // read the live audio and work out how loud it is
  let loud = 0;
  if (analyser) {
    if (!samples || samples.length !== analyser.fftSize) samples = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = (samples[i] - 128) / 128;
      sum += v * v;
    }
    loud = Math.min(1, Math.sqrt(sum / samples.length) * 4);
  }
  // swell quickly, settle slowly, so it does not jitter
  level += (loud - level) * (loud > level ? 0.35 : 0.07);
  const breath = calm ? 0 : 0.15 + 0.05 * Math.sin(t * 0.9);
  const energy = Math.max(level, breath);
  const reach = h * 0.42;

  const reveal = calm ? 1 : Math.min(1, (now - pageOpenedAt) / REVEAL_MS);

  // 0 at the edges so the strand is pinned to the middle there, 1 in the centre
  const envelope = (p) => Math.pow(Math.sin(Math.PI * p), 1.3);
  // the load animation: a front that moves left to right
  const shown = (p) => Math.max(0, Math.min(1, (reveal * 1.2 - p) / 0.2));

  // fainter strands, drawn first so the real line sits on top
  ctx2d.lineJoin = 'round';
  ctx2d.shadowBlur = 0;
  for (let k = 4; k >= 0; k--) {
    ctx2d.beginPath();
    for (let x = 0; x <= w; x += 3) {
      const p = x / w;
      const wave = Math.sin(p * (3.6 + k * 1.9) * Math.PI + t * (0.7 + k * 0.28) + k * 1.4);
      const y = mid + wave * reach * energy * (0.8 - k * 0.12) * envelope(p) * shown(p);
      if (x === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
    }
    ctx2d.strokeStyle = `rgba(${rgb}, ${0.14 + (4 - k) * 0.05})`;
    ctx2d.lineWidth = 1.5;
    ctx2d.stroke();
  }

  // the main line
  ctx2d.beginPath();
  const n = samples ? samples.length : 0;
  for (let x = 0; x <= w; x += 2) {
    const p = x / w;
    let v = 0;
    if (analyser && n) {
      // average a few neighbouring samples to take the fizz off
      const at = Math.floor(p * (n - 1));
      let acc = 0;
      for (let j = -3; j <= 3; j++) acc += samples[Math.max(0, Math.min(n - 1, at + j))] - 128;
      v = acc / 7 / 128;
    }
    const swirl = Math.sin(p * 6 * Math.PI + t * 1.3) * breath * 0.9;
    const y = mid + (Math.tanh(v * 3) * 1.1 + swirl) * reach * envelope(p) * shown(p);
    if (x === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
  }
  ctx2d.strokeStyle = `rgb(${rgb})`;
  ctx2d.lineWidth = 2.5;
  ctx2d.shadowColor = `rgba(${rgb}, 0.65)`;
  ctx2d.shadowBlur = 10 + energy * 22;
  ctx2d.stroke();
  ctx2d.shadowBlur = 0;

  // keep animating. with reduced motion we only animate real audio.
  if (!calm || analyser || reveal < 1 || level > 0.01) drawSoon();
}

new ResizeObserver(() => { sizeCanvas(); drawSoon(); }).observe(canvas);

// ---------- page setup ----------

if (config.githubUrl) {
  const link = document.getElementById('githubLink');
  link.href = config.githubUrl;
  link.hidden = false;
}
if (config.ownerName) {
  const owner = document.getElementById('ownerLine');
  owner.textContent = 'By ' + config.ownerName;
  owner.hidden = false;
}

sizeCanvas();
setMode('idle');
