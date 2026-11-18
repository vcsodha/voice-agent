"""
Real-time voice agent server.

WebSocket protocol (client <-> server on /ws):

  Client -> Server:
    - binary frames: raw PCM16 mono audio @ config.SAMPLE_RATE
    - text frames (JSON):
        {"type": "reset"}          clear conversation history
        {"type": "chunk_played"}   browser finished playing one sentence of audio

  Server -> Client:
    - text frames (JSON):
        {"type": "ready"}
        {"type": "transcript", "text": "..."}         (what STT heard)
        {"type": "response_delta", "text": "..."}      (LLM token stream)
        {"type": "response_done"}                      (claude finished writing, audio may still be playing)
        {"type": "interrupted"}                        (barge-in happened)
        {"type": "error", "message": "..."}
    - binary frames: one WAV file per sentence, played in order

Why chunk_played exists: we send audio way faster than it plays. Without
the browser telling us what finished playing, we would think the assistant
was done talking while it was still going, and barge-in would never work.
"""
import asyncio
import collections
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import config
import stt
import tts
import rag
import intent
from llm import stream_reply, split_into_speakable_chunks
from vad import TurnDetector, BargeInDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent.main")

# how many 30ms frames (about 450ms) we keep from just before the user
# starts talking, so we don't cut off the first syllable
PRE_ROLL_FRAMES = 15

# --- limits for a public link (the numbers live in config.py) ---
active_connections = 0
turns_today = {"day": "", "count": 0}


def check_limits(state):
    """Returns a message if this question is over a limit. Otherwise counts
    it and returns None."""
    if config.MAX_TURNS_PER_SESSION and state.turns >= config.MAX_TURNS_PER_SESSION:
        return "You've used all the questions for this session. Refresh the page to start a new one."

    today = time.strftime("%Y-%m-%d")
    if turns_today["day"] != today:
        turns_today["day"] = today
        turns_today["count"] = 0
    if config.MAX_TURNS_PER_DAY and turns_today["count"] >= config.MAX_TURNS_PER_DAY:
        return "The demo has reached its daily limit. Please try again tomorrow."

    state.turns += 1
    turns_today["count"] += 1
    return None


def reject_reason(ws):
    """Why this visitor can't connect, or None if they can."""
    if config.ACCESS_CODE:
        code = ws.query_params.get("code", "")
        if not hmac.compare_digest(code.encode(), config.ACCESS_CODE.encode()):
            return "This demo needs an access code. Open the full link you were given (it ends in ?code=...)."
    if config.MAX_CONNECTIONS and active_connections >= config.MAX_CONNECTIONS:
        return "The demo is busy right now. Please try again in a minute."
    return None


def check_setup():
    """Stop with a clear message if something obvious is missing, instead of
    letting the first conversation fail with a confusing error."""
    problems = []

    # (the second check catches someone copying .env.example without editing it)
    if not config.ANTHROPIC_API_KEY or "paste-your-key" in config.ANTHROPIC_API_KEY:
        problems.append(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste "
            "your key in (get one at https://console.anthropic.com)."
        )

    if not os.path.exists(config.PIPER_MODEL_PATH):
        problems.append(
            f"Piper voice not found at {config.PIPER_MODEL_PATH}. Download it with:\n"
            f"    python -m piper.download_voices --download-dir models en_US-lessac-medium"
        )

    if problems:
        raise RuntimeError("\n\nSetup problem:\n- " + "\n- ".join(problems) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts
    check_setup()
    logger.info("Ingesting knowledge base for RAG...")
    count = await asyncio.to_thread(rag.ingest_knowledge_base)
    logger.info(f"RAG ready with {count} chunks.")
    logger.info("Loading STT and TTS models (first run downloads Whisper, so it can take a minute)...")
    await asyncio.to_thread(stt.load_model)
    await asyncio.to_thread(tts.load_voice)
    logger.info("Voice agent ready. Open http://localhost:8000")
    yield


app = FastAPI(title="Real-Time Voice Agent", lifespan=lifespan)


class SessionState:
    """Everything we remember for one websocket connection."""

    def __init__(self):
        self.mode = "listening"  # or "speaking" (anything from STT through the end of playback)
        self.frame_accumulator = bytearray()
        self.audio_buffer = bytearray()
        self.recent_frames = collections.deque(maxlen=PRE_ROLL_FRAMES)
        self.turn_detector = TurnDetector()
        self.bargein_detector = BargeInDetector()
        self.history: list[dict] = []
        self.gen_task: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()
        self.turns = 0                 # questions asked on this connection (for the limit)

        # --- interrupt and resume stuff ---
        self.current_question = None   # question we're answering right now
        self.generating = False        # True while claude is still streaming
        self.answer_text = ""          # what claude wrote in this segment
        self.heard_before = ""         # part the user already heard before a resume
        self.sent_chunks = []          # sentences we sent audio for (this segment)
        self.chunks_played = 0         # how many of those the browser finished playing
        self.paused = None             # {"question": ..., "said_so_far": ...}

    def reset_for_new_turn(self):
        self.audio_buffer = bytearray()
        self.turn_detector.reset()

    def reset_bargein(self):
        self.bargein_detector.reset()

    def start_answer(self, question, heard_before=""):
        self.current_question = question
        self.heard_before = heard_before
        self.answer_text = ""
        self.sent_chunks = []
        self.chunks_played = 0

    def heard_text(self):
        # only the sentences the browser actually finished playing
        played = " ".join(self.sent_chunks[:self.chunks_played])
        return (self.heard_before + " " + played).strip()

    def full_answer(self):
        return (self.heard_before + " " + self.answer_text).strip()

    def clear_conversation(self):
        self.history = []
        self.paused = None
        self.current_question = None
        self.generating = False
        self.answer_text = ""
        self.heard_before = ""
        self.sent_chunks = []
        self.chunks_played = 0
        self.mode = "listening"
        self.recent_frames.clear()
        self.reset_for_new_turn()
        self.reset_bargein()


async def send_json(ws: WebSocket, state: SessionState, payload: dict):
    async with state.send_lock:
        await ws.send_text(json.dumps(payload))


async def send_audio(ws: WebSocket, state: SessionState, audio_bytes: bytes):
    if not audio_bytes:
        return
    async with state.send_lock:
        await ws.send_bytes(audio_bytes)


async def speak(ws: WebSocket, state: SessionState, text: str):
    """Turn one sentence into audio and send it. Remember the text so we
    know what the user heard if they interrupt."""
    audio = await asyncio.to_thread(tts.synthesize_wav_bytes, text)
    if not audio:
        return
    state.sent_chunks.append(text)
    await send_audio(ws, state, audio)


def maybe_finish(state: SessionState):
    """Called when claude stops generating and every time the browser says
    a sentence finished playing. If both are done, the answer is over."""
    if state.current_question is None:
        return
    if state.generating:
        return
    if state.chunks_played < len(state.sent_chunks):
        return

    answer = state.full_answer()
    if answer:
        state.history.append({"role": "user", "content": state.current_question})
        state.history.append({"role": "assistant", "content": answer})
        # keep history bounded so prompts (and cost) don't grow unbounded
        state.history = state.history[-20:]

    state.current_question = None
    state.mode = "listening"
    state.reset_for_new_turn()
    state.reset_bargein()


def save_interrupted_to_history(state: SessionState):
    """User interrupted and then asked something else. We keep the part
    they heard so the conversation doesn't lose that context."""
    heard = state.paused["said_so_far"]
    state.history.append({"role": "user", "content": state.paused["question"]})
    if heard:
        state.history.append({"role": "assistant", "content": heard + " [interrupted]"})
    else:
        state.history.append({"role": "assistant", "content": "[interrupted before answering]"})
    state.history = state.history[-20:]


async def stream_and_speak(ws: WebSocket, state: SessionState, prompt: str):
    """Ask claude, send the text to the browser, and speak it one
    sentence at a time while it's still being written."""
    state.generating = True
    buffer = ""

    async for delta in stream_reply(prompt, state.history):
        state.answer_text += delta
        buffer += delta
        await send_json(ws, state, {"type": "response_delta", "text": delta})

        ready_chunks, buffer = split_into_speakable_chunks(buffer)
        for chunk in ready_chunks:
            await speak(ws, state, chunk)

    if buffer.strip():
        await speak(ws, state, buffer)

    state.generating = False
    await send_json(ws, state, {"type": "response_done"})
    maybe_finish(state)


async def run_turn(ws: WebSocket, state: SessionState, audio_bytes: bytes):
    """
    Full pipeline for one user turn: STT -> LLM (streamed) -> TTS (per
    sentence). Cancellable so barge-in can interrupt it.
    """
    try:
        text = await asyncio.to_thread(stt.transcribe_pcm16, audio_bytes)
        if not text:
            state.mode = "listening"
            state.reset_for_new_turn()
            state.reset_bargein()
            return

        await send_json(ws, state, {"type": "transcript", "text": text})
        logger.info("Heard: %s (paused answer: %s)", repr(text) if config.LOG_TRANSCRIPTS else "(hidden)", bool(state.paused))

        # user said "continue" and we have something paused -> resume it
        wants_resume = bool(state.paused) and intent.is_continue(text)

        # user just said "wait" / "stop" to cut it off. that's not a question,
        # so stay quiet and keep the paused answer for a later "continue"
        if state.paused and not wants_resume and intent.is_pause(text):
            logger.info("Treated as a pause, keeping the paused answer.")
            state.mode = "listening"
            state.reset_for_new_turn()
            state.reset_bargein()
            return

        # from here on it costs a Claude call, so check the limits first
        problem = check_limits(state)
        if problem:
            logger.info("Turn refused: %s", problem)
            state.mode = "listening"
            state.reset_for_new_turn()
            state.reset_bargein()
            await send_json(ws, state, {"type": "error", "message": problem})
            return

        if wants_resume:
            logger.info("Treated as continue, resuming.")
            await resume_turn(ws, state)
            return

        # otherwise it's a new question. if something was interrupted
        # earlier, keep just the part the user heard in the history
        if state.paused:
            save_interrupted_to_history(state)
            state.paused = None

        state.start_answer(text)
        await stream_and_speak(ws, state, text)

    except asyncio.CancelledError:
        logger.info("Turn cancelled (barge-in).")
        raise
    except Exception as e:
        logger.exception("Error during turn")
        state.generating = False
        state.current_question = None
        state.mode = "listening"
        state.reset_for_new_turn()
        state.reset_bargein()
        await send_json(ws, state, {"type": "error", "message": str(e)})
    finally:
        state.generating = False


async def resume_turn(ws: WebSocket, state: SessionState):
    """
    Picks back up an answer that got cut off. We tell Claude what the user
    already heard and ask it to keep going without repeating itself.
    """
    question = state.paused["question"]
    heard = state.paused["said_so_far"]
    state.paused = None  # used it, clear it

    state.start_answer(question, heard_before=heard)

    if not heard:
        # they got interrupted before hearing anything, so just answer again
        await stream_and_speak(ws, state, question)
        return

    resume_prompt = (
        f'The user asked: "{question}". You started answering and the user '
        f'heard this much before they interrupted: "{heard}". They now want '
        f'you to continue. Pick up naturally right after that - do not '
        f'repeat what was already said and do not restart the answer.'
    )
    await stream_and_speak(ws, state, resume_prompt)


async def interrupt(ws: WebSocket, state: SessionState):
    """The user started talking over the assistant."""
    # remember what they actually heard, in case they say "continue"
    if state.current_question:
        state.paused = {
            "question": state.current_question,
            "said_so_far": state.heard_text(),
        }
    state.current_question = None
    state.generating = False

    if state.gen_task and not state.gen_task.done():
        state.gen_task.cancel()

    state.mode = "listening"
    state.reset_for_new_turn()
    state.reset_bargein()

    # the speech that set off the barge-in already happened, so start the
    # new recording from the recent frames. otherwise we'd lose the first
    # part of what they said (like the "con" in "continue")
    state.audio_buffer = bytearray(b"".join(state.recent_frames))
    state.recent_frames.clear()
    state.turn_detector.start_with_speech()

    await send_json(ws, state, {"type": "interrupted"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_connections
    await ws.accept()

    # a rejected visitor gets one message the page can show, then we hang up
    reason = reject_reason(ws)
    if reason:
        logger.info("Connection refused: %s", reason)
        await ws.send_text(json.dumps({"type": "error", "message": reason}))
        await ws.close(code=1008)
        return

    active_connections += 1
    state = SessionState()
    try:
        await send_json(ws, state, {"type": "ready"})
        await talk(ws, state)
    finally:
        active_connections -= 1


async def talk(ws: WebSocket, state: SessionState):
    """One conversation: read mic frames and messages until the user leaves."""

    frame_bytes = int(config.SAMPLE_RATE * (config.FRAME_MS / 1000.0)) * 2  # 16-bit mono
    pre_roll_bytes = frame_bytes * PRE_ROLL_FRAMES

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                state.frame_accumulator.extend(message["bytes"])

                # process as many complete VAD frames as we have buffered
                while len(state.frame_accumulator) >= frame_bytes:
                    frame = bytes(state.frame_accumulator[:frame_bytes])
                    del state.frame_accumulator[:frame_bytes]

                    if state.mode == "listening":
                        state.audio_buffer.extend(frame)
                        if not state.turn_detector.speech_started:
                            # nobody is talking yet. only keep a bit of audio
                            # from right before, so silence doesn't pile up
                            del state.audio_buffer[:-pre_roll_bytes]

                        if state.turn_detector.process_frame(frame):
                            # user finished speaking -> run the pipeline
                            state.mode = "speaking"
                            state.recent_frames.clear()
                            audio_to_process = bytes(state.audio_buffer)
                            state.gen_task = asyncio.create_task(
                                run_turn(ws, state, audio_to_process)
                            )

                    elif state.mode == "speaking":
                        state.recent_frames.append(frame)
                        if state.bargein_detector.process_frame(frame):
                            await interrupt(ws, state)

            elif message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if payload.get("type") == "chunk_played":
                    state.chunks_played += 1
                    maybe_finish(state)

                elif payload.get("type") == "reset":
                    if state.gen_task and not state.gen_task.done():
                        state.gen_task.cancel()
                    state.clear_conversation()
                    await send_json(ws, state, {"type": "ready"})

    except WebSocketDisconnect:
        pass
    finally:
        if state.gen_task and not state.gen_task.done():
            state.gen_task.cancel()
        logger.info("Client disconnected.")


# Serve the simple frontend. Path is relative to this file so it works in
# docker (/app/backend -> /app/frontend) and when running locally.
FRONTEND_DIR = os.environ.get(
    "FRONTEND_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"),
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
