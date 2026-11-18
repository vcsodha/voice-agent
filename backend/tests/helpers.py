"""
Fakes and helpers shared by the tests.

Nothing in here imports the project code at import time, so conftest.py can
install the fake model modules first.
"""
import asyncio
import json
import re
import time

FRAME_BYTES = 960                       # 30ms of 16kHz 16-bit mono audio
SPEECH = b"\x01" * FRAME_BYTES          # fake "someone is talking" frame
SILENCE = b"\x00" * FRAME_BYTES         # fake silent frame

HOLD = "<HOLD>"  # put this in a fake answer to make the LLM stall there


class FakeVad:
    """Stand-in for webrtcvad.Vad. A frame counts as speech if it has any
    non-zero byte."""

    def __init__(self, aggressiveness=0):
        pass

    def is_speech(self, frame, sample_rate):
        return any(frame)


def fake_tts(text):
    return b"WAV:" + text.encode()


class FakeSTT:
    """Returns whatever the test queued up, in order. Records how big each
    chunk of audio it was given was."""

    def __init__(self):
        self.utterances = []
        self.audio_sizes = []
        self.audio = []
        self.delay = 0   # seconds to pretend transcription takes

    def __call__(self, audio_bytes, sample_rate=16000):
        self.audio_sizes.append(len(audio_bytes))
        self.audio.append(audio_bytes)
        text = self.utterances.pop(0)
        if self.delay:
            time.sleep(self.delay)
        return text


class FakeLLM:
    """
    Stand-in for llm.stream_reply. `answers` is a list with one entry per
    call. An entry is a string (streamed word by word), or an Exception to
    raise. If the string contains HOLD, the stream stalls there forever, so a
    test can interrupt while the assistant is 'still generating'.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.histories = []
        self.cancelled = 0
        self._never = asyncio.Event()

    async def __call__(self, prompt, history):
        self.prompts.append(prompt)
        self.histories.append(list(history))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        try:
            for token in re.findall(r"\S+\s*", answer):
                if token.strip() == HOLD:
                    await self._never.wait()
                yield token
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class FakeWS:
    """Stand-in for the starlette websocket the endpoint talks to."""

    def __init__(self, query=None):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.query_params = dict(query or {})   # like /ws?code=abc
        self.closed_with = None

    async def accept(self):
        pass

    async def close(self, code=1000):
        self.closed_with = code

    async def receive(self):
        message = await self.incoming.get()
        # real frames arrive spread out over time, so give other tasks
        # (the STT/LLM/TTS pipeline) a chance to run between messages
        await asyncio.sleep(0.001)
        return message

    async def send_text(self, text):
        self.sent.append(("text", json.loads(text)))

    async def send_bytes(self, data):
        self.sent.append(("bytes", data))

    # --- helpers for tests ---
    def push_frames(self, frame, count):
        for _ in range(count):
            self.incoming.put_nowait({"type": "websocket.receive", "bytes": frame})

    def push_json(self, payload):
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def messages(self, kind):
        return [m for t, m in self.sent if t == "text" and m["type"] == kind]

    def audio(self):
        return [m for t, m in self.sent if t == "bytes"]


async def wait_until(condition, timeout=3.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("timed out waiting for condition")


async def pause(seconds=0.1):
    """Let things happen, for tests that check that something did NOT happen."""
    await asyncio.sleep(seconds)


class Agent:
    """A fake user talking to the real endpoint."""

    def __init__(self, main, llm, stt, states, query=None):
        self.main = main
        self.llm = llm
        self.stt = stt
        self.states = states
        self.ws = FakeWS(query)
        self.task = None

    @property
    def state(self):
        return self.states[0]

    async def connect(self, rejected=False):
        self.task = asyncio.create_task(self.main.websocket_endpoint(self.ws))
        if rejected:
            # the server should say why and hang up by itself
            await wait_until(lambda: self.task.done())
        else:
            await wait_until(lambda: self.states and self.ws.messages("ready"))

    async def close(self):
        self.ws.disconnect()
        try:
            await asyncio.wait_for(self.task, 2)
        except asyncio.TimeoutError:
            self.task.cancel()

    def play(self, count=1):
        """The browser says `count` sentences finished playing."""
        for _ in range(count):
            self.ws.push_json({"type": "chunk_played"})

    async def user_says(self, text, speech_frames=20):
        """User talks, then goes quiet long enough for the turn to end.
        Returns once the server has started transcribing."""
        calls = len(self.stt.audio_sizes)
        self.stt.utterances.append(text)
        self.ws.push_frames(SPEECH, speech_frames)
        self.ws.push_frames(SILENCE, 30)  # 900ms, over the 700ms cutoff
        await wait_until(lambda: len(self.stt.audio_sizes) > calls)

    async def user_interrupts(self, text=None, speech_frames=12):
        """User starts talking over the assistant. If `text` is given they
        also finish their sentence (and we wait for it to be transcribed)."""
        count = len(self.ws.messages("interrupted"))
        self.ws.push_frames(SPEECH, speech_frames)
        await wait_until(lambda: len(self.ws.messages("interrupted")) > count)
        if text is not None:
            calls = len(self.stt.audio_sizes)
            self.stt.utterances.append(text)
            self.ws.push_frames(SILENCE, 30)
            await wait_until(lambda: len(self.stt.audio_sizes) > calls)
