"""
Test setup.

The real models (whisper, piper, chromadb, anthropic) are big and slow to
load and none of the tests need them, so we swap them for tiny fake modules
before anything imports them. This has to happen here at the top, before
the test files import main/stt/tts/rag/llm.
"""
import asyncio
import sys
import types

import pytest
import pytest_asyncio

from helpers import (
    Agent, FakeLLM, FakeSTT, FakeVad, fake_tts, wait_until,
)


def make_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass


class FakePiperVoice:
    @classmethod
    def load(cls, *args, **kwargs):
        return cls()


class FakeAsyncAnthropic:
    def __init__(self, *args, **kwargs):
        pass


make_module("faster_whisper", WhisperModel=FakeWhisperModel)
make_module("piper", PiperVoice=FakePiperVoice)
make_module("anthropic", AsyncAnthropic=FakeAsyncAnthropic)

_embed = make_module(
    "chromadb.utils.embedding_functions",
    SentenceTransformerEmbeddingFunction=lambda **kwargs: None,
)
_utils = make_module("chromadb.utils", embedding_functions=_embed)
make_module("chromadb", utils=_utils, PersistentClient=lambda **kwargs: None)


@pytest.fixture
def fake_vad(monkeypatch):
    """Replace webrtcvad's classifier with a dumb one: a frame is 'speech'
    if it has any non-zero byte. Makes the VAD tests deterministic."""
    import vad
    monkeypatch.setattr(vad.webrtcvad, "Vad", FakeVad)


@pytest.fixture(autouse=True)
def fresh_limits(monkeypatch):
    """The visitor limits live in module-level variables, so give every
    test a clean slate (and no limits unless the test sets them)."""
    import config
    import main
    monkeypatch.setattr(main, "active_connections", 0)
    monkeypatch.setattr(main, "turns_today", {"day": "", "count": 0})
    monkeypatch.setattr(config, "ACCESS_CODE", "")
    monkeypatch.setattr(config, "MAX_TURNS_PER_SESSION", 0)
    monkeypatch.setattr(config, "MAX_TURNS_PER_DAY", 0)
    monkeypatch.setattr(config, "MAX_CONNECTIONS", 0)


@pytest_asyncio.fixture
async def make_agent(monkeypatch, fake_vad):
    """
    Runs the REAL websocket endpoint from main.py, but with a fake websocket
    and fake STT / LLM / TTS. Use like:

        agent = await make_agent(["answer one", "answer two"])
        await agent.user_says("some question")
    """
    import main

    agents = []

    async def build(answers, query=None, rejected=False):
        llm = FakeLLM(answers)
        stt = FakeSTT()
        states = []

        class RecordingState(main.SessionState):
            def __init__(self):
                super().__init__()
                states.append(self)

        monkeypatch.setattr(main, "stream_reply", llm)
        monkeypatch.setattr(main.stt, "transcribe_pcm16", stt)
        monkeypatch.setattr(main.tts, "synthesize_wav_bytes", fake_tts)
        monkeypatch.setattr(main, "SessionState", RecordingState)

        agent = Agent(main, llm, stt, states, query)
        await agent.connect(rejected=rejected)
        agents.append(agent)
        return agent

    yield build

    for agent in agents:
        await agent.close()
