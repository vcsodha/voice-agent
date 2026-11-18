"""
Central configuration for the voice agent.
All tunable knobs live here so you don't hunt through files.
"""
import os

from dotenv import load_dotenv

# the project folder = the one that contains backend/, frontend/, data/ and models/
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# if there is a .env file in the project folder, read it (this is where
# ANTHROPIC_API_KEY goes). real environment variables still win over it.
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# --- Anthropic ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Haiku is the fast, cheap one: $0.10 per million input tokens and $0.50 per
# million output tokens (as of Nov 2026, for prompts under 100K tokens, which
# we never come close to). Swap to claude-sonnet-5-5 for smarter answers.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_TOKENS = 400  # keep replies short -> lower TTS latency + lower cost

# --- STT (faster-whisper) ---
# "tiny.en" / "base.en" = fastest, good enough for clear speech.
# "small.en" = better accuracy, still runs fine on CPU.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # fast on CPU

# --- TTS (Piper) ---
# The voice files get downloaded into models/ (see README). The .json file
# sits next to the .onnx file with the same name.
PIPER_MODEL_PATH = os.environ.get(
    "PIPER_MODEL_PATH", os.path.join(PROJECT_DIR, "models", "en_US-lessac-medium.onnx")
)
PIPER_CONFIG_PATH = os.environ.get("PIPER_CONFIG_PATH", PIPER_MODEL_PATH + ".json")

# --- Audio ---
SAMPLE_RATE = 16000        # what the browser sends us / whisper expects
FRAME_MS = 30              # VAD frame size in ms (must be 10/20/30 for webrtcvad)
# These three can be changed in your .env file without touching this file,
# e.g. a line VAD_AGGRESSIVENESS=3 (so updating the code never undoes your tuning).
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))  # 0-3, higher = more aggressive at filtering non-speech
SILENCE_MS_TO_END_TURN = int(os.environ.get("SILENCE_MS_TO_END_TURN", "700"))  # how long a user must pause before we treat it as end of speech
MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN = int(os.environ.get("MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN", "300"))  # ignore tiny noise blips while assistant is talking

# --- RAG ---
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(PROJECT_DIR, "data", "chroma"))
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", os.path.join(PROJECT_DIR, "data", "knowledge"))
RAG_TOP_K = 3

# --- limits for a public link ---
# All off by default (empty or 0), so running it on your own laptop just works.
# On a public site, set them as environment variables on the host.
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")   # visitors must open the page as /?code=THIS
MAX_TURNS_PER_SESSION = int(os.environ.get("MAX_TURNS_PER_SESSION") or 0)  # questions per connection
MAX_TURNS_PER_DAY = int(os.environ.get("MAX_TURNS_PER_DAY") or 0)          # questions per day, everyone together
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS") or 0)              # people talking at the same time

# write what people said into the server log? fine on your own laptop (it helps
# debugging), but on a public site set LOG_TRANSCRIPTS=0 so visitors' words
# aren't stored in your host's logs. the docker image turns it off.
LOG_TRANSCRIPTS = os.environ.get("LOG_TRANSCRIPTS", "1") != "0"

SYSTEM_PROMPT = """You are a helpful, concise voice assistant. You are speaking out loud,
so keep answers short (1-4 sentences unless asked for detail), conversational,
and avoid markdown, bullet points, or anything that doesn't read naturally when spoken.
If context from the knowledge base is provided below, use it to ground your answer;
if it doesn't answer the question, just say so and answer from general knowledge.
"""
