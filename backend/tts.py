"""
Text-to-speech using Piper (local, free, fast neural TTS).

The voice files (.onnx + .onnx.json) are downloaded with:
    python -m piper.download_voices --download-dir ../models en_US-lessac-medium
en_US-lessac-medium is a good default (natural, medium size, low latency).
"""
import io
import logging
import wave
from piper import PiperVoice

import config

logger = logging.getLogger("voice-agent.tts")

_voice: PiperVoice | None = None


def load_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        logger.info(f"Loading Piper voice from {config.PIPER_MODEL_PATH}...")
        _voice = PiperVoice.load(config.PIPER_MODEL_PATH, config_path=config.PIPER_CONFIG_PATH)
        logger.info("Piper voice loaded.")
    return _voice


def synthesize_wav_bytes(text: str) -> bytes:
    """
    Turn `text` into a complete in-memory WAV file (bytes) that the browser
    can decode and play. Returns b"" if there's nothing to say.
    """
    if not text.strip():
        return b""

    voice = load_voice()
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
    except wave.Error:
        # piper made no audio at all (for example the text was just "..."),
        # so the wav file never got its format set. nothing to play.
        return b""
    return buf.getvalue()
