"""
Speech-to-text using faster-whisper (local, free, CPU-friendly).
"""
import io
import logging
import numpy as np
from faster_whisper import WhisperModel

import config

logger = logging.getLogger("voice-agent.stt")

_model: WhisperModel | None = None


def load_model() -> WhisperModel:
    """Load the whisper model once and reuse it across requests."""
    global _model
    if _model is None:
        logger.info(f"Loading faster-whisper model '{config.WHISPER_MODEL_SIZE}'...")
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded.")
    return _model


def transcribe_pcm16(audio_bytes: bytes, sample_rate: int = config.SAMPLE_RATE) -> str:
    """
    Transcribe raw 16-bit PCM mono audio bytes to text.
    Returns "" if nothing intelligible was said.
    """
    if not audio_bytes:
        return ""

    model = load_model()

    # faster-whisper wants float32 samples in [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_np,
        language="en",
        vad_filter=True,          # extra internal VAD pass to strip silence
        beam_size=1,               # greedy decoding = fastest for real-time use
        condition_on_previous_text=False,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text
