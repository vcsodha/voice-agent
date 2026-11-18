"""
Voice Activity Detection (VAD) using webrtcvad.

Two jobs:
1. Turn detection: while the user is talking, detect when they've paused
   long enough (SILENCE_MS_TO_END_TURN) that we should stop listening and
   run STT -> LLM -> TTS.
2. Barge-in: while the assistant is speaking, keep listening on the same
   stream, and if we detect real speech (not just noise), signal the
   server to cancel the current LLM/TTS generation immediately.
"""
import collections
import webrtcvad

import config


class TurnDetector:
    """Feed raw PCM16 frames in; get told when the user has stopped talking."""

    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.frame_bytes = int(config.SAMPLE_RATE * (config.FRAME_MS / 1000.0)) * 2  # 16-bit mono
        self._silence_ms = 0
        self._speech_started = False

    def reset(self):
        self._silence_ms = 0
        self._speech_started = False

    @property
    def speech_started(self):
        return self._speech_started

    def start_with_speech(self):
        # used after a barge-in: the user is already mid-sentence, so the
        # silence timer should be running even if the next frame is quiet
        self._speech_started = True
        self._silence_ms = 0

    def process_frame(self, frame: bytes) -> bool:
        """
        Feed one FRAME_MS-sized frame of PCM16 audio.
        Returns True the moment we decide the user's turn has ended.
        """
        if len(frame) != self.frame_bytes:
            return False  # ignore malformed/partial frames

        is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)

        if is_speech:
            self._speech_started = True
            self._silence_ms = 0
        elif self._speech_started:
            self._silence_ms += config.FRAME_MS
            if self._silence_ms >= config.SILENCE_MS_TO_END_TURN:
                return True
        return False


class BargeInDetector:
    """
    While the assistant is talking, watch for the user starting to speak.
    Requires a short sustained run of speech frames to avoid false triggers
    from background noise or the assistant's own audio leaking into the mic.
    """

    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.frame_bytes = int(config.SAMPLE_RATE * (config.FRAME_MS / 1000.0)) * 2
        self._speech_ms = 0

    def reset(self):
        self._speech_ms = 0

    def process_frame(self, frame: bytes) -> bool:
        """Returns True once sustained speech is detected -> caller should interrupt."""
        if len(frame) != self.frame_bytes:
            return False

        if self.vad.is_speech(frame, config.SAMPLE_RATE):
            self._speech_ms += config.FRAME_MS
            if self._speech_ms >= config.MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN:
                return True
        else:
            self._speech_ms = 0
        return False
