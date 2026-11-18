import io
import wave

import tts


class FakeVoice:
    """Stand-in for piper's PiperVoice. Writes `frames` audio frames, or
    nothing at all if frames is 0."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def synthesize_wav(self, text, wav_file):
        self.calls.append(text)
        if self.frames:
            wav_file.setframerate(22050)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(b"\x00\x01" * self.frames)


def test_blank_text_never_reaches_piper(monkeypatch):
    voice = FakeVoice(10)
    monkeypatch.setattr(tts, "load_voice", lambda: voice)
    assert tts.synthesize_wav_bytes("   ") == b""
    assert voice.calls == []


def test_result_is_a_playable_wav_file(monkeypatch):
    monkeypatch.setattr(tts, "load_voice", lambda: FakeVoice(100))
    data = tts.synthesize_wav_bytes("Hello there.")
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 100


def test_text_with_nothing_to_say_gives_empty_bytes_instead_of_crashing(monkeypatch):
    # piper makes no audio for something like "..." and the wav file is left
    # without a format, which makes wave raise an error when it closes
    monkeypatch.setattr(tts, "load_voice", lambda: FakeVoice(0))
    assert tts.synthesize_wav_bytes("...") == b""
