import webrtcvad

import vad
from helpers import FRAME_BYTES, SILENCE, SPEECH


def feed(detector, frame, count):
    """Feed `count` copies of a frame. Returns the 1-based frame number where
    the detector fired, or None if it never did."""
    for i in range(1, count + 1):
        if detector.process_frame(frame):
            return i
    return None


# --- TurnDetector (is the user done talking?) ---

def test_silence_alone_never_ends_a_turn(fake_vad):
    detector = vad.TurnDetector()
    assert feed(detector, SILENCE, 200) is None


def test_turn_ends_after_700ms_of_silence(fake_vad):
    detector = vad.TurnDetector()
    feed(detector, SPEECH, 10)
    # 30ms frames: 23 frames = 690ms (not yet), 24 frames = 720ms (done)
    assert feed(detector, SILENCE, 100) == 24


def test_short_pause_does_not_end_turn(fake_vad):
    detector = vad.TurnDetector()
    feed(detector, SPEECH, 10)
    assert feed(detector, SILENCE, 10) is None     # 300ms pause mid-sentence
    feed(detector, SPEECH, 10)                      # keeps talking
    assert feed(detector, SILENCE, 100) == 24       # timer started over


def test_wrong_size_frame_is_ignored(fake_vad):
    detector = vad.TurnDetector()
    assert detector.process_frame(b"\x01" * 100) is False
    assert detector.speech_started is False


def test_reset_forgets_speech(fake_vad):
    detector = vad.TurnDetector()
    feed(detector, SPEECH, 5)
    assert detector.speech_started is True
    detector.reset()
    assert detector.speech_started is False
    assert feed(detector, SILENCE, 100) is None


def test_start_with_speech_runs_the_silence_timer(fake_vad):
    detector = vad.TurnDetector()
    detector.start_with_speech()
    assert feed(detector, SILENCE, 100) == 24


# --- BargeInDetector (is the user talking over the assistant?) ---

def test_short_noise_blip_does_not_barge_in(fake_vad):
    detector = vad.BargeInDetector()
    assert feed(detector, SPEECH, 9) is None   # 270ms < 300ms


def test_sustained_speech_barges_in_at_300ms(fake_vad):
    detector = vad.BargeInDetector()
    assert feed(detector, SPEECH, 20) == 10


def test_gap_resets_the_barge_in_counter(fake_vad):
    detector = vad.BargeInDetector()
    feed(detector, SPEECH, 9)
    feed(detector, SILENCE, 1)
    assert feed(detector, SPEECH, 9) is None


def test_barge_in_reset(fake_vad):
    detector = vad.BargeInDetector()
    feed(detector, SPEECH, 9)
    detector.reset()
    assert feed(detector, SPEECH, 9) is None


def test_barge_in_ignores_wrong_size_frames(fake_vad):
    detector = vad.BargeInDetector()
    assert feed(detector, b"\x01" * 100, 50) is None


# --- the real webrtcvad library (no fake) ---

def test_real_webrtcvad_accepts_our_frame_size_and_calls_silence_silence():
    real = webrtcvad.Vad(2)
    assert real.is_speech(SILENCE, 16000) is False


def test_real_detectors_run_on_silence_without_errors():
    assert len(SILENCE) == FRAME_BYTES
    assert feed(vad.TurnDetector(), SILENCE, 50) is None
    assert feed(vad.BargeInDetector(), SILENCE, 50) is None
