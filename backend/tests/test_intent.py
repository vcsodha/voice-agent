import pytest

import intent


@pytest.mark.parametrize("text", [
    "continue",
    "Continue!",
    "please continue",
    "go on",
    "keep going",
    "sorry, keep going",
    "you were saying",
    "carry on please",
    "no wait, go on",
])
def test_continue_phrases_are_recognized(text):
    assert intent.is_continue(text) is True


@pytest.mark.parametrize("text", [
    "what is the capital of france",
    "never mind, what about python instead",
    "continue explaining what a neural network is",   # long = really a new question
    "i want to go online",                            # "go on" inside another word
    "stop",
    "tell me about dogs",
    "",
    "   ",
])
def test_other_speech_is_not_continue(text):
    assert intent.is_continue(text) is False


@pytest.mark.parametrize("text", [
    "wait",
    "Wait.",
    "stop",
    "stop talking",
    "hold on",
    "hold on a second",
    "wait a second",
    "pause",
    "okay",
    "hmm",
    "Great.",                # what whisper often hears for "wait"
    "weight",
])
def test_pause_phrases_are_recognized(text):
    assert intent.is_pause(text) is True


@pytest.mark.parametrize("text", [
    "wait what is a router",                 # has a real question after it
    "okay tell me about dogs",
    "great now tell me about cats",
    "stop explaining and tell me about cats",
    "what is the capital of france",
    "i want to stay on topic",               # "ok"/"on" inside other words
    "that was a good joke",
    "",
    "   ",
])
def test_other_speech_is_not_pause(text):
    assert intent.is_pause(text) is False
