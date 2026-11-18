"""
Checks if the user is asking the assistant to continue/resume after an
interruption, vs just starting a new question.

Kept as a simple keyword check instead of another LLM call - it's free,
instant, and good enough for a small set of phrases. Could swap this for
a real classifier later if the phrase list gets out of hand.
"""
import re

# phrases that mean "keep going with what you were saying"
CONTINUE_WORDS = [
    "continue",
    "go on",
    "keep going",
    "go ahead",
    "carry on",
    "you were saying",
    "finish that",
    "finish your answer",
]

# things people say just to make it stop talking. these are not questions,
# so they shouldn't wipe the paused answer (otherwise "wait" then "continue"
# has nothing left to continue)
PAUSE_WORDS = [
    "wait",
    "stop",
    "hold on",
    "hang on",
    "pause",
    "one second",
    "one sec",
    "just a second",
    "just a moment",
    "hold up",
    "okay",
    "ok",
    "hmm",
    # whisper often hears a short "wait" as one of these
    "great",
    "weight",
]

# a real "continue" request is short. if there are lots of extra words
# ("continue explaining what a neural network is") the user is really
# asking something new, so we don't treat it as a resume
MAX_EXTRA_WORDS = 4

# pause phrases get less room: "wait what is a router" is a question
MAX_PAUSE_EXTRA_WORDS = 2


def clean(text):
    # lowercase and drop punctuation so "Continue!" matches "continue"
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_pause(text):
    text = clean(text)
    if not text:
        return False

    word_count = len(text.split())

    for phrase in PAUSE_WORDS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            if word_count <= len(phrase.split()) + MAX_PAUSE_EXTRA_WORDS:
                return True

    return False


def is_continue(text):
    text = clean(text)
    if not text:
        return False

    word_count = len(text.split())

    for phrase in CONTINUE_WORDS:
        # \b so "go on" doesn't match inside "go online"
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            if word_count <= len(phrase.split()) + MAX_EXTRA_WORDS:
                return True

    return False
