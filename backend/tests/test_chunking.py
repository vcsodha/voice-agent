from llm import split_into_speakable_chunks


def test_no_sentence_end_means_nothing_is_ready():
    assert split_into_speakable_chunks("I am still thinking") == ([], "I am still thinking")


def test_complete_sentences_split_off_and_partial_stays():
    ready, rest = split_into_speakable_chunks("Hello there! How can I help you today? I am thinking")
    assert ready == ["Hello there!", "How can I help you today?"]
    assert rest == "I am thinking"


def test_sentence_waiting_on_trailing_space_is_held_back():
    # no whitespace after the period yet, so we can't tell the sentence is over
    assert split_into_speakable_chunks("One sentence.") == ([], "One sentence.")


def test_decimal_numbers_dont_split_a_sentence():
    ready, rest = split_into_speakable_chunks("It costs 3.5 dollars. Next")
    assert ready == ["It costs 3.5 dollars."]
    assert rest == "Next"


def test_empty_buffer():
    assert split_into_speakable_chunks("") == ([], "")
