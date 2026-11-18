"""
End-to-end tests of the websocket endpoint in main.py.

A fake user "talks" by sending frames (speech frames, then silence), and a
fake browser reports which sentences finished playing. STT, Claude and TTS
are scripted fakes, so these run in milliseconds and need no models or API key.
"""
import asyncio

import main
from helpers import FRAME_BYTES, HOLD, SILENCE, SPEECH, pause, wait_until

# first two sentences get spoken, then claude "stalls" so we can interrupt
STORY = "One fine day. Two fine days. " + HOLD + " three."


async def start_story_and_play_one_sentence(agent):
    """Helper: ask for a story, let 2 sentences be made, browser plays 1."""
    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 2)
    agent.play(1)
    await wait_until(lambda: agent.state.chunks_played == 1)


# --- a normal turn ---

async def test_normal_turn_streams_text_and_audio(make_agent):
    agent = await make_agent(["Paris is the capital. It is on the Seine."])
    await agent.user_says("capital of france")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    assert agent.ws.messages("transcript")[0]["text"] == "capital of france"
    streamed = "".join(m["text"] for m in agent.ws.messages("response_delta"))
    assert streamed == "Paris is the capital. It is on the Seine."
    assert agent.ws.audio() == [b"WAV:Paris is the capital.", b"WAV:It is on the Seine."]


async def test_history_is_written_only_after_playback_finishes(make_agent):
    agent = await make_agent(["Paris is the capital. It is on the Seine."])
    await agent.user_says("capital of france")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    # claude is finished but the browser hasn't played anything yet
    assert agent.state.mode == "speaking"
    assert agent.state.history == []

    agent.play(2)
    await wait_until(lambda: agent.state.mode == "listening")
    assert agent.state.history == [
        {"role": "user", "content": "capital of france"},
        {"role": "assistant", "content": "Paris is the capital. It is on the Seine."},
    ]
    assert agent.state.paused is None


async def test_nothing_intelligible_means_no_llm_call(make_agent):
    agent = await make_agent([])
    await agent.user_says("")
    await pause()
    assert agent.llm.prompts == []
    assert agent.ws.messages("transcript") == []
    assert agent.state.mode == "listening"


async def test_continue_with_nothing_paused_is_just_a_question(make_agent):
    agent = await make_agent(["Sure, what would you like?"])
    await agent.user_says("continue")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert agent.llm.prompts == ["continue"]


async def test_error_from_claude_is_reported_and_session_recovers(make_agent):
    agent = await make_agent([RuntimeError("boom"), "Back to normal."])
    await agent.user_says("first")
    await wait_until(lambda: agent.ws.messages("error"))
    assert agent.ws.messages("error")[0]["message"] == "boom"
    assert agent.state.mode == "listening"

    await agent.user_says("second")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert agent.llm.prompts == ["first", "second"]


# --- barge-in: what gets remembered ---

async def test_interrupt_while_claude_is_still_generating(make_agent):
    agent = await make_agent([STORY])
    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 2)
    assert agent.state.generating is True

    await agent.user_interrupts()

    # the browser never reported playing anything, so nothing was "heard"
    assert agent.state.paused == {"question": "tell me a story", "said_so_far": ""}
    assert agent.state.mode == "listening"
    assert agent.state.generating is False
    await wait_until(lambda: agent.llm.cancelled == 1)   # generation really stopped


async def test_interrupt_remembers_only_what_was_actually_played(make_agent):
    agent = await make_agent([STORY])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts()

    # two sentences were generated but only the first one was heard
    assert agent.state.paused == {"question": "tell me a story", "said_so_far": "One fine day."}


async def test_interrupt_during_playback_after_claude_finished(make_agent):
    agent = await make_agent(["Alpha is first. Beta is second."])
    await agent.user_says("list two things")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    agent.play(1)
    await wait_until(lambda: agent.state.chunks_played == 1)

    await agent.user_interrupts()

    assert agent.state.paused == {"question": "list two things", "said_so_far": "Alpha is first."}
    assert agent.state.gen_task.done()
    assert agent.state.mode == "listening"


async def test_short_noise_blip_does_not_interrupt(make_agent):
    agent = await make_agent([STORY])
    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 2)

    agent.ws.push_frames(SPEECH, 5)
    agent.ws.push_frames(SILENCE, 5)
    agent.ws.push_frames(SPEECH, 5)
    await wait_until(lambda: agent.ws.incoming.empty())
    await pause()

    assert agent.ws.messages("interrupted") == []
    assert agent.state.mode == "speaking"


async def test_late_playback_ack_after_interrupt_is_ignored(make_agent):
    # the browser's "sentence finished" message can land just after the
    # interrupt. it must not turn the cancelled answer into a finished one
    agent = await make_agent([STORY])
    await start_story_and_play_one_sentence(agent)
    await agent.user_interrupts()

    agent.play(1)
    await wait_until(lambda: agent.ws.incoming.empty())
    await pause()

    assert agent.state.history == []
    assert agent.state.paused == {"question": "tell me a story", "said_so_far": "One fine day."}


async def test_talking_again_while_still_transcribing_keeps_the_paused_answer(make_agent):
    agent = await make_agent([STORY, "Two fine days."])
    await start_story_and_play_one_sentence(agent)
    expected = {"question": "tell me a story", "said_so_far": "One fine day."}

    agent.stt.delay = 0.3
    await agent.user_interrupts("continue")   # transcription is now busy for 300ms
    await agent.user_interrupts()             # ...and the user keeps talking
    assert agent.state.paused == expected

    # they finish their sentence; this time it works and the resume goes through
    agent.stt.delay = 0
    calls = len(agent.stt.audio_sizes)
    agent.stt.utterances.append("continue")
    agent.ws.push_frames(SILENCE, 30)
    await wait_until(lambda: len(agent.stt.audio_sizes) > calls)
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert "One fine day." in agent.llm.prompts[1]


async def test_interruption_audio_is_kept_for_transcription(make_agent):
    # regression: the start of what the user says ("con" in "continue") used
    # to be thrown away when the old turn was cancelled
    agent = await make_agent([STORY, "Two fine days."])
    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 2)

    await agent.user_interrupts("continue")

    audio = agent.stt.audio[-1]
    speech_frames = sum(
        1 for i in range(0, len(audio), FRAME_BYTES) if any(audio[i:i + FRAME_BYTES])
    )
    assert speech_frames >= 12     # all 12 frames the user spoke, not just the last one


# --- resume ---

async def test_continue_resumes_instead_of_restarting(make_agent):
    agent = await make_agent([STORY, "Two fine days. Three fine days."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("continue")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    # claude was told what the user heard and asked to pick up from there
    resume_prompt = agent.llm.prompts[1]
    assert "tell me a story" in resume_prompt
    assert "One fine day." in resume_prompt
    assert agent.state.paused is None
    assert agent.ws.audio()[2:] == [b"WAV:Two fine days.", b"WAV:Three fine days."]

    # afterwards the history holds ONE clean exchange with the full answer
    agent.play(2)
    await wait_until(lambda: agent.state.mode == "listening")
    assert agent.state.history == [
        {"role": "user", "content": "tell me a story"},
        {"role": "assistant", "content": "One fine day. Two fine days. Three fine days."},
    ]


async def test_new_question_after_interrupt_keeps_context_and_clears_pause(make_agent):
    agent = await make_agent([STORY, "Dogs are loyal."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("what about dogs")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    assert agent.llm.prompts[1] == "what about dogs"
    assert agent.state.paused is None
    # claude still gets the part the user heard, marked as interrupted
    assert agent.llm.histories[1] == [
        {"role": "user", "content": "tell me a story"},
        {"role": "assistant", "content": "One fine day. [interrupted]"},
    ]

    agent.play(1)
    await wait_until(lambda: agent.state.mode == "listening")
    assert len(agent.state.history) == 4
    assert agent.state.history[-1] == {"role": "assistant", "content": "Dogs are loyal."}


async def test_continue_after_hearing_nothing_answers_the_question_again(make_agent):
    agent = await make_agent([STORY, "Once upon a time."])
    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 2)   # made, but never played

    await agent.user_interrupts("continue")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    # nothing to "continue" from, so it just asks the original question again
    assert agent.llm.prompts[1] == "tell me a story"


async def test_saying_wait_to_interrupt_then_continue_resumes(make_agent):
    # what a real person does: cut it off with a word, THEN ask it to continue
    agent = await make_agent([STORY, "Two fine days. Three fine days."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("Wait.")
    await wait_until(lambda: agent.state.mode == "listening")
    await pause(0.1)

    # "Wait." is not a question: claude wasn't called and the pause is kept
    assert len(agent.llm.prompts) == 1
    assert agent.state.paused == {"question": "tell me a story", "said_so_far": "One fine day."}
    assert agent.ws.messages("transcript")[-1]["text"] == "Wait."

    await agent.user_says("continue")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    assert "One fine day." in agent.llm.prompts[1]
    assert agent.state.paused is None
    assert agent.ws.audio()[2:] == [b"WAV:Two fine days.", b"WAV:Three fine days."]


async def test_several_fillers_in_a_row_still_keep_the_pause(make_agent):
    agent = await make_agent([STORY, "Two fine days. Three fine days."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("Wait.")
    await wait_until(lambda: agent.state.mode == "listening")
    await agent.user_says("hold on a second")
    await wait_until(lambda: agent.state.mode == "listening")
    await pause(0.1)

    assert len(agent.llm.prompts) == 1
    assert agent.state.paused is not None

    await agent.user_says("go on")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert "One fine day." in agent.llm.prompts[1]


async def test_wait_with_a_real_question_after_it_is_a_new_question(make_agent):
    agent = await make_agent([STORY, "Dogs are loyal."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("wait what about dogs")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    assert agent.llm.prompts[1] == "wait what about dogs"
    assert agent.state.paused is None


async def test_wait_when_nothing_is_paused_is_just_a_question(make_agent):
    agent = await make_agent(["Okay, I am here."])
    await agent.user_says("wait")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert agent.llm.prompts == ["wait"]


async def test_second_interrupt_during_a_resume_keeps_everything_heard(make_agent):
    agent = await make_agent([STORY, "Two fine days. " + HOLD + " never said."])
    await start_story_and_play_one_sentence(agent)

    await agent.user_interrupts("continue")
    await wait_until(lambda: len(agent.ws.audio()) == 3)   # first resumed sentence made
    agent.play(1)
    await wait_until(lambda: agent.state.chunks_played == 1)

    await agent.user_interrupts()

    assert agent.state.paused == {
        "question": "tell me a story",
        "said_so_far": "One fine day. Two fine days.",
    }


# --- housekeeping ---

async def test_reset_clears_history_and_paused_answer(make_agent):
    agent = await make_agent(["First answer. Second part.", STORY])
    await agent.user_says("hello")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    agent.play(2)
    await wait_until(lambda: len(agent.state.history) == 2)

    await agent.user_says("tell me a story")
    await wait_until(lambda: len(agent.ws.audio()) == 4)
    await agent.user_interrupts()
    assert agent.state.paused is not None

    agent.ws.push_json({"type": "reset"})
    await wait_until(lambda: len(agent.ws.messages("ready")) == 2)

    assert agent.state.history == []
    assert agent.state.paused is None
    assert agent.state.mode == "listening"


async def test_disconnect_cancels_a_running_generation(make_agent):
    agent = await make_agent([HOLD + " never said."])
    await agent.user_says("hello")
    await wait_until(lambda: agent.state.generating is True)

    agent.ws.disconnect()
    await asyncio.wait_for(agent.task, 2)

    await wait_until(lambda: agent.llm.cancelled == 1)
    assert agent.state.gen_task.cancelled()


async def test_stray_playback_acks_when_idle_are_harmless(make_agent):
    agent = await make_agent([])
    agent.play(3)
    await wait_until(lambda: agent.ws.incoming.empty())
    await pause()
    assert agent.state.mode == "listening"
    assert agent.state.history == []
    assert not agent.task.done()


async def test_long_silence_does_not_pile_up_in_memory(make_agent):
    agent = await make_agent([])
    agent.ws.push_frames(SILENCE, 200)    # 6 seconds of nothing
    await wait_until(lambda: agent.ws.incoming.empty())
    await pause()
    assert len(agent.state.audio_buffer) <= main.PRE_ROLL_FRAMES * FRAME_BYTES


async def test_start_of_speech_keeps_a_short_lead_in(make_agent):
    agent = await make_agent([])
    agent.ws.push_frames(SILENCE, 40)
    agent.ws.push_frames(SPEECH, 3)
    await wait_until(lambda: agent.ws.incoming.empty())
    await pause()
    # lead-in of silence (so the first syllable isn't clipped) + the 3 speech frames
    assert len(agent.state.audio_buffer) == (main.PRE_ROLL_FRAMES + 2) * FRAME_BYTES
