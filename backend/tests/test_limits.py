"""
The protections for a public link: access code, how many people at once,
and how many questions per session / per day. All of them are off by default.
"""
import config
import main
from helpers import pause, wait_until


# --- access code ---

async def test_no_code_needed_when_none_is_set(make_agent):
    agent = await make_agent(["Hello there."])
    assert agent.ws.messages("ready")


async def test_missing_code_is_refused_with_a_message(make_agent, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_CODE", "letmein")
    agent = await make_agent([], rejected=True)

    assert agent.ws.messages("ready") == []
    assert "access code" in agent.ws.messages("error")[0]["message"]
    assert agent.ws.closed_with == 1008


async def test_wrong_code_is_refused(make_agent, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_CODE", "letmein")
    agent = await make_agent([], query={"code": "nope"}, rejected=True)

    assert agent.ws.messages("error")
    assert agent.ws.closed_with == 1008


async def test_right_code_is_let_in_and_can_talk(make_agent, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_CODE", "letmein")
    agent = await make_agent(["Paris."], query={"code": "letmein"})

    await agent.user_says("capital of france")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)


async def test_refused_visitor_never_reaches_the_models(make_agent, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_CODE", "letmein")
    agent = await make_agent([], rejected=True)
    assert agent.stt.audio_sizes == []
    assert agent.llm.prompts == []


# --- people at the same time ---

async def test_too_many_connections_are_refused_until_someone_leaves(make_agent, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONNECTIONS", 1)

    first = await make_agent(["Hi."])
    assert main.active_connections == 1

    second = await make_agent([], rejected=True)
    assert "busy" in second.ws.messages("error")[0]["message"]
    assert main.active_connections == 1      # the refused one doesn't count

    await first.close()
    assert main.active_connections == 0      # leaving frees the spot

    third = await make_agent(["Hi again."])
    assert third.ws.messages("ready")


async def test_connection_count_goes_back_to_zero_after_leaving(make_agent):
    agent = await make_agent(["Hi."])
    assert main.active_connections == 1
    await agent.close()
    assert main.active_connections == 0


# --- questions per session ---

async def test_session_limit_stops_the_extra_question(make_agent, monkeypatch):
    monkeypatch.setattr(config, "MAX_TURNS_PER_SESSION", 2)
    agent = await make_agent(["One.", "Two.", "Three."])

    await agent.user_says("first")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    agent.play(1)
    await wait_until(lambda: agent.state.mode == "listening")

    await agent.user_says("second")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 2)
    agent.play(1)
    await wait_until(lambda: agent.state.mode == "listening")

    await agent.user_says("third")
    await wait_until(lambda: agent.ws.messages("error"))
    await pause(0.1)

    assert "session" in agent.ws.messages("error")[0]["message"]
    assert agent.llm.prompts == ["first", "second"]       # claude was not called a third time
    assert agent.state.mode == "listening"                # and it can still hear you


async def test_saying_wait_does_not_use_up_a_question(make_agent, monkeypatch):
    from test_session import STORY, start_story_and_play_one_sentence

    monkeypatch.setattr(config, "MAX_TURNS_PER_SESSION", 2)
    agent = await make_agent([STORY, "Two fine days. Three fine days."])
    await start_story_and_play_one_sentence(agent)       # question 1

    await agent.user_interrupts("Wait.")                 # free
    await wait_until(lambda: agent.state.mode == "listening")
    await agent.user_says("continue")                    # question 2
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)

    assert agent.state.turns == 2
    assert agent.ws.messages("error") == []


async def test_refresh_the_page_resets_the_session_limit_but_not_the_day_limit(make_agent, monkeypatch):
    monkeypatch.setattr(config, "MAX_TURNS_PER_SESSION", 1)
    monkeypatch.setattr(config, "MAX_TURNS_PER_DAY", 2)

    a = await make_agent(["One."])
    await a.user_says("q1")
    await wait_until(lambda: len(a.ws.messages("response_done")) == 1)
    await a.close()

    b = await make_agent(["Two."])                        # new connection = new session
    await b.user_says("q2")
    await wait_until(lambda: len(b.ws.messages("response_done")) == 1)
    await b.close()

    c = await make_agent(["Three."])
    await c.user_says("q3")                               # day total is 2 already
    await wait_until(lambda: c.ws.messages("error"))
    assert "daily limit" in c.ws.messages("error")[0]["message"]
    assert c.llm.prompts == []


# --- questions per day ---

async def test_day_limit_counts_everyone_together_and_resets_tomorrow(make_agent, monkeypatch):
    monkeypatch.setattr(config, "MAX_TURNS_PER_DAY", 1)

    a = await make_agent(["One."])
    await a.user_says("q1")
    await wait_until(lambda: len(a.ws.messages("response_done")) == 1)

    b = await make_agent(["Two."])                        # a different visitor
    await b.user_says("q2")
    await wait_until(lambda: b.ws.messages("error"))
    assert "daily limit" in b.ws.messages("error")[0]["message"]

    main.turns_today["day"] = "yesterday"                 # pretend midnight passed
    await b.user_says("q3")
    await wait_until(lambda: len(b.ws.messages("response_done")) == 1)


async def test_no_limits_by_default(make_agent):
    agent = await make_agent(["A.", "B.", "C.", "D."])
    for i in range(4):
        await agent.user_says(f"q{i}")
        await wait_until(lambda: len(agent.ws.messages("response_done")) == i + 1)
        agent.play(1)
        await wait_until(lambda: agent.state.mode == "listening")
    assert agent.ws.messages("error") == []


# --- privacy ---

async def test_transcripts_are_logged_locally_but_can_be_hidden(make_agent, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="voice-agent.main")

    monkeypatch.setattr(config, "LOG_TRANSCRIPTS", False)
    agent = await make_agent(["Paris."])
    await agent.user_says("my secret question")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 1)
    assert "my secret question" not in caplog.text
    assert "(hidden)" in caplog.text

    caplog.clear()
    monkeypatch.setattr(config, "LOG_TRANSCRIPTS", True)
    agent.play(1)
    await wait_until(lambda: agent.state.mode == "listening")
    agent.llm.answers.append("Berlin.")
    await agent.user_says("another question")
    await wait_until(lambda: len(agent.ws.messages("response_done")) == 2)
    assert "another question" in caplog.text
