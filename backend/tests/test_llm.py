import config
import llm


class FakeStream:
    def __init__(self, deltas):
        self.deltas = deltas

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def text_stream(self):
        async def gen():
            for d in self.deltas:
                yield d
        return gen()


class FakeMessages:
    def __init__(self, deltas):
        self.deltas = deltas
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.deltas)


class FakeClient:
    def __init__(self, deltas):
        self.messages = FakeMessages(deltas)


async def collect(user_text, history):
    return [d async for d in llm.stream_reply(user_text, history)]


async def test_tokens_come_through_in_order(monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeClient(["Hel", "lo ", "there"]))
    monkeypatch.setattr(llm.rag, "build_context_block", lambda q: "")
    assert await collect("hi", []) == ["Hel", "lo ", "there"]


async def test_rag_context_is_added_to_the_system_prompt(monkeypatch):
    client = FakeClient(["ok"])
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm.rag, "build_context_block", lambda q: "Relevant context from the knowledge base:\n- (x.txt) fact")
    await collect("question", [])
    system = client.messages.calls[0]["system"]
    assert system.startswith(config.SYSTEM_PROMPT)
    assert "- (x.txt) fact" in system


async def test_system_prompt_is_unchanged_without_context(monkeypatch):
    client = FakeClient(["ok"])
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm.rag, "build_context_block", lambda q: "")
    await collect("question", [])
    assert client.messages.calls[0]["system"] == config.SYSTEM_PROMPT


async def test_messages_are_history_plus_new_question(monkeypatch):
    client = FakeClient(["ok"])
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm.rag, "build_context_block", lambda q: "")
    history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    await collect("three", history)
    assert client.messages.calls[0]["messages"] == history + [{"role": "user", "content": "three"}]


async def test_model_and_token_limit_come_from_config(monkeypatch):
    client = FakeClient(["ok"])
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm.rag, "build_context_block", lambda q: "")
    await collect("q", [])
    call = client.messages.calls[0]
    assert call["model"] == config.CLAUDE_MODEL
    assert call["max_tokens"] == config.MAX_TOKENS
