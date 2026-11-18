"""
Calls Claude via the Anthropic API, streaming tokens as they arrive so we
can start TTS on the first sentence instead of waiting for the full reply.
"""
import logging
from anthropic import AsyncAnthropic

import config
import rag

logger = logging.getLogger("voice-agent.llm")

_client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


async def stream_reply(user_text: str, history: list[dict]):
    """
    Async generator yielding response text deltas as they arrive from Claude.
    `history` is a list of {"role": "user"|"assistant", "content": str}.
    Automatically grounds the query with RAG context if the knowledge base
    has anything relevant.
    """
    context_block = rag.build_context_block(user_text)
    system = config.SYSTEM_PROMPT
    if context_block:
        system = f"{system}\n\n{context_block}"

    messages = history + [{"role": "user", "content": user_text}]

    async with _client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=messages,
    ) as stream:
        async for text_delta in stream.text_stream:
            yield text_delta


def split_into_speakable_chunks(buffer: str) -> tuple[list[str], str]:
    """
    Given a growing text buffer, split off complete sentences that are ready
    to hand to TTS, and return (ready_chunks, remaining_buffer).
    This lets us start speaking before the whole LLM response has arrived.
    """
    import re
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_end.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    # last part may be incomplete - keep it in the buffer
    ready, remaining = parts[:-1], parts[-1]
    return ready, remaining
