"""Thin wrapper around the OpenAI client so every node calls the LLM the
same way, with the same logging.
"""
import json
import logging

from agent import config

logger = logging.getLogger(__name__)

if config.LANGFUSE_ENABLED:
    try:
        # Drop-in wrapper with the same interface as the OpenAI client.
        from langfuse.openai import OpenAI as InstrumentedOpenAI

        logger.info("Langfuse tracing enabled for OpenAI calls")
    except Exception as exc:
        logger.warning("Langfuse enabled but unavailable (%s). Falling back to plain OpenAI client.", exc)
        from openai import OpenAI as InstrumentedOpenAI
else:
    from openai import OpenAI as InstrumentedOpenAI

client = InstrumentedOpenAI(api_key=config.OPENAI_API_KEY)


def chat(messages: list[dict], model: str | None = None) -> str:
    """Sends a chat completion request and returns the assistant's reply text."""
    model = model or config.OPENAI_MODEL
    logger.info("LLM call (text) -> model=%s, %d message(s)", model, len(messages))
    completion = client.chat.completions.create(model=model, messages=messages)
    text = completion.choices[0].message.content.strip()
    logger.info("LLM reply (text) <- %s", text[:200])
    return text


def chat_json(messages: list[dict], model: str | None = None) -> dict:
    """Same as chat(), but asks the model to reply with a JSON object and parses it.

    Used by nodes that need a structured decision (e.g. the router's intent
    label, or the guardrail's approve/revise verdict) rather than free text.
    """
    model = model or config.OPENAI_MODEL
    logger.info("LLM call (json) -> model=%s, %d message(s)", model, len(messages))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    logger.info("LLM reply (json) <- %s", raw[:300])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON, treating as empty: %s", raw[:300])
        return {}
