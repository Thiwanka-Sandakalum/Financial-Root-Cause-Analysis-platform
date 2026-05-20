from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_question(question: str) -> str:
    return " ".join(question.split()).strip()


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return normalize_question(content)

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block.strip())
                continue

            if not isinstance(block, dict):
                continue

            block_text = block.get("text")
            if isinstance(block_text, str) and block_text.strip():
                parts.append(block_text.strip())

        if parts:
            return normalize_question(" ".join(parts))

    return ""


def messages_from_input_payload(raw_input: object) -> list:
    if isinstance(raw_input, list):
        return raw_input

    if isinstance(raw_input, dict):
        messages = raw_input.get("messages")
        if isinstance(messages, list):
            return messages

    return []


def latest_human_question(messages: list) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        message_role = getattr(message, "role", None)
        content = getattr(message, "content", None)

        if isinstance(message, dict):
            message_type = message.get("type")
            message_role = message.get("role")
            content = message.get("content")

        if isinstance(message, (tuple, list)) and len(message) == 2:
            message_role = message[0]
            content = message[1]

        is_human = str(message_type or "").lower() in ("human", "user") or str(
            message_role or ""
        ).lower() in ("human", "user")

        if is_human:
            text = content_to_text(content)
            if text:
                return text

    return ""


def question_from_state(state: Mapping[str, Any]) -> str:
    question = state.get("question")
    if isinstance(question, str) and question.strip():
        return normalize_question(question)

    messages = state.get("messages")
    if isinstance(messages, list) and messages:
        text = latest_human_question(messages)
        if text:
            return text

    raw_input = state.get("input")
    if raw_input is not None:
        if isinstance(raw_input, str) and raw_input.strip():
            return normalize_question(raw_input)
        if isinstance(raw_input, dict):
            q = raw_input.get("question")
            if isinstance(q, str) and q.strip():
                return normalize_question(q)
        msgs = messages_from_input_payload(raw_input)
        text = latest_human_question(msgs)
        if text:
            return text

    return ""
