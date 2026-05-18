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

        message_text = content_to_text(content)
        if not message_text:
            continue

        role = ""
        if isinstance(message_type, str):
            role = message_type.lower()
        elif isinstance(message_role, str):
            role = message_role.lower()

        if role in {"human", "user"}:
            return message_text

    return ""


def question_from_state(state: Mapping[str, Any]) -> str:
    messages = state.get("messages") or []
    latest_human = latest_human_question(messages)
    if latest_human:
        return latest_human

    raw_input = state.get("input") or ""
    input_messages = messages_from_input_payload(raw_input)
    if input_messages:
        question_from_input_messages = latest_human_question(input_messages)
        if question_from_input_messages:
            return question_from_input_messages

    if isinstance(raw_input, dict):
        question_in_input = raw_input.get("question")
        if isinstance(question_in_input, str) and question_in_input.strip():
            return normalize_question(question_in_input)

    if isinstance(raw_input, str) and raw_input.strip():
        return normalize_question(raw_input)

    question = state.get("question") or ""
    if isinstance(question, str) and question.strip():
        return normalize_question(question)

    return ""
