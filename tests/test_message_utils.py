from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.query.message_utils import (
    normalize_question,
    content_to_text,
    messages_from_input_payload,
    latest_human_question,
    question_from_state,
)


def test_normalize_question():
    assert (
        normalize_question("   What  is   the   revenue?  ") == "What is the revenue?"
    )
    assert normalize_question("") == ""


def test_content_to_text():
    # String test
    assert content_to_text("  hello   world  ") == "hello world"

    # List of strings test
    assert content_to_text(["hello", "  ", "world"]) == "hello world"

    # List of dicts (standard OpenAI blocks)
    assert (
        content_to_text([{"text": "hello"}, "skipped", {"text": "world"}])
        == "hello skipped world"
    )

    # Invalid types
    assert content_to_text(42) == ""


def test_messages_from_input_payload():
    messages_list = [HumanMessage(content="test")]
    assert messages_from_input_payload(messages_list) == messages_list
    assert messages_from_input_payload({"messages": messages_list}) == messages_list
    assert messages_from_input_payload("invalid") == []


def test_latest_human_question():
    messages = [
        SystemMessage(content="You are an assistant"),
        HumanMessage(content="First question"),
        AIMessage(content="First answer"),
        HumanMessage(content="Second question"),
    ]
    assert latest_human_question(messages) == "Second question"

    # Test with dict formats (standard payload)
    messages_dict = [
        {"role": "user", "content": "Hello user"},
        {"role": "assistant", "content": "Hello back"},
    ]
    assert latest_human_question(messages_dict) == "Hello user"

    # Test with list/tuple pair
    messages_tuple = [
        ("user", "Hello tuple"),
    ]
    assert latest_human_question(messages_tuple) == "Hello tuple"


def test_question_from_state():
    # 1. messages state
    state1 = {"messages": [HumanMessage(content="What is net income?")]}
    assert question_from_state(state1) == "What is net income?"

    # 2. sub-payload input messages
    state2 = {"input": {"messages": [HumanMessage(content="Input question")]}}
    assert question_from_state(state2) == "Input question"

    # 3. input with "question" key
    state3 = {"input": {"question": "Raw question"}}
    assert question_from_state(state3) == "Raw question"

    # 4. input as string
    state4 = {"input": "Direct string question"}
    assert question_from_state(state4) == "Direct string question"

    # 5. directly on question key
    state5 = {"question": "Direct question key"}
    assert question_from_state(state5) == "Direct question key"

    # 6. Fallback empty
    assert question_from_state({}) == ""
