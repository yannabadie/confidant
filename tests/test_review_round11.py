"""Round 11, 29 October 2026: two zero-context reviews of the submission as entered on Devpost (a judge who read the
page only, and a skeptic who checked every claim against this code, with no network).

The skeptic found that a plural weekday ("What happens on Tuesdays?") or a weekday inside a word ("Tuesday-ish")
was not read as a time, so private memories were picked by their words, and full-text search stems "Tuesdays" to
"Tuesday": the request listed only the private item held on a Tuesday, which dated it, the bug of round 5. He also
found dates and times in bare digits that only a private value's digit signature masked ("Anything at 10 30?",
"05112026"), health facts kept as personal because an everyday word stood next to an unlisted medical word
("methadone ... before the gym", "Dr Okafor about palpitations after my run"), an agent's own label ignored when it
said what the fact was ("doctor visit"), and credentials stored instead of refused ("the wifi at home is
BlueCat2026", "my Netflix password's Sonnenblume42", "my phone unlocks with 4821", "the PIN on my bank card reads
4821"). The wifi one was then sent, since it was kept as personal.
"""
from __future__ import annotations

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.assistant import masked_calendar
from confidant.mcp_server import MCPServer
from confidant.proof import prove
from confidant.rules import classify_new_fact, names_time
from confidant.scan import detect_secret

from conftest import MODELS, PERSONA, FakeLLM


def seed() -> MemoryStore:
    store = MemoryStore()
    store.load_seed(PERSONA)
    return store


def sent_requests(store: MemoryStore, message: str) -> list[str]:
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": "ok"})
    Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(message, session_id="s")
    return [m["content"] for call in llm.calls for m in call["messages"] if m["role"] == "user"]


WEEKDAY_FORMS = [
    "What happens on Tuesdays?",
    "What happens on Thursdays?",
    "Anything Tuesday-ish?",
    "Any plans for Mondays?",
    "What's usually on Tuesday's agenda?",
    "Qu'est-ce que j'ai les mardis ?",
    "Was steht dienstags an?",
]


@pytest.mark.parametrize("question", WEEKDAY_FORMS)
def test_a_weekday_in_any_form_is_read_as_a_time(question: str) -> None:
    assert names_time(question)


@pytest.mark.parametrize("question", WEEKDAY_FORMS)
def test_a_weekday_in_any_form_does_not_depend_on_private_items(question: str) -> None:
    proof = prove(seed(), question)
    assert proof.identical, (proof.original, proof.moved)


def test_a_plural_weekday_gets_the_whole_private_schedule_not_the_items_held_that_day() -> None:
    (tuesdays,) = [text for text in sent_requests(seed(), "What happens on Tuesdays?") if "Request:" in text]
    (thursdays,) = [text for text in sent_requests(seed(), "What happens on Thursdays?") if "Request:" in text]
    def withheld(request: str) -> list[str]:  # shared memories may differ: they are picked by their own words
        return [line for line in request.splitlines() if line.startswith("⟦")]

    assert withheld(tuesdays) == withheld(thursdays) == [
        "⟦m7⟧ (private: health appointment, value withheld)", "⟦m11⟧ (private: private appointment, value withheld)"]


@pytest.mark.parametrize("typed, sent", [
    ("Anything at 10 30?", "Anything at [time]?"),
    ("Remind me what 05112026 holds.", "Remind me what [date] holds."),
    ("Remind me what 05122026 holds.", "Remind me what [date] holds."),
    ("Remind me what 11052026 holds.", "Remind me what [date] holds."),
    ("Can we do 1030 instead?", "Can we do [time] instead?"),
    ("Can we do 1035 instead?", "Can we do [time] instead?"),
    ("Let's move it to 1100.", "Let's move it to [time]."),
    ("Order 05112026 shipped.", "Order [date] shipped."),  # masked by its shape: harmless, and no oracle
])
def test_more_dates_and_times_in_bare_digits_are_masked_by_their_shape_alone(typed: str, sent: str) -> None:
    assert masked_calendar(typed) == sent


@pytest.mark.parametrize("question", ["Anything at 10 30?", "Remind me what 05112026 holds.",
                                      "Can we do 1030 instead?", "Let's move it to 1100."])
def test_bare_digit_dates_and_times_do_not_depend_on_private_items(question: str) -> None:
    proof = prove(seed(), question)
    assert proof.identical, (proof.original, proof.moved)


@pytest.mark.parametrize("fact", [
    "Remember that I pick up my methadone every Friday before the gym.",
    "Remember that I'm seeing Dr Okafor about palpitations after my run on Monday 9 Nov at 8:30.",
    "Remember that my insulin pens arrive on Friday, before my run.",
    "Remember that I take sertraline every morning before the gym.",
    "Remember that I have a vaccination on Friday, then dinner with friends.",
])
def test_health_facts_next_to_an_everyday_word_stay_private(fact: str) -> None:
    assert classify_new_fact(fact) == "private"


def test_an_agents_label_that_says_what_the_fact_is_makes_it_private() -> None:
    store = seed()
    server = MCPServer(Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), FakeLLM({}), models=MODELS))
    response = server.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "memory_record", "arguments": {
            "text": "Seeing Okafor after my run on Monday 9 Nov at 8:30", "kind": "event",
            "sensitivity": "personal", "label": "doctor visit"}},
    })
    result = response["result"]
    memory = result.get("structuredContent", result)["memory"]
    assert memory.get("withheld") is True
    assert store.get(memory["id"]).sensitivity == "private"


@pytest.mark.parametrize("message", [
    "Remember that the wifi at home is BlueCat2026.",
    "Remember that my Netflix password's Sonnenblume42.",
    "Remember that my phone unlocks with 4821.",
    "Remember that the PIN on my bank card reads 4821.",
    "Remember that the wifi password at the cottage: Lavande-77!",
])
def test_these_credentials_are_refused(message: str) -> None:
    assert detect_secret(message)


@pytest.mark.parametrize("message", [
    "Remember that the wifi at home is down again.",
    "Remember that the wifi at home is slow after 20:00.",
    "Remember that my phone unlocks with my face.",
    "Remember that the password's strength meter is too strict.",
    "Remember that the PIN pad on the office printer reads cards badly.",
])
def test_everyday_sentences_about_wifi_phones_and_pins_are_not_refused(message: str) -> None:
    assert detect_secret(message) is None


def test_a_refused_wifi_password_is_never_sent() -> None:
    store = seed()
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": "ok"})
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS)
    assistant.handle("Remember that the wifi at home is BlueCat2026.", session_id="s")
    assistant.handle("What's the wifi at home?", session_id="s")
    sent = "\n".join(m["content"] for call in llm.calls for m in call["messages"])
    assert "BlueCat2026" not in sent
    assert all("BlueCat2026" not in memory.text for memory in store.all_current())
