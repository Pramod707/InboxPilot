import importlib

import pytest

from app.ai.schemas import AgentReply, Classification

decide_module = importlib.import_module("app.ai.decide")


# --------------------------------------------------
# Test data helpers
# --------------------------------------------------


def make_classification(
    needs_human: bool = False,
) -> Classification:
    return Classification(
        section="shelving",
        language="English",
        needs_human=needs_human,
        reason="Customer reported a damaged shelving panel.",
        acknowledgement=(
            "Thank you for reaching out. A colleague will review your enquiry."
        ),
    )


def make_answer(
    action: str = "reply",
    enough_information: bool = True,
    promises_an_action: bool = False,
    reply: str = "Thank you for contacting Harbor and Pine.",
) -> AgentReply:
    return AgentReply(
        action=action,
        reply=reply,
        reasoning="Based on the shelving knowledge base.",
        enough_information=enough_information,
        promises_an_action=promises_an_action,
        issue=None,
    )


def mock_pipeline(monkeypatch, answer):
    """Replace classifier and LLM calls with predictable test data."""

    classification = make_classification()

    monkeypatch.setattr(
        decide_module,
        "classify",
        lambda thread: classification,
    )

    monkeypatch.setattr(
        decide_module,
        "write_reply",
        lambda thread, section: answer,
    )

    return classification


# --------------------------------------------------
# 1. Action-promise detection
# --------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "We can send a replacement panel.",
        "We will issue a refund.",
        "We will dispatch your order.",
        "Your replacement is being arranged.",
        "A colleague will follow up within one business day.",
        "A team member will reply to you within one business day.",
        "We will contact you tomorrow.",
        "To send you a replacement panel, please provide a photo.",
        "This helps us confirm the part number and process the shipment.",
    ],
)
def test_detects_action_promises(reply):
    assert decide_module.contains_action_promise(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        "Thank you for contacting Harbor and Pine.",
        "Please send us a photo of the damaged panel.",
        "We appreciate your patience.",
        "Could you share your order number?",
    ],
)
def test_allows_ordinary_customer_service_language(reply):
    assert decide_module.contains_action_promise(reply) is False


# --------------------------------------------------
# 2. Attachment escalation
# --------------------------------------------------


def test_escalates_when_email_has_attachments(monkeypatch):
    classification = make_classification()

    monkeypatch.setattr(
        decide_module,
        "classify",
        lambda thread: classification,
    )

    def unexpected_agent_call(*args, **kwargs):
        pytest.fail("Agent should not run when attachments exist")

    monkeypatch.setattr(
        decide_module,
        "write_reply",
        unexpected_agent_call,
    )

    result = decide_module.decide(
        "Customer email",
        has_attachments=True,
    )

    assert result.action == "escalate"
    assert result.reply == classification.acknowledgement


# --------------------------------------------------
# 3. Human escalation rules
# --------------------------------------------------


def test_escalates_when_classifier_requires_human(monkeypatch):
    classification = make_classification(needs_human=True)

    monkeypatch.setattr(
        decide_module,
        "classify",
        lambda thread: classification,
    )

    def unexpected_agent_call(*args, **kwargs):
        pytest.fail("Agent should not run when human escalation is required")

    monkeypatch.setattr(
        decide_module,
        "write_reply",
        unexpected_agent_call,
    )

    result = decide_module.decide("Customer email")

    assert result.action == "escalate"
    assert result.reason == classification.reason


# --------------------------------------------------
# 4. Agent escalation
# --------------------------------------------------


def test_escalates_when_agent_requests_escalation(monkeypatch):
    mock_pipeline(
        monkeypatch,
        make_answer(action="escalate"),
    )

    result = decide_module.decide("Customer email")

    assert result.action == "escalate"


# --------------------------------------------------
# 5. Missing knowledge-base information
# --------------------------------------------------


def test_escalates_when_information_is_insufficient(monkeypatch):
    mock_pipeline(
        monkeypatch,
        make_answer(enough_information=False),
    )

    result = decide_module.decide("Customer email")

    assert result.action == "escalate"
    assert result.reason == "The knowledge base does not answer this"


# --------------------------------------------------
# 6. Model-detected action promises
# --------------------------------------------------


def test_escalates_when_agent_flags_action_promise(monkeypatch):
    mock_pipeline(
        monkeypatch,
        make_answer(promises_an_action=True),
    )

    result = decide_module.decide("Customer email")

    assert result.action == "escalate"


# --------------------------------------------------
# 7. Regex-detected action promises
# --------------------------------------------------


def test_escalates_when_reply_contains_action_promise(monkeypatch):
    mock_pipeline(
        monkeypatch,
        make_answer(reply=("A colleague will reply to you within one business day.")),
    )

    result = decide_module.decide("Customer email")

    assert result.action == "escalate"
    assert (
        result.reason == "The generated reply contains a prohibited action commitment."
    )


# --------------------------------------------------
# 8. Safe customer reply
# --------------------------------------------------


def test_allows_safe_customer_reply(monkeypatch):
    mock_pipeline(
        monkeypatch,
        make_answer(
            reply=(
                "Thank you for contacting Harbor and Pine. "
                "Could you please share your order number?"
            )
        ),
    )

    result = decide_module.decide("Customer email")

    assert result.action == "reply"
    assert "order number" in result.reply
    assert result.section == "shelving"
