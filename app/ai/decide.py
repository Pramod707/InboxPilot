import logging
import re

from app.ai.agent import write_reply
from app.ai.classifier import classify
from app.ai.rules import CUSTOMER_STEPS, resolve_issue
from app.ai.schemas import Decision

logger = logging.getLogger(__name__)


def contains_action_promise(reply: str) -> bool:
    """Detect commitments that must be handled by a human."""

    patterns = [
        r"\bwe\s+(can|will|shall|'ll)\s+"
        r"(send|ship|replace|refund|dispatch|issue|process|arrange)\b",
        r"\b(we|i)\s+(will|can|'ll)\s+"
        r"(take care of|handle|arrange|organize)\b",
        r"\b(once|after)\s+.{0,80}\b"
        r"(we can|we will|we'll)\s+"
        r"(send|ship|replace|refund|process|dispatch)\b",
        r"\b(to arrange|to process|to issue)\s+"
        r"(a |the )?(replacement|refund|shipment)\b",
        r"\b(your|the)\s+"
        r"(replacement|refund|shipment)\s+"
        r"(is|has been|will be)\s+"
        r"(being arranged|arranged|processed|shipped|dispatched)\b",
    ]

    return any(re.search(pattern, reply, re.IGNORECASE) for pattern in patterns)


def decide(thread: str, has_attachments: bool = False) -> Decision:
    """
    Run the AI steps for one email thread.
    The models propose; this function makes the final decision.
    """

    classification = classify(thread)

    def escalate(
        reason: str,
        proposed_action: str | None = None,
    ) -> Decision:
        return Decision(
            action="escalate",
            section=classification.section,
            reply=classification.acknowledgement,
            reason=reason,
            proposed_action=proposed_action,
        )

    # 1. Escalate emails containing attachments.
    if has_attachments:
        return escalate("The email has attachments")

    # 2. Follow classifier escalation rules.
    if classification.needs_human:
        return escalate(classification.reason)

    # 3. Generate a proposed customer response.
    answer = write_reply(thread, classification.section)

    logger.info(
        "Agent structured output:\n%s",
        answer.model_dump_json(indent=2),
    )

    # 4. Respect the agent's escalation decision.
    if answer.action == "escalate":
        return escalate(answer.reasoning)

    # 5. Escalate if the knowledge base lacks sufficient information.
    if not answer.enough_information:
        return escalate("The knowledge base does not answer this")

    # 6. Resolve reported product issues.
    if answer.issue:
        step = resolve_issue(answer.issue)

        if step not in CUSTOMER_STEPS:
            return escalate(
                f"Next step: {step}",
                proposed_action=step,
            )

    # 7. Check both the agent's flag and the actual generated response.
    if answer.promises_an_action or contains_action_promise(answer.reply):
        return escalate("The generated reply contains a prohibited action commitment.")

    # 8. Approve the response only after all safety checks pass.
    return Decision(
        action="reply",
        section=classification.section,
        reply=answer.reply,
        reason=answer.reasoning,
    )
