from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModelSettings

from app.ai.schemas import AgentReply
from app.config import settings

agent = Agent(
    settings.agent_model,
    output_type=AgentReply,
    instructions=(
        "Return a valid AgentReply. "
        "Do not call external tools. "
        "Return all required fields."
    ),
    model_settings=GroqModelSettings(
        groq_reasoning_format="hidden",
        groq_reasoning_effort="low",
    ),
)

if __name__ == "__main__":
    result = agent.run_sync(
        """
    A customer reports a cracked shelf panel.
    They provided order number 12345 and part number HP-S3-04.

    Return a valid AgentReply.

    IMPORTANT: Use these exact enum values:
    - action: "escalate"
    - issue.product: "shelving"
    - issue.problem: "damaged-panels"

    Other fields:
    - reply: A short acknowledgement to the customer.
    - reasoning: The damaged panel requires human review.
    - enough_information: true
    - promises_an_action: false
    - issue.order_number: "12345"
    - issue.part_numbers: ["HP-S3-04"]

    Do not invent enum values or change their spelling.
    """
    )

print("STRUCTURED OUTPUT SUCCESSFUL")
print(result.output.model_dump_json(indent=2))
