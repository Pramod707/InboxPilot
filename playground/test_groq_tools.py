from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModelSettings

from app import db
from app.config import settings

tool_agent = Agent(
    settings.agent_model,
    output_type=str,
    instructions=(
        "You are testing knowledge retrieval. "
        "Call get_section with section='brand-and-tone'. "
        "After receiving the result, summarize the brand tone in two sentences."
    ),
    model_settings=GroqModelSettings(
        groq_reasoning_format="hidden",
        groq_reasoning_effort="low",
    ),
)


@tool_agent.tool_plain
def get_section(section: str) -> str:
    """Retrieve entries from a knowledge-base section."""
    print(f"TOOL CALLED: get_section({section})")

    entries = db.get_section(section)

    if not entries:
        return f"No entries found for {section}"

    return "\n\n".join(
        f"## {entry['title']}\n{entry['content']}"
        for entry in entries
    )


if __name__ == "__main__":
    result = tool_agent.run_sync(
        "Retrieve the brand-and-tone section and summarize it."
    )

    print("\nFINAL RESPONSE:")
    print(result.output)