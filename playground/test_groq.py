from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModelSettings
from app.config import settings

test_agent = Agent(
    settings.agent_model,
    output_type=str,
    model_settings=GroqModelSettings(
        groq_reasoning_format="hidden",
        groq_reasoning_effort="low",
    ),
)

result = test_agent.run_sync(
    "Reply with exactly this sentence: InboxPilot model test successful."
)

print("Model:", settings.agent_model)
print("Response:", result.output)
