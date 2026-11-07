from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    google_client_id: str
    google_client_secret: str
    google_refresh_token: str
    database_url: str

    classifier_model: str = "groq:openai/gpt-oss-20b"
    agent_model: str = "groq:openai/gpt-oss-120b"

    poll_interval_seconds: int = 60
    dashboard_username: str = "admin"
    dashboard_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
