
from fastapi import FastAPI

app = FastAPI(title="Ai customer Support")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
