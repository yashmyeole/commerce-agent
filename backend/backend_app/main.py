from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Commerce Agent Demo Backend")

class HealthOut(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthOut)
async def root():
    return HealthOut(status="ok", message="Commerce Agent Backend is running")

# simple echo endpoint to test POSTs
class EchoIn(BaseModel):
    text: str

class EchoOut(BaseModel):
    text: str

@app.post("/api/echo", response_model=EchoOut)
async def echo(payload: EchoIn):
    return EchoOut(text=payload.text)
