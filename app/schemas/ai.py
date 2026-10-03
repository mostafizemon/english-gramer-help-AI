from pydantic import BaseModel, Field


class AITestRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Test message to send to the AI provider",
    )


# Backwards compatibility aliases
GeminiTestRequest = AITestRequest
GroqTestRequest = AITestRequest
AIRouterTestRequest = AITestRequest


class GeminiTestResponse(BaseModel):
    success: bool
    provider: str
    model: str
    response: str


class GroqTestResponse(BaseModel):
    success: bool
    provider: str
    model: str
    response: str


class AIRouterTestResponse(BaseModel):
    success: bool
    provider: str
    model: str
    response: str
