"""Automated unit and integration tests for AI Provider Router (app/services/ai_router.py).

Zero network calls, zero AI/LLM API calls ($0 cost).
Verifies primary provider dispatch, structured output typing, and failover behavior.
"""

import asyncio
from unittest.mock import AsyncMock
import pytest
from pydantic import BaseModel

from app.services.ai_router import (
    AIProvidersUnavailableError,
    AIRouter,
    AIRouterRequestError,
    AIResult,
    AIStructuredResult,
)
from app.services.gemini_service import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiResult,
    GeminiStructuredResult,
    GeminiUnavailableError,
)
from app.services.groq_service import (
    GroqRequestError,
    GroqResponseError,
    GroqResult,
    GroqStructuredResult,
    GroqUnavailableError,
)


class MockResponseModel(BaseModel):
    message: str
    code: int


# ===========================================================================
# Structured Generation Router Tests
# ===========================================================================


def test_gemini_structured_success():
    """TEST 1: Primary provider (Gemini) succeeds on structured generation."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        sample_model = MockResponseModel(message="Success from Gemini", code=200)
        mock_gemini.generate_structured.return_value = GeminiStructuredResult(
            data=sample_model,
            model="gemini-2.5-flash",
            credential_name="gemini_key_1",
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)
        result = await router.generate_structured("Analyze sentence", MockResponseModel)

        assert isinstance(result, AIStructuredResult)
        assert result.provider == "gemini"
        assert result.model == "gemini-2.5-flash"
        assert isinstance(result.data, MockResponseModel)
        assert result.data.message == "Success from Gemini"
        assert result.data.code == 200

        mock_gemini.generate_structured.assert_called_once()
        mock_groq.generate_structured.assert_not_called()

    asyncio.run(_run())


def test_gemini_unavailable_fallback_to_groq_structured():
    """TEST 2: Gemini unavailable triggers automatic fallback to Groq for structured generation."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate_structured.side_effect = GeminiUnavailableError(
            "All Gemini credentials in cooldown."
        )
        sample_model = MockResponseModel(message="Success from Groq", code=200)
        mock_groq.generate_structured.return_value = GroqStructuredResult(
            data=sample_model,
            model="llama-3.3-70b-versatile",
            credential_name="groq_key_1",
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)
        result = await router.generate_structured("Analyze sentence", MockResponseModel)

        assert isinstance(result, AIStructuredResult)
        assert result.provider == "groq"
        assert result.model == "llama-3.3-70b-versatile"
        assert isinstance(result.data, MockResponseModel)
        assert result.data.message == "Success from Groq"

        mock_gemini.generate_structured.assert_called_once()
        mock_groq.generate_structured.assert_called_once()

    asyncio.run(_run())


def test_gemini_response_error_fallback_to_groq_structured():
    """TEST 3: Gemini response validation error triggers fallback to Groq."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate_structured.side_effect = GeminiResponseError(
            "Gemini returned invalid structured output."
        )
        sample_model = MockResponseModel(message="Recovered via Groq", code=200)
        mock_groq.generate_structured.return_value = GroqStructuredResult(
            data=sample_model,
            model="llama-3.3-70b-versatile",
            credential_name="groq_key_1",
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)
        result = await router.generate_structured("Analyze sentence", MockResponseModel)

        assert result.provider == "groq"
        assert result.data.message == "Recovered via Groq"
        mock_gemini.generate_structured.assert_called_once()
        mock_groq.generate_structured.assert_called_once()

    asyncio.run(_run())


def test_gemini_request_error_does_not_fallback():
    """TEST 4: Client / bad request errors must NOT trigger fallback to Groq."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate_structured.side_effect = GeminiRequestError(
            "Malformed request payload."
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)

        with pytest.raises(AIRouterRequestError) as exc_info:
            await router.generate_structured("Invalid request", MockResponseModel)

        assert "Malformed request payload" in str(exc_info.value)
        mock_gemini.generate_structured.assert_called_once()
        mock_groq.generate_structured.assert_not_called()

    asyncio.run(_run())


def test_both_providers_unavailable_structured():
    """TEST 5: When all Gemini and Groq providers fail, AIProvidersUnavailableError is raised."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate_structured.side_effect = GeminiUnavailableError(
            "Gemini unavailable."
        )
        mock_groq.generate_structured.side_effect = GroqUnavailableError(
            "Groq unavailable."
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)

        with pytest.raises(AIProvidersUnavailableError) as exc_info:
            await router.generate_structured("Analyze sentence", MockResponseModel)

        assert "All AI providers" in str(exc_info.value)
        mock_gemini.generate_structured.assert_called_once()
        mock_groq.generate_structured.assert_called_once()

    asyncio.run(_run())


# ===========================================================================
# Raw Text Generation Router Tests
# ===========================================================================


def test_gemini_text_generate_success():
    """TEST 6: Primary text generation succeeds on Gemini."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate.return_value = GeminiResult(
            text="Grammar explanation text",
            model="gemini-2.5-flash",
            credential_name="gemini_key_1",
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)
        result = await router.generate("Explain tense")

        assert isinstance(result, AIResult)
        assert result.text == "Grammar explanation text"
        assert result.provider == "gemini"
        assert result.model == "gemini-2.5-flash"
        mock_gemini.generate.assert_called_once()
        mock_groq.generate.assert_not_called()

    asyncio.run(_run())


def test_gemini_unavailable_fallback_to_groq_text_generate():
    """TEST 7: Text generation falls back to Groq when Gemini is unavailable."""
    async def _run():
        mock_gemini = AsyncMock()
        mock_groq = AsyncMock()

        mock_gemini.generate.side_effect = GeminiUnavailableError("Gemini down.")
        mock_groq.generate.return_value = GroqResult(
            text="Groq explanation text",
            model="llama-3.3-70b-versatile",
            credential_name="groq_key_1",
        )

        router = AIRouter(gemini=mock_gemini, groq=mock_groq)
        result = await router.generate("Explain tense")

        assert result.text == "Groq explanation text"
        assert result.provider == "groq"
        mock_gemini.generate.assert_called_once()
        mock_groq.generate.assert_called_once()

    asyncio.run(_run())


def test_empty_prompt_validation():
    """TEST 8: Empty or whitespace prompts are rejected immediately."""
    async def _run():
        router = AIRouter()

        with pytest.raises(AIRouterRequestError):
            await router.generate("")

        with pytest.raises(AIRouterRequestError):
            await router.generate("   ")

        with pytest.raises(AIRouterRequestError):
            await router.generate_structured("", MockResponseModel)

        with pytest.raises(AIRouterRequestError):
            await router.generate_structured("   ", MockResponseModel)

    asyncio.run(_run())
