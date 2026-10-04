"""Automated tests for provider credential state machine, cooldowns, and failover.

Zero network calls, zero AI/LLM API calls ($0 cost).
Tests in-memory key disablement (401/403), short rate-limit cooldowns (429),
daily quota cooldowns, transient server errors (5xx/timeouts), and non-retryable 4xx errors.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from google.genai import errors as gerrors
from groq import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
import httpx

from app.services.gemini_service import (
    GeminiRequestError,
    GeminiResult,
    GeminiService,
    GeminiUnavailableError,
)
from app.services.groq_service import (
    GroqRequestError,
    GroqResult,
    GroqService,
    GroqUnavailableError,
)


# ===========================================================================
# Gemini Credential State Machine Tests
# ===========================================================================


def test_gemini_401_disables_key_and_rotates():
    """TEST 1: 401 Authentication error permanently disables key for process and uses next key."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1", "fake-key-2"])

        mock_client1 = MagicMock()
        mock_client1.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ClientError(
                401, {"error": {"message": "API key not valid"}}
            )
        )
        mock_client1.aio.aclose = AsyncMock()
        mock_client1.close = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.text = "Valid response from key 2"
        mock_client2 = MagicMock()
        mock_client2.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_client2.aio.aclose = AsyncMock()
        mock_client2.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Valid response from key 2"
        assert result.credential_name == "gemini_key_2"
        # Key 1 must be disabled permanently for the process
        assert service.credentials[0].enabled is False
        assert service.credentials[1].enabled is True

    asyncio.run(_run())


def test_gemini_429_rate_limit_sets_cooldown_and_rotates():
    """TEST 2: 429 Rate limit puts key into cooldown (enabled remains True) and uses next key."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1", "fake-key-2"])

        mock_client1 = MagicMock()
        mock_client1.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ClientError(
                429, {"error": {"message": "Resource exhausted. Rate limit exceeded"}}
            )
        )
        mock_client1.aio.aclose = AsyncMock()
        mock_client1.close = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.text = "Success on key 2"
        mock_client2 = MagicMock()
        mock_client2.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_client2.aio.aclose = AsyncMock()
        mock_client2.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Success on key 2"
        assert result.credential_name == "gemini_key_2"
        # Key 1 must remain enabled but have active cooldown
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_until is not None
        assert service.credentials[0].cooldown_reason == "rate_limit"

    asyncio.run(_run())


def test_gemini_short_rate_limit_cooldown_recovery():
    """TEST 3: Cooldown expires cleanly after elapsed time without permanent disablement."""
    service = GeminiService(api_keys=["fake-key-1"])
    cred = service.credentials[0]

    now = datetime.now(timezone.utc)
    # Set cooldown in the past
    cred.cooldown_until = now - timedelta(seconds=10)
    cred.cooldown_reason = "rate_limit"

    assert cred.is_available(now) is True


def test_gemini_daily_quota_long_cooldown():
    """TEST 4: Daily quota error sets cooldown_reason to 'daily_quota' with extended reset time."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1", "fake-key-2"])

        mock_client1 = MagicMock()
        mock_client1.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ClientError(
                429, {"error": {"message": "Daily limit exceeded. Requests per day exhausted."}}
            )
        )
        mock_client1.aio.aclose = AsyncMock()
        mock_client1.close = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.text = "Success on key 2"
        mock_client2 = MagicMock()
        mock_client2.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_client2.aio.aclose = AsyncMock()
        mock_client2.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client",
            side_effect=[mock_client1, mock_client2],
        ):
            await service.generate("Test prompt")

        assert service.credentials[0].cooldown_reason == "daily_quota"

    asyncio.run(_run())


def test_gemini_500_server_error_transient_cooldown():
    """TEST 5: 5xx server error sets transient cooldown without disabling key."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1", "fake-key-2"])

        mock_client1 = MagicMock()
        mock_client1.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ServerError(500, {"error": {"message": "Internal Server Error"}})
        )
        mock_client1.aio.aclose = AsyncMock()
        mock_client1.close = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.text = "Success after 500"
        mock_client2 = MagicMock()
        mock_client2.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_client2.aio.aclose = AsyncMock()
        mock_client2.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Success after 500"
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_reason == "transient_error"

    asyncio.run(_run())


def test_gemini_400_client_error_does_not_disable_keys():
    """TEST 6: Malformed 4xx client errors raise GeminiRequestError without disabling keys."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1"])

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ClientError(400, {"error": {"message": "Invalid argument format"}})
        )
        mock_client.aio.aclose = AsyncMock()
        mock_client.close = MagicMock()

        with patch("app.services.gemini_service.genai.Client", return_value=mock_client):
            with pytest.raises(GeminiRequestError) as exc_info:
                await service.generate("Test prompt")

        assert "Bad request sent to Gemini" in str(exc_info.value)
        # Credential should NOT be disabled or cooled down for client input errors
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_until is None

    asyncio.run(_run())


def test_gemini_all_keys_unavailable_raises_unavailable():
    """TEST 7: When all configured keys are in cooldown or disabled, raises GeminiUnavailableError."""
    async def _run():
        service = GeminiService(api_keys=["fake-key-1", "fake-key-2"])
        now = datetime.now(timezone.utc)
        service.credentials[0].cooldown_until = now + timedelta(seconds=600)
        service.credentials[1].enabled = False

        with pytest.raises(GeminiUnavailableError):
            await service.generate("Test prompt")

    asyncio.run(_run())


# ===========================================================================
# Groq Credential State Machine Tests
# ===========================================================================


def test_groq_401_disables_key_and_rotates():
    """TEST 8: Groq 401 AuthenticationError permanently disables failing key and rotates to next."""
    async def _run():
        service = GroqService(api_keys=["fake-groq-1", "fake-groq-2"])

        req = httpx.Request("POST", "https://api.groq.com")
        resp401 = httpx.Response(401, request=req)

        mock_client1 = MagicMock()
        mock_client1.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError(
                "Invalid API Key",
                response=resp401,
                body={"error": {"message": "Invalid API Key"}},
            )
        )
        mock_client1.close = AsyncMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq success text"
        mock_resp2 = MagicMock()
        mock_resp2.choices = [mock_choice]
        mock_client2 = MagicMock()
        mock_client2.chat.completions.create = AsyncMock(return_value=mock_resp2)
        mock_client2.close = AsyncMock()

        with patch(
            "app.services.groq_service.AsyncGroq",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Groq success text"
        assert result.credential_name == "groq_key_2"
        assert service.credentials[0].enabled is False
        assert service.credentials[1].enabled is True

    asyncio.run(_run())


def test_groq_429_rate_limit_sets_cooldown_and_rotates():
    """TEST 9: Groq 429 RateLimitError puts key into cooldown and rotates to next key."""
    async def _run():
        service = GroqService(api_keys=["fake-groq-1", "fake-groq-2"])

        req = httpx.Request("POST", "https://api.groq.com")
        resp429 = httpx.Response(429, request=req, headers={"retry-after": "120"})

        mock_client1 = MagicMock()
        mock_client1.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit exceeded",
                response=resp429,
                body={"error": {"message": "Rate limit exceeded"}},
            )
        )
        mock_client1.close = AsyncMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Success on Groq key 2"
        mock_resp2 = MagicMock()
        mock_resp2.choices = [mock_choice]
        mock_client2 = MagicMock()
        mock_client2.chat.completions.create = AsyncMock(return_value=mock_resp2)
        mock_client2.close = AsyncMock()

        with patch(
            "app.services.groq_service.AsyncGroq",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Success on Groq key 2"
        assert result.credential_name == "groq_key_2"
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_until is not None
        assert service.credentials[0].cooldown_reason == "rate_limit"

    asyncio.run(_run())


def test_groq_500_server_error_transient_cooldown():
    """TEST 10: Groq 500 InternalServerError sets transient cooldown without disabling key."""
    async def _run():
        service = GroqService(api_keys=["fake-groq-1", "fake-groq-2"])

        req = httpx.Request("POST", "https://api.groq.com")
        resp500 = httpx.Response(500, request=req)

        mock_client1 = MagicMock()
        mock_client1.chat.completions.create = AsyncMock(
            side_effect=InternalServerError(
                "Internal Server Error",
                response=resp500,
                body={"error": {"message": "Internal Server Error"}},
            )
        )
        mock_client1.close = AsyncMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Success after Groq 500"
        mock_resp2 = MagicMock()
        mock_resp2.choices = [mock_choice]
        mock_client2 = MagicMock()
        mock_client2.chat.completions.create = AsyncMock(return_value=mock_resp2)
        mock_client2.close = AsyncMock()

        with patch(
            "app.services.groq_service.AsyncGroq",
            side_effect=[mock_client1, mock_client2],
        ):
            result = await service.generate("Test prompt")

        assert result.text == "Success after Groq 500"
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_reason == "transient_error"

    asyncio.run(_run())


def test_groq_400_bad_request_does_not_disable_keys():
    """TEST 11: Groq 400 BadRequestError raises GroqRequestError without modifying key states."""
    async def _run():
        service = GroqService(api_keys=["fake-groq-1"])

        req = httpx.Request("POST", "https://api.groq.com")
        resp400 = httpx.Response(400, request=req)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=BadRequestError(
                "Invalid prompt payload",
                response=resp400,
                body={"error": {"message": "Invalid prompt payload"}},
            )
        )
        mock_client.close = AsyncMock()

        with patch("app.services.groq_service.AsyncGroq", return_value=mock_client):
            with pytest.raises(GroqRequestError) as exc_info:
                await service.generate("Test prompt")

        assert "Bad request sent to Groq" in str(exc_info.value)
        assert service.credentials[0].enabled is True
        assert service.credentials[0].cooldown_until is None

    asyncio.run(_run())


def test_groq_all_keys_unavailable_raises_unavailable():
    """TEST 12: When all Groq keys are exhausted/disabled, raises GroqUnavailableError."""
    async def _run():
        service = GroqService(api_keys=["fake-groq-1"])
        service.credentials[0].enabled = False

        with pytest.raises(GroqUnavailableError):
            await service.generate("Test prompt")

    asyncio.run(_run())


# ===========================================================================
# In-Memory State Retention Tests
# ===========================================================================


def test_in_memory_credential_state_persists_across_calls():
    """TEST 13: Disabled and cooldown states persist in-memory across subsequent requests."""
    async def _run():
        service = GeminiService(api_keys=["key-1", "key-2"])

        # 1. First call fails key 1 with 401 -> permanently disables key 1
        mock_c1 = MagicMock()
        mock_c1.aio.models.generate_content = AsyncMock(
            side_effect=gerrors.ClientError(401, {"error": {"message": "Unauthenticated"}})
        )
        mock_c1.aio.aclose = AsyncMock()
        mock_c1.close = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.text = "Key 2 success"
        mock_c2 = MagicMock()
        mock_c2.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_c2.aio.aclose = AsyncMock()
        mock_c2.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client", side_effect=[mock_c1, mock_c2]
        ):
            await service.generate("Prompt 1")

        assert service.credentials[0].enabled is False

        # 2. Second call should completely skip key 1 and directly use key 2
        mock_c2_again = MagicMock()
        mock_c2_again.aio.models.generate_content = AsyncMock(return_value=mock_resp2)
        mock_c2_again.aio.aclose = AsyncMock()
        mock_c2_again.close = MagicMock()

        with patch(
            "app.services.gemini_service.genai.Client", return_value=mock_c2_again
        ) as mock_client_cls:
            res2 = await service.generate("Prompt 2")
            assert res2.credential_name == "gemini_key_2"
            # genai.Client was called only ONCE (for key-2), not for disabled key-1
            assert mock_client_cls.call_count == 1

    asyncio.run(_run())
