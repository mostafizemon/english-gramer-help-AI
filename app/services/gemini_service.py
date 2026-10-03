import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Optional

from google import genai
from google.genai import errors, types
import httpx
from pydantic import BaseModel

from app.core.config import settings

TRANSIENT_COOLDOWN_SECONDS: int = 60
RATE_LIMIT_COOLDOWN_SECONDS: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Custom Provider Exceptions
# ---------------------------------------------------------------------------


class GeminiServiceError(Exception):
    """Base exception for all Gemini service errors."""


class GeminiUnavailableError(GeminiServiceError):
    """Raised when all Gemini credentials are exhausted, disabled, or unavailable."""


class GeminiRequestError(GeminiServiceError):
    """Raised when a non-retryable bad request or client error is encountered."""


class GeminiResponseError(GeminiServiceError):
    """Raised when Gemini returns an empty or invalid response."""


# ---------------------------------------------------------------------------
# Result & Credential State Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeminiResult:
    text: str
    model: str
    credential_name: str


@dataclass(frozen=True)
class GeminiStructuredResult:
    data: Any
    model: str
    credential_name: str


@dataclass
class GeminiCredentialState:
    name: str
    api_key: str
    enabled: bool = True
    cooldown_until: Optional[datetime] = None
    cooldown_reason: Optional[str] = None
    last_error: Optional[str] = None

    def is_available(self, now: datetime) -> bool:
        """Check whether this credential can currently be used."""
        if not self.enabled:
            return False
        if self.cooldown_until is not None and self.cooldown_until > now:
            return False
        return True

    def __repr__(self) -> str:
        # Never expose the raw API key in representation or logs
        return (
            f"GeminiCredentialState(name={self.name!r}, enabled={self.enabled!r}, "
            f"cooldown_until={self.cooldown_until!r}, cooldown_reason={self.cooldown_reason!r}, "
            f"last_error={self.last_error!r})"
        )


# ---------------------------------------------------------------------------
# Helper Functions for Error Analysis
# ---------------------------------------------------------------------------


def parse_retry_delay(error: Any) -> Optional[float]:
    """Extract explicit retry delay in seconds from error details or message."""
    # 1. Check structured error details
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", details)
        if isinstance(inner, dict):
            detail_list = inner.get("details", [])
            if isinstance(detail_list, list):
                for item in detail_list:
                    if isinstance(item, dict):
                        delay = item.get("retryDelay") or item.get("retry_delay")
                        if delay:
                            if isinstance(delay, (int, float)):
                                return float(delay)
                            if isinstance(delay, str):
                                match = re.search(r"([\d.]+)\s*s?", delay)
                                if match:
                                    try:
                                        return float(match.group(1))
                                    except ValueError:
                                        pass

    # 2. Check message text for retry patterns
    msg = getattr(error, "message", "") or str(error)
    patterns = [
        r"retry\s+(?:after|in)\s+([\d.]+)\s*(?:s|sec|seconds?)?",
        r"retryDelay[\"']?\s*:\s*[\"']?([\d.]+)\s*s?",
        r"reset\s+in\s+([\d.]+)\s*(?:s|sec|seconds?)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:
                    return val
            except ValueError:
                pass

    return None


def is_daily_quota_error(error: Any) -> bool:
    """Determine if an error indicates daily/per-day quota exhaustion."""
    text_chunks: list[str] = []
    msg = getattr(error, "message", "")
    if msg:
        text_chunks.append(str(msg))

    details = getattr(error, "details", None)
    if details:
        text_chunks.append(str(details))

    full_text = " ".join(text_chunks).lower()

    daily_indicators = (
        "per day",
        "requests per day",
        "requests_per_day",
        "rpd",
        "daily quota",
        "daily limit",
        "daily_quota",
        "daily_limit",
        "per_day",
        "day limit",
        "generations per day",
    )
    return any(indicator in full_text for indicator in daily_indicators)


def get_next_daily_reset_time(now: datetime) -> datetime:
    """Calculate the next likely daily reset time (next UTC midnight), ensuring at least 1h."""
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if (next_midnight - now).total_seconds() < 3600:
        next_midnight += timedelta(days=1)
    return next_midnight


# ---------------------------------------------------------------------------
# Gemini Provider Service
# ---------------------------------------------------------------------------


class GeminiService:
    def __init__(
        self,
        model: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.model: str = model or settings.GEMINI_MODEL
        keys = api_keys if api_keys is not None else settings.gemini_api_keys
        self.timeout_seconds: int = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.AI_REQUEST_TIMEOUT_SECONDS
        )
        self.credentials: list[GeminiCredentialState] = [
            GeminiCredentialState(
                name=f"gemini_key_{i + 1}",
                api_key=key,
            )
            for i, key in enumerate(keys)
        ]

    def get_credential_states(self) -> list[dict[str, Any]]:
        """Return safe summaries of all credential states for diagnostics."""
        return [
            {
                "name": cred.name,
                "enabled": cred.enabled,
                "cooldown_until": (
                    cred.cooldown_until.isoformat()
                    if cred.cooldown_until
                    else None
                ),
                "cooldown_reason": cred.cooldown_reason,
                "last_error": cred.last_error,
            }
            for cred in self.credentials
        ]

    async def generate(self, prompt: str) -> GeminiResult:
        if not prompt or not prompt.strip():
            raise GeminiRequestError("Prompt must not be empty.")

        if not self.credentials:
            raise GeminiUnavailableError("No Gemini API keys are configured.")

        now = datetime.now(timezone.utc)
        eligible_credentials = [
            cred for cred in self.credentials if cred.is_available(now)
        ]

        if not eligible_credentials:
            all_disabled = all(not cred.enabled for cred in self.credentials)
            if all_disabled:
                raise GeminiUnavailableError(
                    "All Gemini API credentials have been disabled due to authentication/permission errors."
                )
            raise GeminiUnavailableError(
                "All Gemini API credentials are currently in cooldown."
            )

        for cred in eligible_credentials:
            now = datetime.now(timezone.utc)
            if not cred.is_available(now):
                continue

            client = genai.Client(
                api_key=cred.api_key,
                http_options=types.HttpOptions(
                    timeout=self.timeout_seconds * 1000,
                ),
            )

            config = types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )

            try:
                try:
                    response = await client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                finally:
                    try:
                        await client.aio.aclose()
                    except Exception:
                        pass
                    try:
                        client.close()
                    except Exception:
                        pass

                text = getattr(response, "text", None)
                if not text or not text.strip():
                    now = datetime.now(timezone.utc)
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "empty_response"
                    cred.last_error = "Empty response received from Gemini"
                    continue

                # Successful generation -> clear error and cooldown state
                cred.last_error = None
                cred.cooldown_until = None
                cred.cooldown_reason = None
                return GeminiResult(
                    text=text.strip(),
                    model=self.model,
                    credential_name=cred.name,
                )

            except errors.APIError as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "code", None)
                msg = (getattr(exc, "message", "") or "").lower()

                # Check for authentication / permission failures -> permanently disable for process
                is_auth_error = status_code in (401, 403) or (
                    status_code == 400
                    and any(
                        keyword in msg
                        for keyword in (
                            "api key",
                            "api_key_invalid",
                            "unauthenticated",
                            "permission",
                        )
                    )
                )

                if is_auth_error:
                    cred.enabled = False
                    cred.cooldown_until = None
                    cred.cooldown_reason = None
                    cred.last_error = (
                        f"Authentication/permission error (status: {status_code or 401})"
                    )
                    continue

                # Rate limit (429 / RESOURCE_EXHAUSTED)
                if status_code == 429 or any(
                    keyword in msg
                    for keyword in ("resource_exhausted", "quota", "rate limit")
                ):
                    retry_delay = parse_retry_delay(exc)
                    if is_daily_quota_error(exc):
                        if retry_delay is not None and retry_delay > 0:
                            cred.cooldown_until = now + timedelta(
                                seconds=retry_delay
                            )
                        else:
                            cred.cooldown_until = get_next_daily_reset_time(now)
                        cred.cooldown_reason = "daily_quota"
                        cred.last_error = "Daily quota exhausted"
                    else:
                        if retry_delay is not None and retry_delay > 0:
                            cred.cooldown_until = now + timedelta(
                                seconds=retry_delay
                            )
                        else:
                            cred.cooldown_until = now + timedelta(
                                seconds=RATE_LIMIT_COOLDOWN_SECONDS
                            )
                        cred.cooldown_reason = "rate_limit"
                        cred.last_error = "Rate limit exceeded (429)"
                    continue

                # Server error (5xx) -> transient cooldown
                if (
                    status_code is not None and 500 <= status_code < 600
                ) or isinstance(exc, errors.ServerError):
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "transient_error"
                    cred.last_error = f"Gemini server error (status: {status_code})"
                    continue

                # Non-retryable client errors (other 4xx)
                if (
                    status_code is not None and 400 <= status_code < 500
                ) or isinstance(exc, errors.ClientError):
                    raise GeminiRequestError(
                        f"Bad request sent to Gemini provider: {getattr(exc, 'message', 'Client error')}"
                    ) from exc

                # Other generic API errors -> transient cooldown
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = (
                    f"Gemini API error (status: {status_code}): {getattr(exc, 'message', 'Unknown error')}"
                )
                continue

            except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Request timed out"
                continue

            except (httpx.NetworkError, httpx.ConnectError, ConnectionError):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Network/connection error"
                continue

            except GeminiServiceError:
                raise

            except Exception as exc:
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Unexpected error: {type(exc).__name__}"
                continue

        raise GeminiUnavailableError(
            "All eligible Gemini API credentials failed to process the request."
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
    ) -> GeminiStructuredResult:
        if not prompt or not prompt.strip():
            raise GeminiRequestError("Prompt must not be empty.")

        if not self.credentials:
            raise GeminiUnavailableError("No Gemini API keys are configured.")

        now = datetime.now(timezone.utc)
        eligible_credentials = [
            cred for cred in self.credentials if cred.is_available(now)
        ]

        if not eligible_credentials:
            all_disabled = all(not cred.enabled for cred in self.credentials)
            if all_disabled:
                raise GeminiUnavailableError(
                    "All Gemini API credentials have been disabled due to authentication/permission errors."
                )
            raise GeminiUnavailableError(
                "All Gemini API credentials are currently in cooldown."
            )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_model,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        for cred in eligible_credentials:
            now = datetime.now(timezone.utc)
            if not cred.is_available(now):
                continue

            client = genai.Client(
                api_key=cred.api_key,
                http_options=types.HttpOptions(
                    timeout=self.timeout_seconds * 1000,
                ),
            )

            try:
                try:
                    response = await client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                finally:
                    try:
                        await client.aio.aclose()
                    except Exception:
                        pass
                    try:
                        client.close()
                    except Exception:
                        pass

                validated_data: Optional[BaseModel] = None

                parsed_val = getattr(response, "parsed", None)
                if isinstance(parsed_val, response_model):
                    validated_data = parsed_val
                elif isinstance(parsed_val, dict):
                    try:
                        validated_data = response_model.model_validate(parsed_val)
                    except Exception:
                        pass
                elif parsed_val is not None:
                    try:
                        validated_data = response_model.model_validate(parsed_val)
                    except Exception:
                        pass

                if validated_data is None:
                    text = getattr(response, "text", None)
                    if not text or not text.strip():
                        now = datetime.now(timezone.utc)
                        cred.cooldown_until = now + timedelta(
                            seconds=TRANSIENT_COOLDOWN_SECONDS
                        )
                        cred.cooldown_reason = "empty_response"
                        cred.last_error = "Empty response received from Gemini"
                        continue

                    try:
                        validated_data = response_model.model_validate_json(
                            text.strip()
                        )
                    except Exception as val_exc:
                        now = datetime.now(timezone.utc)
                        cred.cooldown_until = now + timedelta(
                            seconds=TRANSIENT_COOLDOWN_SECONDS
                        )
                        cred.cooldown_reason = "invalid_structured_response"
                        cred.last_error = f"Structured validation error: {val_exc}"
                        raise GeminiResponseError(
                            f"Gemini structured output failed schema validation: {val_exc}"
                        ) from val_exc

                # Successful generation -> clear error and cooldown state
                cred.last_error = None
                cred.cooldown_until = None
                cred.cooldown_reason = None
                return GeminiStructuredResult(
                    data=validated_data,
                    model=self.model,
                    credential_name=cred.name,
                )

            except errors.APIError as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "code", None)
                msg = (getattr(exc, "message", "") or "").lower()

                # Check for authentication / permission failures -> permanently disable for process
                is_auth_error = status_code in (401, 403) or (
                    status_code == 400
                    and any(
                        keyword in msg
                        for keyword in (
                            "api key",
                            "api_key_invalid",
                            "unauthenticated",
                            "permission",
                        )
                    )
                )

                if is_auth_error:
                    cred.enabled = False
                    cred.cooldown_until = None
                    cred.cooldown_reason = None
                    cred.last_error = (
                        f"Authentication/permission error (status: {status_code or 401})"
                    )
                    continue

                # Rate limit (429 / RESOURCE_EXHAUSTED)
                if status_code == 429 or any(
                    keyword in msg
                    for keyword in ("resource_exhausted", "quota", "rate limit")
                ):
                    retry_delay = parse_retry_delay(exc)
                    if is_daily_quota_error(exc):
                        if retry_delay is not None and retry_delay > 0:
                            cred.cooldown_until = now + timedelta(
                                seconds=retry_delay
                            )
                        else:
                            cred.cooldown_until = get_next_daily_reset_time(now)
                        cred.cooldown_reason = "daily_quota"
                        cred.last_error = "Daily quota exhausted"
                    else:
                        if retry_delay is not None and retry_delay > 0:
                            cred.cooldown_until = now + timedelta(
                                seconds=retry_delay
                            )
                        else:
                            cred.cooldown_until = now + timedelta(
                                seconds=RATE_LIMIT_COOLDOWN_SECONDS
                            )
                        cred.cooldown_reason = "rate_limit"
                        cred.last_error = "Rate limit exceeded (429)"
                    continue

                # Server error (5xx) -> transient cooldown
                if (
                    status_code is not None and 500 <= status_code < 600
                ) or isinstance(exc, errors.ServerError):
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "transient_error"
                    cred.last_error = f"Gemini server error (status: {status_code})"
                    continue

                # Non-retryable client errors (other 4xx)
                if (
                    status_code is not None and 400 <= status_code < 500
                ) or isinstance(exc, errors.ClientError):
                    raise GeminiRequestError(
                        f"Bad request sent to Gemini provider: {getattr(exc, 'message', 'Client error')}"
                    ) from exc

                # Other generic API errors -> transient cooldown
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = (
                    f"Gemini API error (status: {status_code}): {getattr(exc, 'message', 'Unknown error')}"
                )
                continue

            except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Request timed out"
                continue

            except (httpx.NetworkError, httpx.ConnectError, ConnectionError):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Network/connection error"
                continue

            except GeminiServiceError:
                raise

            except Exception as exc:
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Unexpected error: {type(exc).__name__}"
                continue

        raise GeminiUnavailableError(
            "All eligible Gemini API credentials failed to process the request."
        )


gemini_service = GeminiService()
