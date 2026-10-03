import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Optional

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    AuthenticationError,
    BadRequestError,
    GroqError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
import httpx
from pydantic import BaseModel

from app.core.config import settings

TRANSIENT_COOLDOWN_SECONDS: int = 60
RATE_LIMIT_COOLDOWN_SECONDS: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Custom Provider Exceptions
# ---------------------------------------------------------------------------


class GroqServiceError(Exception):
    """Base exception for all Groq service errors."""


class GroqUnavailableError(GroqServiceError):
    """Raised when all Groq credentials are exhausted, disabled, or unavailable."""


class GroqRequestError(GroqServiceError):
    """Raised when a non-retryable bad request or client error is encountered."""


class GroqResponseError(GroqServiceError):
    """Raised when Groq returns an empty or invalid response."""


# ---------------------------------------------------------------------------
# Result & Credential State Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroqResult:
    text: str
    model: str
    credential_name: str


@dataclass(frozen=True)
class GroqStructuredResult:
    data: Any
    model: str
    credential_name: str


@dataclass
class GroqCredentialState:
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
            f"GroqCredentialState(name={self.name!r}, enabled={self.enabled!r}, "
            f"cooldown_until={self.cooldown_until!r}, cooldown_reason={self.cooldown_reason!r}, "
            f"last_error={self.last_error!r})"
        )


# ---------------------------------------------------------------------------
# Helper Functions for Error Analysis
# ---------------------------------------------------------------------------


def parse_retry_delay(error: Any) -> Optional[float]:
    """Extract explicit retry delay in seconds from error response headers or message."""
    # 1. Check response headers (e.g. retry-after, x-ratelimit-reset-requests, etc.)
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers:
            for header_key in (
                "retry-after",
                "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens",
            ):
                header_val = headers.get(header_key)
                if header_val:
                    try:
                        val = float(header_val)
                        if val > 0:
                            return val
                    except ValueError:
                        pass

    # 2. Check message text for retry patterns
    msg = getattr(error, "message", "") or str(error)
    patterns = [
        r"try\s+again\s+in\s+([\d.]+)\s*(?:s|sec|seconds?)?",
        r"retry\s+(?:after|in)\s+([\d.]+)\s*(?:s|sec|seconds?)?",
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


def _to_strict_json_schema(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Recursively enforce strict JSON schema requirements for Groq structured output."""
    schema = copy.deepcopy(schema_dict)

    def _process(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for v in node.values():
                _process(v)
        elif isinstance(node, list):
            for item in node:
                _process(item)

    _process(schema)
    return schema


# ---------------------------------------------------------------------------
# Groq Provider Service
# ---------------------------------------------------------------------------


class GroqService:
    def __init__(
        self,
        model: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.model: str = model or settings.GROQ_MODEL
        keys = api_keys if api_keys is not None else settings.groq_api_keys
        self.timeout_seconds: int = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.AI_REQUEST_TIMEOUT_SECONDS
        )
        self.credentials: list[GroqCredentialState] = [
            GroqCredentialState(
                name=f"groq_key_{i + 1}",
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

    async def generate(self, prompt: str) -> GroqResult:
        if not prompt or not prompt.strip():
            raise GroqRequestError("Prompt must not be empty.")

        if not self.credentials:
            raise GroqUnavailableError("No Groq API keys are configured.")

        now = datetime.now(timezone.utc)
        eligible_credentials = [
            cred for cred in self.credentials if cred.is_available(now)
        ]

        if not eligible_credentials:
            all_disabled = all(not cred.enabled for cred in self.credentials)
            if all_disabled:
                raise GroqUnavailableError(
                    "All Groq API credentials have been disabled due to authentication/permission errors."
                )
            raise GroqUnavailableError(
                "All Groq API credentials are currently in cooldown."
            )

        for cred in eligible_credentials:
            now = datetime.now(timezone.utc)
            if not cred.is_available(now):
                continue

            client = AsyncGroq(
                api_key=cred.api_key,
                timeout=float(self.timeout_seconds),
            )

            try:
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass

                text: Optional[str] = None
                if response and response.choices:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        text = choice.message.content

                if not text or not text.strip():
                    now = datetime.now(timezone.utc)
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "empty_response"
                    cred.last_error = "Empty response received from Groq"
                    continue

                # Successful generation -> clear error and cooldown state
                cred.last_error = None
                cred.cooldown_until = None
                cred.cooldown_reason = None
                return GroqResult(
                    text=text.strip(),
                    model=self.model,
                    credential_name=cred.name,
                )

            except (AuthenticationError, PermissionDeniedError) as exc:
                cred.enabled = False
                cred.cooldown_until = None
                cred.cooldown_reason = None
                status_code = getattr(exc, "status_code", 401)
                cred.last_error = (
                    f"Authentication/permission error (status: {status_code})"
                )
                continue

            except RateLimitError as exc:
                now = datetime.now(timezone.utc)
                retry_delay = parse_retry_delay(exc)
                if retry_delay is not None and retry_delay > 0:
                    cred.cooldown_until = now + timedelta(seconds=retry_delay)
                else:
                    cred.cooldown_until = now + timedelta(
                        seconds=RATE_LIMIT_COOLDOWN_SECONDS
                    )
                cred.cooldown_reason = "rate_limit"
                cred.last_error = "Rate limit exceeded (429)"
                continue

            except (
                BadRequestError,
                NotFoundError,
                UnprocessableEntityError,
            ) as exc:
                # Non-retryable client errors -> raise immediately without key cycling
                raise GroqRequestError(
                    f"Bad request sent to Groq provider: {getattr(exc, 'message', 'Client error')}"
                ) from exc

            except InternalServerError as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "status_code", 500)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Groq server error (status: {status_code})"
                continue

            except (APIStatusError, APIError) as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "status_code", None)

                # Check if it's 401 / 403
                if status_code in (401, 403):
                    cred.enabled = False
                    cred.cooldown_until = None
                    cred.cooldown_reason = None
                    cred.last_error = (
                        f"Authentication/permission error (status: {status_code})"
                    )
                    continue

                # Check if it's 429
                if status_code == 429:
                    retry_delay = parse_retry_delay(exc)
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

                # Server error (5xx)
                if status_code is not None and 500 <= status_code < 600:
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "transient_error"
                    cred.last_error = (
                        f"Groq server error (status: {status_code})"
                    )
                    continue

                # Non-retryable client errors (other 4xx)
                if status_code is not None and 400 <= status_code < 500:
                    raise GroqRequestError(
                        f"Bad request sent to Groq provider: {getattr(exc, 'message', 'Client error')}"
                    ) from exc

                # Generic API error -> transient cooldown
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = (
                    f"Groq API error (status: {status_code}): {getattr(exc, 'message', 'Unknown error')}"
                )
                continue

            except (
                APITimeoutError,
                httpx.TimeoutException,
                asyncio.TimeoutError,
                TimeoutError,
            ):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Request timed out"
                continue

            except (
                APIConnectionError,
                httpx.NetworkError,
                httpx.ConnectError,
                ConnectionError,
            ):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Network/connection error"
                continue

            except GroqServiceError:
                raise

            except Exception as exc:
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Unexpected error: {type(exc).__name__}"
                continue

        raise GroqUnavailableError(
            "All eligible Groq API credentials failed to process the request."
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
    ) -> GroqStructuredResult:
        if not prompt or not prompt.strip():
            raise GroqRequestError("Prompt must not be empty.")

        if not self.credentials:
            raise GroqUnavailableError("No Groq API keys are configured.")

        now = datetime.now(timezone.utc)
        eligible_credentials = [
            cred for cred in self.credentials if cred.is_available(now)
        ]

        if not eligible_credentials:
            all_disabled = all(not cred.enabled for cred in self.credentials)
            if all_disabled:
                raise GroqUnavailableError(
                    "All Groq API credentials have been disabled due to authentication/permission errors."
                )
            raise GroqUnavailableError(
                "All Groq API credentials are currently in cooldown."
            )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": _to_strict_json_schema(response_model.model_json_schema()),
            },
        }

        for cred in eligible_credentials:
            now = datetime.now(timezone.utc)
            if not cred.is_available(now):
                continue

            client = AsyncGroq(
                api_key=cred.api_key,
                timeout=float(self.timeout_seconds),
            )

            try:
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        response_format=response_format,
                        max_completion_tokens=4096,
                    )
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass

                text: Optional[str] = None
                if response and response.choices:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        text = choice.message.content

                if not text or not text.strip():
                    now = datetime.now(timezone.utc)
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "empty_response"
                    cred.last_error = "Empty response received from Groq"
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
                    raise GroqResponseError(
                        f"Groq structured output failed schema validation: {val_exc}"
                    ) from val_exc

                # Successful generation -> clear error and cooldown state
                cred.last_error = None
                cred.cooldown_until = None
                cred.cooldown_reason = None
                return GroqStructuredResult(
                    data=validated_data,
                    model=self.model,
                    credential_name=cred.name,
                )

            except (AuthenticationError, PermissionDeniedError) as exc:
                cred.enabled = False
                cred.cooldown_until = None
                cred.cooldown_reason = None
                status_code = getattr(exc, "status_code", 401)
                cred.last_error = (
                    f"Authentication/permission error (status: {status_code})"
                )
                continue

            except RateLimitError as exc:
                now = datetime.now(timezone.utc)
                retry_delay = parse_retry_delay(exc)
                if retry_delay is not None and retry_delay > 0:
                    cred.cooldown_until = now + timedelta(seconds=retry_delay)
                else:
                    cred.cooldown_until = now + timedelta(
                        seconds=RATE_LIMIT_COOLDOWN_SECONDS
                    )
                cred.cooldown_reason = "rate_limit"
                cred.last_error = "Rate limit exceeded (429)"
                continue

            except (
                BadRequestError,
                NotFoundError,
                UnprocessableEntityError,
            ) as exc:
                # Non-retryable client errors -> raise immediately without key cycling
                raise GroqRequestError(
                    f"Bad request sent to Groq provider: {getattr(exc, 'message', 'Client error')}"
                ) from exc

            except InternalServerError as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "status_code", 500)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Groq server error (status: {status_code})"
                continue

            except (APIStatusError, APIError) as exc:
                now = datetime.now(timezone.utc)
                status_code = getattr(exc, "status_code", None)

                # Check if it's 401 / 403
                if status_code in (401, 403):
                    cred.enabled = False
                    cred.cooldown_until = None
                    cred.cooldown_reason = None
                    cred.last_error = (
                        f"Authentication/permission error (status: {status_code})"
                    )
                    continue

                # Check if it's 429
                if status_code == 429:
                    retry_delay = parse_retry_delay(exc)
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

                # Server error (5xx)
                if status_code is not None and 500 <= status_code < 600:
                    cred.cooldown_until = now + timedelta(
                        seconds=TRANSIENT_COOLDOWN_SECONDS
                    )
                    cred.cooldown_reason = "transient_error"
                    cred.last_error = (
                        f"Groq server error (status: {status_code})"
                    )
                    continue

                # Non-retryable client errors (other 4xx)
                if status_code is not None and 400 <= status_code < 500:
                    raise GroqRequestError(
                        f"Bad request sent to Groq provider: {getattr(exc, 'message', 'Client error')}"
                    ) from exc

                # Generic API error -> transient cooldown
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = (
                    f"Groq API error (status: {status_code}): {getattr(exc, 'message', 'Unknown error')}"
                )
                continue

            except (
                APITimeoutError,
                httpx.TimeoutException,
                asyncio.TimeoutError,
                TimeoutError,
            ):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Request timed out"
                continue

            except (
                APIConnectionError,
                httpx.NetworkError,
                httpx.ConnectError,
                ConnectionError,
            ):
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = "Network/connection error"
                continue

            except GroqServiceError:
                raise

            except Exception as exc:
                now = datetime.now(timezone.utc)
                cred.cooldown_until = now + timedelta(
                    seconds=TRANSIENT_COOLDOWN_SECONDS
                )
                cred.cooldown_reason = "transient_error"
                cred.last_error = f"Unexpected error: {type(exc).__name__}"
                continue

        raise GroqUnavailableError(
            "All eligible Groq API credentials failed to process the request."
        )


groq_service = GroqService()
