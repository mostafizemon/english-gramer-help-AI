from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel

from app.services.gemini_service import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiService,
    GeminiUnavailableError,
    gemini_service,
)
from app.services.groq_service import (
    GroqRequestError,
    GroqResponseError,
    GroqService,
    GroqUnavailableError,
    groq_service,
)


# ---------------------------------------------------------------------------
# Custom AI Router Exceptions
# ---------------------------------------------------------------------------


class AIRouterError(Exception):
    """Base exception for all AI router errors."""


class AIRouterRequestError(AIRouterError):
    """Raised when an invalid request or bad argument is encountered that should not be retried."""


class AIProvidersUnavailableError(AIRouterError):
    """Raised when all configured AI providers (primary and fallbacks) are unavailable or exhausted."""


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AIResult:
    text: str
    provider: str
    model: str


@dataclass(frozen=True)
class AIStructuredResult:
    data: Any
    provider: str
    model: str


# ---------------------------------------------------------------------------
# Central AI Provider Router
# ---------------------------------------------------------------------------


class AIRouter:
    def __init__(
        self,
        gemini: Optional[GeminiService] = None,
        groq: Optional[GroqService] = None,
    ) -> None:
        # Use singleton instances by default so credential cooldown/disabled states persist
        self.gemini_service: GeminiService = gemini or gemini_service
        self.groq_service: GroqService = groq or groq_service

    async def generate(self, prompt: str) -> AIResult:
        if not prompt or not prompt.strip():
            raise AIRouterRequestError("Prompt must not be empty.")

        # -------------------------------------------------------------------
        # 1. Primary Provider: Gemini
        # -------------------------------------------------------------------
        try:
            gemini_res = await self.gemini_service.generate(prompt)
            return AIResult(
                text=gemini_res.text,
                provider="gemini",
                model=gemini_res.model,
            )
        except GeminiRequestError as exc:
            # Client / bad request errors should not fallback as they will fail on all providers
            raise AIRouterRequestError(str(exc)) from exc
        except (GeminiUnavailableError, GeminiResponseError):
            # Gemini credentials exhausted, in cooldown, or returned invalid response -> fallback to Groq
            pass

        # -------------------------------------------------------------------
        # 2. Fallback Provider: Groq
        # -------------------------------------------------------------------
        try:
            groq_res = await self.groq_service.generate(prompt)
            return AIResult(
                text=groq_res.text,
                provider="groq",
                model=groq_res.model,
            )
        except GroqRequestError as exc:
            raise AIRouterRequestError(str(exc)) from exc
        except (GroqUnavailableError, GroqResponseError) as exc:
            raise AIProvidersUnavailableError(
                "All AI providers (Gemini and Groq) are currently unavailable or exhausted."
            ) from exc

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
    ) -> AIStructuredResult:
        if not prompt or not prompt.strip():
            raise AIRouterRequestError("Prompt must not be empty.")

        # -------------------------------------------------------------------
        # 1. Primary Provider: Gemini
        # -------------------------------------------------------------------
        try:
            gemini_res = await self.gemini_service.generate_structured(
                prompt=prompt,
                response_model=response_model,
            )
            return AIStructuredResult(
                data=gemini_res.data,
                provider="gemini",
                model=gemini_res.model,
            )
        except GeminiRequestError as exc:
            raise AIRouterRequestError(str(exc)) from exc
        except (GeminiUnavailableError, GeminiResponseError):
            # Gemini credentials exhausted, in cooldown, or failed structured validation -> fallback to Groq
            pass

        # -------------------------------------------------------------------
        # 2. Fallback Provider: Groq
        # -------------------------------------------------------------------
        try:
            groq_res = await self.groq_service.generate_structured(
                prompt=prompt,
                response_model=response_model,
            )
            return AIStructuredResult(
                data=groq_res.data,
                provider="groq",
                model=groq_res.model,
            )
        except GroqRequestError as exc:
            raise AIRouterRequestError(str(exc)) from exc
        except (GroqUnavailableError, GroqResponseError) as exc:
            raise AIProvidersUnavailableError(
                "All AI providers (Gemini and Groq) are currently unavailable or returned invalid structured outputs."
            ) from exc


ai_router = AIRouter()
