from fastapi import APIRouter, HTTPException, status

from app.schemas.ai import (
    AITestRequest,
    AIRouterTestResponse,
    GeminiTestResponse,
    GroqTestResponse,
)
from app.services.ai_router import (
    AIProvidersUnavailableError,
    AIRouterRequestError,
    ai_router,
)
from app.services.gemini_service import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    gemini_service,
)
from app.services.groq_service import (
    GroqRequestError,
    GroqResponseError,
    GroqUnavailableError,
    groq_service,
)

router = APIRouter()


@router.post(
    "/gemini-test",
    response_model=GeminiTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test Gemini provider connectivity",
)
async def test_gemini_connection(
    request: AITestRequest,
) -> GeminiTestResponse:
    try:
        result = await gemini_service.generate(request.message)
        return GeminiTestResponse(
            success=True,
            provider="gemini",
            model=result.model,
            response=result.text,
        )
    except GeminiRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except GeminiResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/groq-test",
    response_model=GroqTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test Groq provider connectivity",
)
async def test_groq_connection(
    request: AITestRequest,
) -> GroqTestResponse:
    try:
        result = await groq_service.generate(request.message)
        return GroqTestResponse(
            success=True,
            provider="groq",
            model=result.model,
            response=result.text,
        )
    except GroqRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except GroqResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except GroqUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/router-test",
    response_model=AIRouterTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test AI provider router with Gemini-first and Groq-fallback",
)
async def test_ai_router(
    request: AITestRequest,
) -> AIRouterTestResponse:
    try:
        result = await ai_router.generate(request.message)
        return AIRouterTestResponse(
            success=True,
            provider=result.provider,
            model=result.model,
            response=result.text,
        )
    except AIRouterRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AIProvidersUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
