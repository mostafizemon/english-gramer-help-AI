from fastapi import APIRouter, HTTPException, status

from app.schemas.grammar import (
    GrammarAnalyzeRequest,
    GrammarAnalyzeResponse,
    WordAnalysisRequest,
    WordAnalysisResponse,
)
from app.services.grammar_service import (
    GrammarContentQualityError,
    GrammarInputError,
    GrammarProviderUnavailableError,
    GrammarResponseValidationError,
    GrammarServiceError,
    grammar_service,
)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=GrammarAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze English text (word or sentence)",
    description=(
        "Universal analysis endpoint that accepts a single English word or a full English "
        "sentence and provides deep pedagogical grammatical analysis for Bengali learners."
    ),
)
async def analyze_text(
    payload: GrammarAnalyzeRequest,
) -> GrammarAnalyzeResponse:
    try:
        return await grammar_service.analyze(payload.text)
    except GrammarInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (GrammarResponseValidationError, GrammarContentQualityError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except GrammarProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except GrammarServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during grammar analysis.",
        ) from exc


@router.post(
    "/word",
    response_model=WordAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an English word",
    description=(
        "Analyzes an English word and returns Bengali meanings, English senses, "
        "grammar roles, usage examples, contexts, synonyms, antonyms, idioms and "
        "phrasal verbs, verb forms, tense usage, common mistakes, conversation, "
        "quiz, related words, and a learning story."
    ),
)
async def analyze_word(
    payload: WordAnalysisRequest,
) -> WordAnalysisResponse:
    try:
        return await grammar_service.analyze_word(payload.word)
    except GrammarInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (GrammarResponseValidationError, GrammarContentQualityError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except GrammarProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except GrammarServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during grammar analysis.",
        ) from exc
