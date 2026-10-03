import json
import logging
import re
from typing import Literal, Optional

from pydantic import ValidationError

from app.prompts.sentence_analysis import build_sentence_analysis_prompt
from app.prompts.word_analysis import build_word_analysis_prompt
from app.schemas.grammar import (
    GrammarAnalyzeResponse,
    SentenceAnalysisResponse,
    WordAnalysisResponse,
)
from app.services.ai_router import (
    AIProvidersUnavailableError,
    AIRouter,
    AIRouterError,
    AIRouterRequestError,
    ai_router,
)
from app.services.quality_validator import (
    validate_bangla_text_quality,
    validate_sentence_analysis_quality,
    validate_word_analysis_quality,
)

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:['’\-][A-Za-z]+)*$")
MAX_WORD_LENGTH: int = 50
MAX_SENTENCE_LENGTH: int = 2000


# ---------------------------------------------------------------------------
# Custom Grammar Service Exceptions
# ---------------------------------------------------------------------------


class GrammarServiceError(Exception):
    """Base exception for all grammar service errors."""


class GrammarInputError(GrammarServiceError):
    """Raised when the user provides invalid word or sentence input."""


class GrammarResponseValidationError(GrammarServiceError):
    """Raised when the AI response cannot be parsed or validated as a response model."""


class GrammarContentQualityError(GrammarResponseValidationError):
    """Raised when the AI response fails semantic content quality validation (e.g. foreign script contamination)."""


class GrammarProviderUnavailableError(GrammarServiceError):
    """Raised when all underlying AI providers are unavailable."""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def detect_input_type(text: str) -> Literal["word", "sentence"]:
    """Deterministically classify input text as either 'word' or 'sentence'."""
    cleaned = text.strip()
    if not cleaned:
        return "word"

    tokens = [tok.strip() for tok in cleaned.split() if tok.strip()]
    if len(tokens) <= 1:
        return "word"

    return "sentence"


def _clean_json_text(text: str) -> str:
    """Clean markdown code blocks and extract raw JSON content."""
    cleaned = text.strip()

    # Remove leading markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)

    # Remove trailing markdown code fences
    if cleaned.endswith("```"):
        cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()

    # Extract JSON object substring if surrounding commentary exists
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    return cleaned


def _build_repair_prompt(word: str, invalid_text: str, error_details: str) -> str:
    """Build a targeted prompt to repair an invalid JSON response."""
    return f"""You previously generated a response for the English word "{word}" that failed validation.

Validation Error:
{error_details}

Invalid Response:
{invalid_text}

TASK:
Repair the response into a STRICTLY VALID raw JSON object matching the required WordAnalysisResponse schema.
- Retain all linguistic and grammar learning data for "{word}".
- Ensure all required fields exist and match expected types.
- Do NOT output markdown code fences (no ```json or ```).
- Do NOT output any conversational text or commentary. Output RAW JSON ONLY.
"""


# ---------------------------------------------------------------------------
# Grammar Service
# ---------------------------------------------------------------------------


class GrammarService:
    def __init__(self, router: Optional[AIRouter] = None) -> None:
        self.ai_router: AIRouter = router or ai_router

    def _validate_word(self, word: str) -> str:
        """Validate that input is a single valid English word."""
        if not word:
            raise GrammarInputError("Input word cannot be empty.")

        cleaned = word.strip()
        if not cleaned:
            raise GrammarInputError("Input word cannot be empty or whitespace.")

        if len(cleaned) > MAX_WORD_LENGTH:
            raise GrammarInputError(
                f"Word exceeds maximum length of {MAX_WORD_LENGTH} characters."
            )

        # Reject sentences or non-word tokens (spaces, multiple words, numbers, sentence punctuation)
        if not WORD_PATTERN.match(cleaned):
            raise GrammarInputError(
                f"'{cleaned}' is not a valid single English word. Please enter one word (e.g. 'run', 'well-known', \"don't\")."
            )

        return cleaned

    def _validate_sentence(self, sentence: str) -> str:
        """Validate that input is a valid non-empty sentence within limits."""
        if not sentence:
            raise GrammarInputError("Input sentence cannot be empty.")

        cleaned = sentence.strip()
        if not cleaned:
            raise GrammarInputError("Input sentence cannot be empty or whitespace.")

        if len(cleaned) > MAX_SENTENCE_LENGTH:
            raise GrammarInputError(
                f"Sentence exceeds maximum length of {MAX_SENTENCE_LENGTH} characters."
            )

        return cleaned

    def _parse_and_validate(self, raw_text: str) -> WordAnalysisResponse:
        """Parse raw AI text into WordAnalysisResponse."""
        cleaned_json = _clean_json_text(raw_text)
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON syntax: {exc.msg}") from exc

        if not isinstance(data, dict):
            raise ValueError("Expected top-level JSON object, got non-dict.")

        return WordAnalysisResponse.model_validate(data)

    async def analyze_word(self, word: str) -> WordAnalysisResponse:
        """Analyze an English word and return structured pedagogical grammar data."""
        cleaned_word = self._validate_word(word)
        base_prompt = build_word_analysis_prompt(cleaned_word)

        # Primary generation attempt
        try:
            result = await self.ai_router.generate_structured(
                prompt=base_prompt,
                response_model=WordAnalysisResponse,
            )
            data = (
                result.data
                if isinstance(result.data, WordAnalysisResponse)
                else WordAnalysisResponse.model_validate(result.data)
            )
        except AIRouterRequestError as exc:
            raise GrammarInputError(str(exc)) from exc
        except AIProvidersUnavailableError as exc:
            raise GrammarProviderUnavailableError(str(exc)) from exc
        except AIRouterError as exc:
            raise GrammarProviderUnavailableError(
                "AI routing failed to complete word analysis."
            ) from exc

        # Semantic content quality validation
        violations = validate_word_analysis_quality(data)
        if not violations:
            return data

        # Make at most ONE regeneration attempt with explicit script purity reinforcement
        logger.warning(
            "Word analysis for '%s' failed semantic quality check (%d violations): %s. Attempting 1 regeneration.",
            cleaned_word,
            len(violations),
            "; ".join(violations),
        )

        reinforcement = (
            "\n\nCRITICAL SCRIPT PURITY REQUIREMENT:\n"
            "Your previous response was rejected because it contained forbidden foreign-script characters (such as Devanagari/Hindi, Arabic, or Urdu) inside Bangla fields.\n"
            "ALL Bangla translations, explanations, and meanings MUST contain ONLY pure Bengali script, necessary English grammar terminology in Latin script, and standard punctuation.\n"
            "Never insert Devanagari, Hindi, Arabic, or Urdu characters anywhere in Bangla fields."
        )
        retry_prompt = base_prompt + reinforcement

        try:
            retry_result = await self.ai_router.generate_structured(
                prompt=retry_prompt,
                response_model=WordAnalysisResponse,
            )
            retry_data = (
                retry_result.data
                if isinstance(retry_result.data, WordAnalysisResponse)
                else WordAnalysisResponse.model_validate(retry_result.data)
            )
        except AIRouterRequestError as exc:
            raise GrammarInputError(str(exc)) from exc
        except AIProvidersUnavailableError as exc:
            raise GrammarProviderUnavailableError(str(exc)) from exc
        except AIRouterError as exc:
            raise GrammarResponseValidationError(
                "Regeneration after semantic quality check failure was unsuccessful."
            ) from exc

        retry_violations = validate_word_analysis_quality(retry_data)
        if retry_violations:
            logger.error(
                "Word analysis for '%s' failed semantic quality validation after regeneration: %s",
                cleaned_word,
                "; ".join(retry_violations),
            )
            raise GrammarContentQualityError(
                f"Generated word analysis for '{cleaned_word}' failed content quality validation due to foreign-script contamination."
            )

        return retry_data

    async def analyze_sentence(self, sentence: str) -> SentenceAnalysisResponse:
        """Analyze an English sentence and return structured syntax and pedagogical explanations."""
        cleaned_sentence = self._validate_sentence(sentence)
        base_prompt = build_sentence_analysis_prompt(cleaned_sentence)

        # Primary generation attempt
        try:
            result = await self.ai_router.generate_structured(
                prompt=base_prompt,
                response_model=SentenceAnalysisResponse,
            )
            data = (
                result.data
                if isinstance(result.data, SentenceAnalysisResponse)
                else SentenceAnalysisResponse.model_validate(result.data)
            )
        except AIRouterRequestError as exc:
            raise GrammarInputError(str(exc)) from exc
        except AIProvidersUnavailableError as exc:
            raise GrammarProviderUnavailableError(str(exc)) from exc
        except AIRouterError as exc:
            raise GrammarProviderUnavailableError(
                "AI routing failed to complete sentence analysis."
            ) from exc

        # Semantic & structural quality validation
        violations = validate_sentence_analysis_quality(data)
        if not violations:
            return data

        # Make at most ONE regeneration attempt with targeted quality/structural reinforcement
        logger.warning(
            "Sentence analysis failed quality check (%d violations): %s. Attempting 1 regeneration.",
            len(violations),
            "; ".join(violations),
        )

        script_violations = [v for v in violations if "forbidden foreign-script" in v]
        structural_violations = [v for v in violations if "forbidden foreign-script" not in v]

        reinforcement_sections: list[str] = []

        if script_violations:
            reinforcement_sections.append(
                "CRITICAL SCRIPT PURITY REQUIREMENT:\n"
                "Your previous response was rejected because it contained forbidden foreign-script characters (such as Devanagari/Hindi, Arabic, or Urdu) inside Bangla fields.\n"
                "ALL Bangla translations and explanations MUST contain ONLY pure Bengali script, necessary English grammar terminology in Latin script, and standard punctuation.\n"
                "Never insert Devanagari, Hindi, Arabic, or Urdu characters anywhere in Bangla fields."
            )

        if structural_violations:
            issues_summary = "\n".join(f"- {v}" for v in structural_violations)
            reinforcement_sections.append(
                f"CRITICAL STRUCTURAL CONSISTENCY REQUIREMENTS (PHRASES & CLAUSES):\n"
                f"Your previous response had the following structural inconsistencies:\n"
                f"{issues_summary}\n\n"
                f"MANDATORY FIXES:\n"
                f"1. 'phrases' must contain ONLY actual phrase types (e.g. Noun Phrase, Verb Phrase, Prepositional Phrase, Adjective Phrase, Adverb Phrase). NEVER place clause types (e.g. Relative Clause, Conditional Clause, Independent Clause, Dependent Clause) in 'phrases'.\n"
                f"2. 'phrases.function' must NEVER identify the item as a clause (e.g. not 'Independent Clause', 'Dependent Clause', 'Conditional Clause', 'Result Clause', 'Relative Clause', 'Main Clause').\n"
                f"3. For 'Verb Phrase', the text must contain ONLY the verbal group (e.g. 'would have attended', 'has been working', 'was completed'). Do NOT include the subject (e.g. not 'She has been working'), and do NOT include direct objects, complements, or prepositional phrases (e.g. not 'would have attended it', not 'had known about the meeting'). Separate them into their own phrase/POS entries.\n"
                f"4. Clauses belong strictly in the 'clauses' list."
            )

        reinforcement = "\n\n" + "\n\n".join(reinforcement_sections)
        retry_prompt = base_prompt + reinforcement

        try:
            retry_result = await self.ai_router.generate_structured(
                prompt=retry_prompt,
                response_model=SentenceAnalysisResponse,
            )
            retry_data = (
                retry_result.data
                if isinstance(retry_result.data, SentenceAnalysisResponse)
                else SentenceAnalysisResponse.model_validate(retry_result.data)
            )
        except AIRouterRequestError as exc:
            raise GrammarInputError(str(exc)) from exc
        except AIProvidersUnavailableError as exc:
            raise GrammarProviderUnavailableError(str(exc)) from exc
        except AIRouterError as exc:
            raise GrammarResponseValidationError(
                "Regeneration after semantic quality check failure was unsuccessful."
            ) from exc

        retry_violations = validate_sentence_analysis_quality(retry_data)
        if retry_violations:
            logger.error(
                "Sentence analysis failed quality validation after regeneration: %s",
                "; ".join(retry_violations),
            )
            raise GrammarContentQualityError(
                f"Generated sentence analysis failed content quality / structural validation: {'; '.join(retry_violations)}"
            )

        return retry_data

    async def analyze(self, text: str) -> GrammarAnalyzeResponse:
        """Universal analysis method for either a single word or a full sentence."""
        if not text or not text.strip():
            raise GrammarInputError("Input text cannot be empty.")

        cleaned = text.strip()
        input_type = detect_input_type(cleaned)

        if input_type == "word":
            word_candidate = cleaned.strip(".,!?;:\"' ")
            if not word_candidate:
                word_candidate = cleaned
            word_res = await self.analyze_word(word_candidate)
            return GrammarAnalyzeResponse(
                input_type="word",
                analysis=word_res,
            )
        else:
            sentence_res = await self.analyze_sentence(cleaned)
            return GrammarAnalyzeResponse(
                input_type="sentence",
                analysis=sentence_res,
            )


grammar_service = GrammarService()

