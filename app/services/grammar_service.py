import json
import logging
import re
from typing import Any, Literal, Optional

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
from app.services.grammar_engine import grammar_engine
from app.services.nlp_parser import (
    NLPParserError,
    NLPParserUnavailableError,
    nlp_parser,
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

    def _extract_deterministic_facts(
        self, sentence: str
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Extract low-level parser facts and high-level grammar facts deterministically,
        with graceful fallback if NLP parser or GrammarEngine is unavailable.
        """
        parser_facts: Optional[dict[str, Any]] = None
        grammar_facts: Optional[dict[str, Any]] = None

        try:
            parser_facts = nlp_parser.parse(sentence)
        except NLPParserUnavailableError as exc:
            logger.warning(
                "NLPParser unavailable, falling back to LLM-only sentence analysis: %s",
                exc,
            )
            return None, None
        except NLPParserError as exc:
            logger.warning(
                "NLPParser error on sentence '%s': %s. Continuing with LLM fallback.",
                sentence[:50],
                exc,
            )
            return None, None
        except Exception as exc:
            logger.error("Unexpected error in NLPParser on sentence: %s", exc)
            return None, None

        if parser_facts:
            try:
                grammar_facts = grammar_engine.analyze(parser_facts)
            except Exception as exc:
                logger.error(
                    "Unexpected error in GrammarEngine on sentence '%s': %s",
                    sentence[:50],
                    exc,
                )
                grammar_facts = None

        return parser_facts, grammar_facts

    def _reconcile_sentence_analysis(
        self,
        response: SentenceAnalysisResponse,
        parser_facts: Optional[dict[str, Any]],
        grammar_facts: Optional[dict[str, Any]],
    ) -> SentenceAnalysisResponse:
        """Reconcile LLM-generated fields with authoritative deterministic facts.
        Only overrides fields where deterministic facts are reliable and non-null.
        """
        if not parser_facts and not grammar_facts:
            return response

        updates: dict[str, Any] = {}

        if parser_facts:
            # 1. Subject
            if parser_facts.get("subjects"):
                updates["subject"] = parser_facts["subjects"]

            # 2. Main verbs
            if parser_facts.get("main_verbs"):
                updates["main_verbs"] = parser_facts["main_verbs"]

            # 3. Auxiliary verbs
            if "auxiliary_verbs" in parser_facts:
                updates["auxiliary_verbs"] = parser_facts["auxiliary_verbs"]

            # 4. Objects: combine direct_objects + indirect_objects (EXCLUDE objects_of_prepositions)
            direct_objs = parser_facts.get("direct_objects", [])
            indirect_objs = parser_facts.get("indirect_objects", [])
            combined_objs: list[str] = []
            seen: set[str] = set()
            for obj in direct_objs + indirect_objs:
                if obj and obj not in seen:
                    seen.add(obj)
                    combined_objs.append(obj)
            updates["objects"] = combined_objs

            # 5. Prepositions
            if "prepositions" in parser_facts:
                updates["prepositions"] = parser_facts["prepositions"]

            # 6. Determiners
            if "determiners" in parser_facts:
                updates["determiners"] = parser_facts["determiners"]

            # 7. Nouns
            if "nouns" in parser_facts:
                updates["nouns"] = parser_facts["nouns"]

            # 8. Pronouns
            if "pronouns" in parser_facts:
                updates["pronouns"] = parser_facts["pronouns"]

            # 9. Adjectives
            if "adjectives" in parser_facts:
                updates["adjectives"] = parser_facts["adjectives"]

            # 10. Adverbs
            if "adverbs" in parser_facts:
                updates["adverbs"] = parser_facts["adverbs"]

            # 11. Conjunctions
            if "conjunctions" in parser_facts:
                updates["conjunctions"] = parser_facts["conjunctions"]

        if grammar_facts:
            # 12. Tense: override ONLY when grammar_facts["tense"] is non-null
            if grammar_facts.get("tense"):
                updates["tense"] = grammar_facts["tense"]

            # 13. Voice: override ONLY when grammar_facts["voice"] is non-null
            if grammar_facts.get("voice"):
                updates["voice"] = grammar_facts["voice"]

        if not updates:
            return response

        # Reconstruct updated model safely
        updated_data = response.model_dump()
        updated_data.update(updates)
        return SentenceAnalysisResponse.model_validate(updated_data)

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
        """Analyze an English sentence and return structured syntax and pedagogical explanations
        combining deterministic NLP/Grammar facts with LLM teaching explanations.
        """
        cleaned_sentence = self._validate_sentence(sentence)

        # 1. Deterministic NLP & Grammar facts extraction
        parser_facts, grammar_facts = self._extract_deterministic_facts(
            cleaned_sentence
        )

        # 2. Build structured prompt with deterministic facts
        base_prompt = build_sentence_analysis_prompt(
            cleaned_sentence,
            parser_facts=parser_facts,
            grammar_facts=grammar_facts,
        )

        # 3. Primary generation attempt
        try:
            result = await self.ai_router.generate_structured(
                prompt=base_prompt,
                response_model=SentenceAnalysisResponse,
            )
            raw_data = (
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

        # 4. Authoritative deterministic field reconciliation
        data = self._reconcile_sentence_analysis(
            raw_data, parser_facts, grammar_facts
        )

        # 5. Semantic & structural quality validation
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
        structural_violations = [
            v for v in violations if "forbidden foreign-script" not in v
        ]

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
            raw_retry_data = (
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

        # Reconcile regenerated response
        retry_data = self._reconcile_sentence_analysis(
            raw_retry_data, parser_facts, grammar_facts
        )

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


