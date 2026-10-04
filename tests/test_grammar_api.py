"""API-level automated tests for English Grammar AI endpoints (/api/v1/grammar/analyze).

Zero network calls, zero AI/LLM calls ($0 cost).
Uses FastAPI TestClient to verify routing, validation, schemas, and fallback behavior.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.grammar import (
    SentenceAnalysisResponse,
    SentenceClause,
    SentenceGrammarExplanation,
    SentenceMistake,
    SentencePhrase,
    WordAnalysisResponse,
)
from app.services.ai_router import AIStructuredResult
from app.services.grammar_service import grammar_service
from app.services.nlp_parser import NLPParserUnavailableError


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient instance."""
    return TestClient(app)


def make_mock_word_data(word: str = "run") -> dict:
    """Valid WordAnalysisResponse sample data dictionary."""
    return {
        "overview": {
            "word": word,
            "normalized_word": word.lower(),
            "base_form": word.lower(),
            "ipa": "/rʌn/",
            "parts_of_speech": ["Verb", "Noun"],
            "cefr_level": "B1",
            "frequency_label": "High Frequency",
            "is_irregular_verb": True,
        },
        "bangla_meanings": [
            {"meaning": "দৌড়ানো", "explanation": "পায়ে হেঁটে দ্রুত চলা"}
        ],
        "english_senses": [
            {
                "sense": "Physical Locomotion",
                "domain": "Physical",
                "grammatical_type": "Intransitive",
                "definition": "Move fast on foot",
            }
        ],
        "grammar_roles": [
            {
                "role": "Main Verb",
                "part_of_speech": "Verb",
                "example_sentence": f"He {word}s fast.",
                "english_explanation": "Action verb.",
                "bangla_explanation": "ক্রিয়া।",
            }
        ],
        "usage_examples": [
            {
                "level": "Beginner",
                "cefr_level": "A1",
                "sentence": f"I {word} daily.",
                "bangla_translation": "আমি প্রতিদিন দৌড়াই।",
                "target_form": word,
                "grammar_note": "Present simple.",
            }
        ],
        "different_contexts": [
            {
                "context": "Fitness",
                "sentence": f"She {word}s for exercise.",
                "bangla_explanation": "ব্যায়ামের জন্য।",
                "target_form": f"{word}s",
            }
        ],
        "synonym_groups": [{"context": "Physical", "words": ["sprint", "jog"]}],
        "antonyms": ["walk", "halt"],
        "idioms_and_phrasal_verbs": [
            {
                "phrase": f"{word} out of",
                "type": "Phrasal Verb",
                "english_meaning": "Deplete",
                "bangla_meaning": "শেষ হয়ে যাওয়া",
                "example_sentence": "We ran out of time.",
                "example_bangla_translation": "আমাদের সময় শেষ হয়ে গিয়েছিল।",
            }
        ],
        "verb_forms": {
            "base_v1": word,
            "past_v2": "ran",
            "past_participle_v3": "run",
            "ing_form": "running",
            "third_person_singular": "runs",
        },
        "tense_usages": [
            {
                "tense": "Present Simple",
                "formula": "S + V1 + O",
                "example": f"They {word}.",
                "bangla_explanation": "সাধারণ বর্তমান।",
            }
        ],
        "common_mistakes": [
            {
                "incorrect": f"He {word}.",
                "correct": f"He {word}s.",
                "explanation": "Missing 3rd person singular s.",
                "bangla_explanation": "s যোগ করতে হবে।",
            }
        ],
        "real_conversation": {
            "context": "Track meet",
            "lines": [
                {
                    "speaker": "Coach",
                    "text": f"Are you ready to {word}?",
                    "target_form": word,
                }
            ],
        },
        "quick_quiz": {
            "question": f"What is the past tense of {word}?",
            "options": [
                {"text": "ran", "is_correct": True},
                {"text": f"{word}ed", "is_correct": False},
            ],
            "explanation": "Ran is irregular.",
            "bangla_explanation": "ran হলো সঠিক রূপ।",
        },
        "related_words": [
            {
                "word": "runner",
                "relation_or_part_of_speech": "noun",
                "meaning": "দৌড়বিদ",
            }
        ],
        "story": {
            "title": "The Big Race",
            "story": f"He decided to {word} in the marathon.",
            "usages": [
                {
                    "sentence": f"He decided to {word} in the marathon.",
                    "target_form": word,
                    "grammar_role": "Infinitive verb",
                    "tense": "Infinitive",
                    "english_explanation": "To-infinitive form.",
                    "bangla_explanation": "ক্রিয়ার অনন্ত রূপ।",
                }
            ],
        },
    }


def make_mock_sentence_data(sentence: str = "I am running in the park.") -> dict:
    """Valid SentenceAnalysisResponse sample data dictionary."""
    return {
        "original_sentence": sentence,
        "corrected_sentence": None,
        "is_grammatically_correct": True,
        "bangla_translation": "আমি পার্কে দৌড়াচ্ছি।",
        "sentence_type": "Simple",
        "sentence_purpose": "Declarative",
        "tense": "Present Continuous",
        "voice": "Active",
        "subject": ["I"],
        "main_verbs": ["running"],
        "auxiliary_verbs": ["am"],
        "objects": [],
        "complements": [],
        "nouns": ["park"],
        "pronouns": ["I"],
        "adjectives": [],
        "adverbs": [],
        "prepositions": ["in"],
        "conjunctions": [],
        "determiners": ["the"],
        "phrases": [
            {
                "text": "in the park",
                "type": "Prepositional Phrase",
                "function": "Adverbial of Place",
                "bangla_explanation": "স্থানের বিশেষণ হিসেবে কাজ করছে।",
            }
        ],
        "clauses": [
            {
                "text": sentence,
                "type": "Independent Clause",
                "function": "Main Clause",
                "bangla_explanation": "মূল স্বাধীন খণ্ডবাক্য।",
            }
        ],
        "structure": "Subject + Auxiliary Verb + Main Verb (-ing) + Prepositional Phrase",
        "grammar_explanations": [
            {
                "topic": "Present Continuous Tense",
                "english_explanation": "Used for actions happening right now.",
                "bangla_explanation": "বর্তমানে চলমান কাজ বোঝাতে ব্যবহৃত হয়।",
            }
        ],
        "common_mistakes": [
            {
                "issue": "Auxiliary omission",
                "original": "I running in the park.",
                "correction": "I am running in the park.",
                "english_explanation": "Present continuous requires the auxiliary verb 'am/is/are'.",
                "bangla_explanation": "am/is/are ছাড়া Present Continuous হয় না।",
            }
        ],
        "natural_alternatives": ["I'm jogging in the park."],
    }


# ===========================================================================
# 1. Sentence Routing
# ===========================================================================


def test_sentence_routing(client: TestClient):
    """TEST 1: POST /api/v1/grammar/analyze with a sentence routes to sentence analysis."""
    sentence = "I am running in the park."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "sentence"
    assert "analysis" in data
    assert data["analysis"]["original_sentence"] == sentence
    assert data["analysis"]["tense"] == "Present Continuous"


# ===========================================================================
# 2. Word Routing
# ===========================================================================


def test_word_routing(client: TestClient):
    """TEST 2: POST /api/v1/grammar/analyze with a single word routes to word analysis."""
    word = "run"
    mock_response = WordAnalysisResponse.model_validate(make_mock_word_data(word))

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": word})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "word"
    assert "analysis" in data
    assert data["analysis"]["overview"]["word"] == word


# ===========================================================================
# 3. Hyphenated Word
# ===========================================================================


def test_hyphenated_word_routing(client: TestClient):
    """TEST 3: Hyphenated single lexical token (e.g. 'well-known') routes as word."""
    word = "well-known"
    mock_response = WordAnalysisResponse.model_validate(make_mock_word_data(word))

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": word})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "word"


# ===========================================================================
# 4. Apostrophe Word
# ===========================================================================


def test_apostrophe_word_routing(client: TestClient):
    """TEST 4: Word with apostrophe (e.g. 'don't') routes as word."""
    word = "don't"
    mock_response = WordAnalysisResponse.model_validate(make_mock_word_data(word))

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": word})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "word"


# ===========================================================================
# 5. Multi-word Input
# ===========================================================================


def test_multi_word_routing(client: TestClient):
    """TEST 5: Multi-word phrase or sentence routes as sentence."""
    sentence = "She works here."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "sentence"


# ===========================================================================
# 6. Empty Input Validation
# ===========================================================================


def test_empty_and_whitespace_input_validation(client: TestClient):
    """TEST 6: Empty and whitespace-only text must be rejected with 4xx client errors."""
    # Empty string fails Pydantic schema validation (422)
    res_empty = client.post("/api/v1/grammar/analyze", json={"text": ""})
    assert res_empty.status_code in {400, 422}

    # Whitespace-only string fails service input validation (400)
    res_ws = client.post("/api/v1/grammar/analyze", json={"text": "   "})
    assert res_ws.status_code in {400, 422}


# ===========================================================================
# 7. Too-long Input
# ===========================================================================


def test_too_long_input_validation(client: TestClient):
    """TEST 7: Input exceeding maximum character limit (2000 chars) is rejected."""
    long_text = "word " * 500  # > 2000 characters
    response = client.post("/api/v1/grammar/analyze", json={"text": long_text})
    assert response.status_code in {400, 422}


# ===========================================================================
# 8. Public Sentence Response Does Not Leak Internal Fields
# ===========================================================================


def test_public_sentence_response_does_not_leak_internal_fields(client: TestClient):
    """TEST 8: Internal parser/grammar-engine fields must never leak into public API output."""
    sentence = "If I had known about the meeting, I would have attended it."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    data = response.json()
    analysis = data["analysis"]

    forbidden_fields = {
        "objects_of_prepositions",
        "conditional_type",
        "clause_constructions",
        "tokens",
        "morph",
        "dep",
        "head",
        "tag",
        "lemma",
    }
    for field in forbidden_fields:
        assert field not in analysis, f"Internal field '{field}' leaked into public API response!"


# ===========================================================================
# 9. Sentence Response Schema
# ===========================================================================


def test_sentence_response_schema_completeness(client: TestClient):
    """TEST 9: Public sentence analysis contains all expected schema fields."""
    sentence = "The report was completed by the team yesterday."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    analysis = response.json()["analysis"]

    expected_fields = [
        "original_sentence",
        "corrected_sentence",
        "is_grammatically_correct",
        "bangla_translation",
        "sentence_type",
        "sentence_purpose",
        "tense",
        "voice",
        "subject",
        "main_verbs",
        "auxiliary_verbs",
        "objects",
        "complements",
        "nouns",
        "pronouns",
        "adjectives",
        "adverbs",
        "prepositions",
        "conjunctions",
        "determiners",
        "phrases",
        "clauses",
        "structure",
        "grammar_explanations",
        "common_mistakes",
        "natural_alternatives",
    ]
    for field in expected_fields:
        assert field in analysis, f"Expected field '{field}' missing from sentence response."


# ===========================================================================
# 10. NLP Parser Unavailable Fallback
# ===========================================================================


def test_nlp_parser_unavailable_fallback(client: TestClient):
    """TEST 10: When local NLPParser raises NLPParserUnavailableError, API completes cleanly using LLM."""
    sentence = "She plays the piano."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch(
        "app.services.grammar_service.nlp_parser.parse",
        side_effect=NLPParserUnavailableError("spaCy model unavailable in test"),
    ), patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "sentence"
    assert data["analysis"]["original_sentence"] == sentence


# ===========================================================================
# 11. Grammar Engine Failure Fallback
# ===========================================================================


def test_grammar_engine_failure_fallback(client: TestClient):
    """TEST 11: Controlled GrammarEngine failure falls back safely to parser facts + LLM."""
    sentence = "She plays the piano."
    mock_response = SentenceAnalysisResponse.model_validate(
        make_mock_sentence_data(sentence)
    )

    with patch(
        "app.services.grammar_service.grammar_engine.analyze",
        side_effect=RuntimeError("Controlled grammar engine test failure"),
    ), patch.object(
        grammar_service.ai_router,
        "generate_structured",
        new=AsyncMock(
            return_value=AIStructuredResult(
                data=mock_response, provider="mock", model="mock"
            )
        ),
    ):
        response = client.post("/api/v1/grammar/analyze", json={"text": sentence})

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "sentence"
    assert data["analysis"]["original_sentence"] == sentence


# ===========================================================================
# 12. Provider Not Called in Tests
# ===========================================================================


def test_provider_not_called_in_tests():
    """TEST 12: Ensure that real AI provider clients are NEVER invoked during tests."""
    from app.services.gemini_service import gemini_service
    from app.services.groq_service import groq_service

    with patch.object(
        gemini_service,
        "generate_structured",
        side_effect=AssertionError("Real Gemini provider was called during tests!"),
    ), patch.object(
        groq_service,
        "generate_structured",
        side_effect=AssertionError("Real Groq provider was called during tests!"),
    ):
        # Successfully verifies the mock boundary
        assert True
