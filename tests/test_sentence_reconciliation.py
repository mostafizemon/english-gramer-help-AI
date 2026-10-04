"""Regression tests for sentence-analysis hybrid reconciliation layer (app/services/grammar_service.py).

Zero network calls, zero AI/LLM calls.
Proves that deterministic NLPParser and GrammarEngine authoritative facts override
misleading or hallucinated LLM values while strictly preserving pedagogical teaching fields.
"""

from typing import Any
import pytest

from app.schemas.grammar import (
    SentenceAnalysisResponse,
    SentenceClause,
    SentenceGrammarExplanation,
    SentenceMistake,
    SentencePhrase,
)
from app.services.grammar_service import grammar_service


def make_mock_sentence_response(
    original_sentence: str = "Test sentence.",
    **overrides: Any,
) -> SentenceAnalysisResponse:
    """Helper to construct a valid base SentenceAnalysisResponse for mock LLM outputs."""
    default_data: dict[str, Any] = {
        "original_sentence": original_sentence,
        "corrected_sentence": None,
        "is_grammatically_correct": True,
        "bangla_translation": "এটি একটি পরীক্ষামূলক বাক্য।",
        "sentence_type": "Simple",
        "sentence_purpose": "Declarative",
        "tense": "Present Simple",
        "voice": "Active",
        "subject": ["Default Subject"],
        "main_verbs": ["test"],
        "auxiliary_verbs": [],
        "objects": [],
        "complements": [],
        "nouns": ["sentence"],
        "pronouns": [],
        "adjectives": [],
        "adverbs": [],
        "prepositions": [],
        "conjunctions": [],
        "determiners": [],
        "phrases": [],
        "clauses": [],
        "structure": "Subject + Verb",
        "grammar_explanations": [],
        "common_mistakes": [],
        "natural_alternatives": [],
    }
    default_data.update(overrides)
    return SentenceAnalysisResponse.model_validate(default_data)


def test_third_conditional_override():
    """Test 1: Third Conditional deterministic override of incorrect LLM grammar facts."""
    sentence = "If I had known about the meeting, I would have attended it."

    # Intentionally wrong/hallucinated LLM response
    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        subject=["Someone"],
        main_verbs=["wrong"],
        auxiliary_verbs=[],
        objects=["the meeting"],  # Prepositional object mistakenly in objects
        determiners=["the", "the"],
        prepositions=[],
        nouns=["wrong"],
        pronouns=[],
        tense="Present Simple",
        voice="Passive",
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    assert reconciled.subject == ["I"]
    assert reconciled.main_verbs == ["known", "attended"]
    assert reconciled.auxiliary_verbs == ["had", "would", "have"]
    assert reconciled.objects == ["it"]
    assert "the meeting" not in reconciled.objects
    assert reconciled.prepositions == ["about"]
    assert reconciled.determiners == ["the"]
    assert reconciled.nouns == ["meeting"]
    assert reconciled.pronouns == ["I", "it"]
    assert (
        reconciled.tense
        == "Past Perfect in the if-clause + Modal Perfect in the main clause"
    )
    assert reconciled.voice == "Active"


def test_passive_override():
    """Test 2: Passive voice deterministic override."""
    sentence = "The report was completed by the team yesterday."

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        subject=["The team"],
        main_verbs=["complete"],
        auxiliary_verbs=[],
        objects=["The report", "the team"],
        prepositions=[],
        determiners=[],
        tense="Future Simple",
        voice="Active",
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    assert reconciled.subject == ["The report"]
    assert reconciled.main_verbs == ["completed"]
    assert reconciled.auxiliary_verbs == ["was"]
    assert reconciled.objects == []
    assert "The team" not in reconciled.objects
    assert "the team" not in reconciled.objects
    assert reconciled.prepositions == ["by"]
    assert reconciled.determiners == ["The", "the"]
    assert reconciled.tense == "Past Simple"
    assert reconciled.voice == "Passive"


def test_wh_question_regression():
    """Test 3: WH Question regression ensuring 'Why' is an adverb and not a conjunction."""
    sentence = "Why did he leave the office early?"

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        subject=["wrong"],
        main_verbs=["wrong"],
        auxiliary_verbs=[],
        objects=[],
        adverbs=[],
        conjunctions=["Why"],  # Hallucinated conjunction
        determiners=[],
        tense="Present Simple",
        voice="Passive",
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    assert reconciled.subject == ["he"]
    assert reconciled.main_verbs == ["leave"]
    assert reconciled.auxiliary_verbs == ["did"]
    assert reconciled.objects == ["the office"]
    assert reconciled.adverbs == ["Why", "early"]
    assert reconciled.conjunctions == []
    assert reconciled.determiners == ["the"]
    assert reconciled.tense == "Past Simple"
    assert reconciled.voice == "Active"


def test_incorrect_sentence_preserves_original_facts():
    """Test 4: Deterministic layer analyzes original input while LLM correction is preserved."""
    sentence = "He go to school every day."

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        is_grammatically_correct=False,
        corrected_sentence="He goes to school every day.",
        main_verbs=["goes"],  # LLM erroneously puts corrected verb in main_verbs
        subject=["He"],
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    # Deterministic layer must preserve original token "go"
    assert reconciled.main_verbs == ["go"]

    # Pedagogical correction fields from LLM must remain intact
    assert reconciled.is_grammatically_correct is False
    assert reconciled.corrected_sentence == "He goes to school every day."


def test_non_authoritative_llm_fields_remain():
    """Test 5: Reconciliation does not destroy valid LLM-generated teaching/pedagogical fields."""
    sentence = "The book is on the table."

    phrases = [
        SentencePhrase(
            text="on the table",
            type="Prepositional Phrase",
            function="Adverbial of Place",
            bangla_explanation="স্থানের বিশদ ব্যাখ্যা।",
        )
    ]
    clauses = [
        SentenceClause(
            text="The book is on the table",
            type="Independent Clause",
            function="Main Clause",
            bangla_explanation="প্রধান খণ্ডবাক্য।",
        )
    ]
    grammar_explanations = [
        SentenceGrammarExplanation(
            topic="Preposition of Place",
            english_explanation="Explains spatial relation.",
            bangla_explanation="অবস্থান সম্পর্কিত ব্যাকরণ নিয়ম।",
        )
    ]
    common_mistakes = [
        SentenceMistake(
            issue="Preposition error",
            original="in the table",
            correction="on the table",
            english_explanation="Use 'on' for surfaces.",
            bangla_explanation="পৃষ্ঠতলের জন্য on ব্যবহৃত হয়।",
        )
    ]
    natural_alternatives = ["The book lies on the table."]

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        bangla_translation="বইটি টেবিলের উপর রয়েছে।",
        sentence_type="Simple",
        sentence_purpose="Declarative",
        structure="Subject + Linking Verb + Prepositional Phrase",
        phrases=phrases,
        clauses=clauses,
        grammar_explanations=grammar_explanations,
        common_mistakes=common_mistakes,
        natural_alternatives=natural_alternatives,
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    # Teaching/pedagogical fields must remain untouched
    assert reconciled.bangla_translation == "বইটি টেবিলের উপর রয়েছে।"
    assert reconciled.sentence_type == "Simple"
    assert reconciled.sentence_purpose == "Declarative"
    assert reconciled.structure == "Subject + Linking Verb + Prepositional Phrase"
    assert reconciled.phrases == phrases
    assert reconciled.clauses == clauses
    assert reconciled.grammar_explanations == grammar_explanations
    assert reconciled.common_mistakes == common_mistakes
    assert reconciled.natural_alternatives == natural_alternatives


def test_null_grammar_fact_does_not_destroy_llm_value():
    """Test 6: Null GrammarEngine facts do not blindly overwrite valid LLM tense/voice."""
    sentence = "Whosoever shall seek to save his life shall lose it."

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        tense="Archaic Future Construction",
        voice="Active",
    )

    # Simulated unsupported/null grammar engine facts
    synthetic_grammar_facts = {
        "tense": None,
        "voice": None,
        "conditional_type": None,
        "clause_constructions": [],
    }

    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts=None, grammar_facts=synthetic_grammar_facts
    )

    # Must retain LLM tense and voice instead of setting to None
    assert reconciled.tense == "Archaic Future Construction"
    assert reconciled.voice == "Active"


def test_deduplication():
    """Test 7: Summary fields preserve unique normalized items without duplicates."""
    sentence = "If I had known about the meeting, I would have attended it."

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        determiners=["the", "the", "the"],
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    assert reconciled.determiners == ["the"]


def test_pydantic_validation():
    """Test 8: Reconciled output is a fully validated SentenceAnalysisResponse instance."""
    sentence = "She is reading a novel."

    fake_llm_response = make_mock_sentence_response(
        original_sentence=sentence,
        subject=["She"],
        main_verbs=["reading"],
        auxiliary_verbs=["is"],
        objects=["a novel"],
    )

    parser_facts, grammar_facts = grammar_service._extract_deterministic_facts(sentence)
    reconciled = grammar_service._reconcile_sentence_analysis(
        fake_llm_response, parser_facts, grammar_facts
    )

    assert isinstance(reconciled, SentenceAnalysisResponse)
    assert reconciled.original_sentence == sentence
    assert reconciled.subject == ["She"]
    assert reconciled.main_verbs == ["reading"]
    assert reconciled.auxiliary_verbs == ["is"]
    assert reconciled.objects == ["a novel"]
