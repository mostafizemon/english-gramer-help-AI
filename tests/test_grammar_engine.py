"""Regression tests for deterministic Grammar Engine (app/services/grammar_engine.py).

Zero network calls, zero AI/LLM calls.
Flow tested:
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)
"""

import pytest
from app.services.nlp_parser import nlp_parser
from app.services.grammar_engine import grammar_engine


def test_present_simple():
    sentence = "I work every day."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Simple"
    assert result["voice"] == "Active"


def test_past_simple():
    sentence = "He worked yesterday."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Past Simple"
    assert result["voice"] == "Active"


def test_present_continuous():
    sentence = "I am running in the park."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Continuous"
    assert result["voice"] == "Active"


def test_present_perfect_continuous():
    sentence = "She has been working here for five years."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Perfect Continuous"
    assert result["voice"] == "Active"


def test_third_conditional():
    sentence = "If I had known about the meeting, I would have attended it."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["conditional_type"] == "Third Conditional"
    assert (
        result["tense"]
        == "Past Perfect in the if-clause + Modal Perfect in the main clause"
    )
    assert result["voice"] == "Active"

    # Find the clause construction for verb 'known'
    known_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "known"), None
    )
    assert known_clause is not None
    assert known_clause["auxiliaries"] == ["had"]
    assert known_clause["tense"] == "Past Perfect"

    # Find the clause construction for verb 'attended'
    attended_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "attended"), None
    )
    assert attended_clause is not None
    assert attended_clause["auxiliaries"] == ["would", "have"]
    assert attended_clause["construction"] == "Modal Perfect"


def test_past_simple_passive():
    sentence = "The report was completed by the team yesterday."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Past Simple"
    assert result["voice"] == "Passive"
    assert result["conditional_type"] is None

    completed_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "completed"), None
    )
    assert completed_clause is not None
    assert completed_clause["auxiliaries"] == ["was"]
    assert completed_clause["construction"] == "Past Simple Passive"


def test_present_perfect_passive():
    sentence = "The report has been completed."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Perfect"
    assert result["voice"] == "Passive"


def test_past_simple_question_with_do_support():
    sentence = "Why did he leave the office early?"
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Past Simple"
    assert result["voice"] == "Active"


def test_present_simple_question_with_do_support():
    sentence = "Does he work here?"
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Simple"
    assert result["voice"] == "Active"


def test_first_conditional():
    sentence = "If it rains, we will stay home."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["conditional_type"] == "First Conditional"

    rains_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "rains"), None
    )
    assert rains_clause is not None
    assert rains_clause["tense"] == "Present Simple"

    stay_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "stay"), None
    )
    assert stay_clause is not None
    assert stay_clause["auxiliaries"] == ["will"]
    assert stay_clause["construction"] == "Future Simple"


def test_negated_present_perfect():
    sentence = "She has not finished the work."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["tense"] == "Present Perfect"
    assert result["voice"] == "Active"


def test_negated_third_conditional():
    sentence = "If I had not known, I would not have attended."
    parsed = nlp_parser.parse(sentence)
    result = grammar_engine.analyze(parsed)

    assert result["conditional_type"] == "Third Conditional"

    known_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "known"), None
    )
    assert known_clause is not None
    assert known_clause["auxiliaries"] == ["had"]
    assert known_clause["tense"] == "Past Perfect"

    attended_clause = next(
        (c for c in result["clause_constructions"] if c["verb"] == "attended"), None
    )
    assert attended_clause is not None
    assert attended_clause["auxiliaries"] == ["would", "have"]
    assert attended_clause["construction"] == "Modal Perfect"
