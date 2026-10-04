"""Regression tests for deterministic NLP Parser (app/services/nlp_parser.py).

Zero network calls, zero AI/LLM calls.
Uses the local spaCy model to test token extraction, POS mapping, and dependency normalization.
"""

import pytest
from app.services.nlp_parser import nlp_parser


def test_third_conditional_parsing():
    sentence = "If I had known about the meeting, I would have attended it."
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["I"]
    assert parsed["main_verbs"] == ["known", "attended"]
    assert parsed["auxiliary_verbs"] == ["had", "would", "have"]
    assert parsed["direct_objects"] == ["it"]
    assert parsed["indirect_objects"] == []
    assert parsed["objects_of_prepositions"] == ["the meeting"]
    assert parsed["prepositions"] == ["about"]
    assert parsed["determiners"] == ["the"]
    assert parsed["nouns"] == ["meeting"]
    assert parsed["pronouns"] == ["I", "it"]
    assert parsed["conjunctions"] == ["If"]
    assert parsed["root"] == "attended"


def test_passive_sentence_parsing():
    sentence = "The report was completed by the team yesterday."
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["The report"]
    assert parsed["main_verbs"] == ["completed"]
    assert parsed["auxiliary_verbs"] == ["was"]
    assert parsed["direct_objects"] == []
    assert parsed["prepositions"] == ["by"]
    assert parsed["objects_of_prepositions"] == ["the team"]
    assert parsed["determiners"] == ["The", "the"]


def test_wh_question_parsing():
    sentence = "Why did he leave the office early?"
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["he"]
    assert parsed["main_verbs"] == ["leave"]
    assert parsed["auxiliary_verbs"] == ["did"]
    assert parsed["direct_objects"] == ["the office"]
    assert parsed["determiners"] == ["the"]
    # Regression check: "Why" (tag WRB / dep advmod) must be an adverb and NOT a conjunction
    assert parsed["adverbs"] == ["Why", "early"]
    assert parsed["conjunctions"] == []
    assert parsed["root"] == "leave"


def test_incorrect_subject_verb_agreement_parsing():
    sentence = "He go to school every day."
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["He"]
    # Parser must preserve original verbatim token "go", not normalize to "goes"
    assert parsed["main_verbs"] == ["go"]
    assert parsed["auxiliary_verbs"] == []
    assert parsed["prepositions"] == ["to"]
    assert parsed["determiners"] == ["every"]
    assert "school" in parsed["nouns"]
    assert "day" in parsed["nouns"]


def test_present_continuous_parsing():
    sentence = "I am running in the park."
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["I"]
    assert parsed["main_verbs"] == ["running"]
    assert parsed["auxiliary_verbs"] == ["am"]
    assert parsed["prepositions"] == ["in"]
    assert parsed["objects_of_prepositions"] == ["the park"]


def test_present_perfect_continuous_parsing():
    sentence = "She has been working here for five years."
    parsed = nlp_parser.parse(sentence)

    assert parsed["subjects"] == ["She"]
    assert parsed["main_verbs"] == ["working"]
    assert parsed["auxiliary_verbs"] == ["has", "been"]
    assert "for" in parsed["prepositions"]
    assert "five years" in parsed["objects_of_prepositions"]


def test_coordinating_conjunction_parsing():
    sentence = "She sings and dances."
    parsed = nlp_parser.parse(sentence)

    assert "and" in parsed["conjunctions"]
    assert "and" not in parsed["adverbs"]


def test_subordinating_conjunction_parsing():
    sentence = "Because I was tired, I went home."
    parsed = nlp_parser.parse(sentence)

    assert "Because" in parsed["conjunctions"]
