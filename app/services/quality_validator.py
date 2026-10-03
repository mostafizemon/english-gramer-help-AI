import re
from typing import Optional

from app.schemas.grammar import (
    SentenceAnalysisResponse,
    SentenceClause,
    SentencePhrase,
    WordAnalysisResponse,
)

# Regex pattern matching forbidden foreign scripts that should never appear in Bangla fields:
# - Devanagari (letters, signs, digits; excluding \u0964 DANDA and \u0965 DOUBLE DANDA used in Bengali)
# - Devanagari Extended
# - Arabic & Arabic-derived (Urdu, Persian, Pashto, etc.)
# - Arabic Supplement, Extended-A, Extended-B, Presentation Forms
# - Hebrew
# - Other Indic scripts: Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala
# - Southeast Asian scripts: Thai, Lao, Tibetan, Myanmar
# - Greek & Coptic
# - Cyrillic & Cyrillic Supplement
# - CJK Ideographs & Japanese Kana
# - Korean Hangul
FORBIDDEN_SCRIPTS_PATTERN = re.compile(
    r"["
    r"\u0900-\u0963\u0966-\u097F"      # Devanagari (excluding \u0964, \u0965)
    r"\uA8E0-\uA8FF"                  # Devanagari Extended
    r"\u0600-\u06FF"                  # Arabic
    r"\u0750-\u077F"                  # Arabic Supplement
    r"\u0870-\u089F\u08A0-\u08FF"      # Arabic Extended-B & Extended-A
    r"\uFB50-\uFDFF\uFE70-\uFEFF"      # Arabic Presentation Forms
    r"\u0590-\u05FF"                  # Hebrew
    r"\u0A00-\u0A7F"                  # Gurmukhi
    r"\u0A80-\u0AFF"                  # Gujarati
    r"\u0B00-\u0B7F"                  # Oriya
    r"\u0B80-\u0BFF"                  # Tamil
    r"\u0C00-\u0C7F"                  # Telugu
    r"\u0C80-\u0CFF"                  # Kannada
    r"\u0D00-\u0D7F"                  # Malayalam
    r"\u0D80-\u0DFF"                  # Sinhala
    r"\u0E00-\u0E7F"                  # Thai
    r"\u0E80-\u0EFF"                  # Lao
    r"\u0F00-\u0FFF"                  # Tibetan
    r"\u1000-\u109F"                  # Myanmar
    r"\u0370-\u03FF"                  # Greek
    r"\u0400-\u052F"                  # Cyrillic
    r"\u3040-\u30FF"                  # Japanese Hiragana / Katakana
    r"\u3400-\u4DBF\u4E00-\u9FFF"      # CJK Unified Ideographs
    r"\uAC00-\uD7AF"                  # Korean Hangul
    r"]"
)

# ---------------------------------------------------------------------------
# Sentence Structure Patterns & Constants (Rules 1-5)
# ---------------------------------------------------------------------------

CLAUSE_TERMS_PATTERN = re.compile(
    r"\b(?:independent|dependent|conditional|relative|adverbial|noun|complement|subordinate|coordinate|result|main)?\s*clause\b",
    re.IGNORECASE,
)

CLAUSE_FUNCTION_PATTERN = re.compile(
    r"\b(?:independent|dependent|conditional|result|relative|main|adverbial|noun|complement|subordinate|coordinate)?\s*clause\b",
    re.IGNORECASE,
)

SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
OBJECT_PRONOUNS = {"it", "them", "him", "her", "us", "me", "this", "that", "these", "those"}
ADVERBIAL_ENDINGS = {"here", "there", "yesterday", "today", "tomorrow", "now", "then", "everywhere", "anywhere", "somewhere"}

TRAILING_PREP_PHRASE_PATTERN = re.compile(
    r"\b(?:about|in|on|at|for|by|with|from|into|through|during|under|over|between|against|without|toward|towards)\s+(?:the|a|an|my|your|his|her|our|their|this|that|these|those|\d+|[a-zA-Z]+)\b",
    re.IGNORECASE,
)


def validate_bangla_text_quality(text: Optional[str]) -> bool:
    """Check if a Bangla text string is free from forbidden foreign-script contamination.

    Allowed:
    - Bengali Unicode characters (U+0980 - U+09FF)
    - Normal English / Latin letters (a-z, A-Z, accents)
    - Bengali danda (U+0964) and double danda (U+0965)
    - Standard numbers (0-9, ০-৯)
    - Standard punctuation, quotes, brackets, whitespace, symbols

    Forbidden:
    - Characters from Devanagari, Arabic, Urdu, Cyrillic, CJK, etc.
    """
    if not text:
        return True
    return FORBIDDEN_SCRIPTS_PATTERN.search(text) is None


def find_forbidden_characters(text: Optional[str]) -> list[str]:
    """Return a unique, sorted list of forbidden foreign-script characters found in text."""
    if not text:
        return []
    return sorted(list(set(FORBIDDEN_SCRIPTS_PATTERN.findall(text))))


def validate_word_analysis_quality(response: WordAnalysisResponse) -> list[str]:
    """Validate that all Bangla-intended fields in WordAnalysisResponse
    are strictly free from forbidden foreign-script contamination.

    Returns:
        A list of error descriptions for any violated fields, empty if all valid.
    """
    violations: list[str] = []

    def check_field(field_path: str, value: Optional[str]) -> None:
        if value and not validate_bangla_text_quality(value):
            forbidden = find_forbidden_characters(value)
            violations.append(
                f"Field '{field_path}' contains forbidden foreign-script characters {forbidden}: {value[:60]!r}"
            )

    # 1. bangla_meanings
    for idx, item in enumerate(response.bangla_meanings or []):
        check_field(f"bangla_meanings[{idx}].meaning", item.meaning)
        check_field(f"bangla_meanings[{idx}].explanation", item.explanation)

    # 2. grammar_roles
    for idx, item in enumerate(response.grammar_roles or []):
        check_field(f"grammar_roles[{idx}].bangla_explanation", item.bangla_explanation)

    # 3. usage_examples
    for idx, item in enumerate(response.usage_examples or []):
        check_field(f"usage_examples[{idx}].bangla_translation", item.bangla_translation)

    # 4. different_contexts
    for idx, item in enumerate(response.different_contexts or []):
        check_field(f"different_contexts[{idx}].bangla_explanation", item.bangla_explanation)

    # 5. idioms_and_phrasal_verbs
    for idx, item in enumerate(response.idioms_and_phrasal_verbs or []):
        check_field(f"idioms_and_phrasal_verbs[{idx}].bangla_meaning", item.bangla_meaning)
        check_field(
            f"idioms_and_phrasal_verbs[{idx}].example_bangla_translation",
            item.example_bangla_translation,
        )

    # 6. tense_usages
    for idx, item in enumerate(response.tense_usages or []):
        check_field(f"tense_usages[{idx}].bangla_explanation", item.bangla_explanation)

    # 7. common_mistakes
    for idx, item in enumerate(response.common_mistakes or []):
        check_field(f"common_mistakes[{idx}].bangla_explanation", item.bangla_explanation)

    # 8. quick_quiz
    if response.quick_quiz and response.quick_quiz.bangla_explanation:
        check_field("quick_quiz.bangla_explanation", response.quick_quiz.bangla_explanation)

    # 9. related_words
    for idx, item in enumerate(response.related_words or []):
        if item.meaning:
            check_field(f"related_words[{idx}].meaning", item.meaning)

    # 10. story.usages
    if response.story and response.story.usages:
        for idx, item in enumerate(response.story.usages or []):
            check_field(f"story.usages[{idx}].bangla_explanation", item.bangla_explanation)

    return violations


# ---------------------------------------------------------------------------
# Sentence Structure Validation Helper Functions (Rules 1-5)
# ---------------------------------------------------------------------------


def is_clause_type_name(type_name: Optional[str]) -> bool:
    """Check if a type string contains clause terminology."""
    if not type_name:
        return False
    return bool(CLAUSE_TERMS_PATTERN.search(type_name))


def is_clause_function_name(func_name: Optional[str]) -> bool:
    """Check if a function string identifies the item as a clause."""
    if not func_name:
        return False
    return bool(CLAUSE_FUNCTION_PATTERN.search(func_name))


def is_verb_phrase_type(type_name: Optional[str]) -> bool:
    """Check if a type string indicates a Verb Phrase."""
    if not type_name:
        return False
    lower = type_name.strip().lower()
    return "verb phrase" in lower or lower in {"verb", "main verb", "predicate verb", "predicate verb phrase"}


def validate_phrase_type_not_clause(phrase_idx: int, phrase_type: Optional[str]) -> Optional[str]:
    """RULE 1: Phrases must not have clause types."""
    if is_clause_type_name(phrase_type):
        return (
            f"phrases[{phrase_idx}].type '{phrase_type}' contains clause terminology. "
            f"phrases must contain only actual phrase categories (e.g. Noun Phrase, Verb Phrase, Prepositional Phrase, Adjective Phrase, Adverb Phrase)."
        )
    return None


def validate_phrase_function_not_clause(phrase_idx: int, phrase_function: Optional[str]) -> Optional[str]:
    """RULE 2: Phrase functions must not claim the item is a clause."""
    if is_clause_function_name(phrase_function):
        return (
            f"phrases[{phrase_idx}].function '{phrase_function}' identifies the phrase as a clause. "
            f"Phrase functions must not claim the item is an Independent Clause, Dependent Clause, Conditional Clause, Result Clause, Relative Clause, or Main Clause."
        )
    return None


def validate_verb_phrase_no_subject(
    phrase_idx: int, phrase_text: str, response: SentenceAnalysisResponse
) -> Optional[str]:
    """RULE 3: A Verb Phrase must not include an explicit subject."""
    cleaned_text = phrase_text.strip()
    if not cleaned_text:
        return None

    lower_text = cleaned_text.lower()
    words = cleaned_text.split()

    # 1. Check against response.subject
    for subj in response.subject or []:
        s_clean = subj.strip()
        if s_clean:
            s_lower = s_clean.lower()
            if lower_text == s_lower or lower_text.startswith(s_lower + " "):
                return (
                    f"phrases[{phrase_idx}] ('{cleaned_text}') is labeled as a Verb Phrase but includes explicit subject '{s_clean}'. "
                    f"Verb Phrases must only contain the verbal group (e.g. auxiliary + main verbs)."
                )

    # 2. Check against subject personal pronouns
    if len(words) > 1 and words[0].lower() in SUBJECT_PRONOUNS:
        return (
            f"phrases[{phrase_idx}] ('{cleaned_text}') is labeled as a Verb Phrase but starts with subject pronoun '{words[0]}'. "
            f"Verb Phrases must not include the subject."
        )

    # 3. Check if identical to any clause text
    for c in response.clauses or []:
        if c.text and lower_text == c.text.strip().lower():
            return (
                f"phrases[{phrase_idx}] ('{cleaned_text}') is labeled as a Verb Phrase but contains full clause text '{c.text}'."
            )

    return None


def validate_verb_phrase_no_objects_or_adjuncts(
    phrase_idx: int, phrase_text: str, response: SentenceAnalysisResponse
) -> Optional[str]:
    """RULE 4: A Verb Phrase should not include clear direct objects, complements, or prepositional/adverbial adjuncts."""
    cleaned_text = phrase_text.strip()
    if not cleaned_text:
        return None

    lower_text = cleaned_text.lower()
    words = cleaned_text.split()

    # 1. Check against identified response.objects
    for obj in response.objects or []:
        o_clean = obj.strip()
        if o_clean:
            o_lower = o_clean.lower()
            if lower_text == o_lower or lower_text.endswith(" " + o_lower):
                return (
                    f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but includes identified object '{o_clean}'. "
                    f"Objects must be separated from the Verb Phrase."
                )

    # 2. Check if ends with an object pronoun
    if len(words) > 1 and words[-1].lower() in OBJECT_PRONOUNS:
        return (
            f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but ends with object pronoun '{words[-1]}'. "
            f"Objects must be separated from the Verb Phrase."
        )

    # 3. Check against other phrases (e.g. Prepositional Phrase, Adverb Phrase, Noun Phrase)
    for other_idx, other in enumerate(response.phrases or []):
        if other_idx != phrase_idx and other.text and other.type:
            other_text = other.text.strip()
            if other_text:
                other_lower = other_text.lower()
                if is_verb_phrase_type(other.type):
                    continue
                if lower_text.endswith(" " + other_lower) and lower_text != other_lower:
                    return (
                        f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but includes trailing {other.type} '{other_text}'. "
                        f"Objects, prepositional phrases, and adverbials must be separated from the Verb Phrase."
                    )

    # 4. Check for trailing prepositional phrase patterns
    if TRAILING_PREP_PHRASE_PATTERN.search(cleaned_text):
        return (
            f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but includes a trailing prepositional phrase or complement. "
            f"Verb Phrases must contain only the verbal group."
        )

    # 5. Check against identified response.complements
    for comp in response.complements or []:
        c_clean = comp.strip()
        if c_clean:
            c_lower = c_clean.lower()
            if lower_text.endswith(" " + c_lower) and lower_text != c_lower:
                return (
                    f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but includes complement '{c_clean}'. "
                    f"Complements must be separated from the Verb Phrase."
                )

    # 6. Check if ends with standalone adverbial endings
    if len(words) > 1 and words[-1].lower() in ADVERBIAL_ENDINGS:
        return (
            f"phrases[{phrase_idx}] ('{cleaned_text}') is a Verb Phrase but ends with adverbial '{words[-1]}'. "
            f"Adverbials must be separated from the Verb Phrase."
        )

    return None


def validate_sentence_structure_consistency(response: SentenceAnalysisResponse) -> list[str]:
    """Validate structural consistency of phrases and clauses in SentenceAnalysisResponse.
    Enforces Rules 1 to 5:
    - Rule 1: Clause types must not appear in phrases
    - Rule 2: Phrase functions must not claim the item is a clause
    - Rule 3: Verb Phrases must not include an explicit subject
    - Rule 4: Verb Phrases should not include clear objects or adjuncts
    - Rule 5: Clause items belong in clauses (valid clause items are permitted)
    """
    violations: list[str] = []

    for idx, phrase in enumerate(response.phrases or []):
        # Rule 1: Type check
        err = validate_phrase_type_not_clause(idx, phrase.type)
        if err:
            violations.append(err)

        # Rule 2: Function check
        err = validate_phrase_function_not_clause(idx, phrase.function)
        if err:
            violations.append(err)

        # Rules 3 & 4: Verb Phrase structural checks
        if phrase.type and is_verb_phrase_type(phrase.type):
            err = validate_verb_phrase_no_subject(idx, phrase.text, response)
            if err:
                violations.append(err)

            err = validate_verb_phrase_no_objects_or_adjuncts(idx, phrase.text, response)
            if err:
                violations.append(err)

    return violations


def validate_sentence_analysis_quality(response: SentenceAnalysisResponse) -> list[str]:
    """Validate that all Bangla-intended fields in SentenceAnalysisResponse
    are strictly free from forbidden foreign-script contamination, and that phrases
    and clauses satisfy structural consistency rules.

    Returns:
        A list of error descriptions for any violated fields or rules, empty if all valid.
    """
    violations: list[str] = []

    def check_field(field_path: str, value: Optional[str]) -> None:
        if value and not validate_bangla_text_quality(value):
            forbidden = find_forbidden_characters(value)
            violations.append(
                f"Field '{field_path}' contains forbidden foreign-script characters {forbidden}: {value[:60]!r}"
            )

    check_field("bangla_translation", response.bangla_translation)

    for idx, item in enumerate(response.phrases or []):
        check_field(f"phrases[{idx}].bangla_explanation", item.bangla_explanation)

    for idx, item in enumerate(response.clauses or []):
        check_field(f"clauses[{idx}].bangla_explanation", item.bangla_explanation)

    for idx, item in enumerate(response.grammar_explanations or []):
        check_field(
            f"grammar_explanations[{idx}].bangla_explanation", item.bangla_explanation
        )

    for idx, item in enumerate(response.common_mistakes or []):
        check_field(
            f"common_mistakes[{idx}].bangla_explanation", item.bangla_explanation
        )

    # Structural consistency validation (Rules 1 - 5)
    structure_violations = validate_sentence_structure_consistency(response)
    violations.extend(structure_violations)

    return violations

