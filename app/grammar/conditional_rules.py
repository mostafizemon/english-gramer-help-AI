from typing import Any, Optional

CONDITIONAL_MARKERS = {"if", "unless", "provided", "providing"}


def find_conditional_marker(tokens: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Find the subordinating conditional marker (e.g., 'if', 'unless') in the token list."""
    for tok in tokens:
        lemma = tok.get("lemma", "").lower()
        text = tok.get("text", "").lower()
        pos = tok.get("pos", "")
        dep = tok.get("dep", "")

        if (lemma in CONDITIONAL_MARKERS or text in CONDITIONAL_MARKERS) and (
            pos in {"SCONJ", "ADP", "CCONJ"} or dep in {"mark", "advmod", "prep"}
        ):
            return tok

    return None


def detect_conditional_pattern(
    parsed_data: dict[str, Any],
    clause_constructions: list[dict[str, Any]],
) -> dict[str, Optional[str]]:
    """Conservatively detect conditional types (Zero, First, Second, Third, Mixed)
    based on the presence of conditional markers and compatible clause verb constructions.

    Returns:
        A dictionary with "conditional_type", "if_clause_verb", and "main_clause_verb".
    """
    tokens = parsed_data.get("tokens", [])
    root_text = parsed_data.get("root", "")

    marker_token = find_conditional_marker(tokens)
    if not marker_token or len(clause_constructions) < 2:
        return {
            "conditional_type": None,
            "if_clause_verb": None,
            "main_clause_verb": None,
        }

    # Find the verb associated with the conditional marker (marker's head)
    marker_head_text = marker_token.get("head", "")

    # Locate the if-clause construction and main-clause construction
    if_clause: Optional[dict[str, Any]] = None
    main_clause: Optional[dict[str, Any]] = None

    for clause in clause_constructions:
        verb_name = clause.get("verb", "")
        if verb_name == marker_head_text:
            if_clause = clause
        elif verb_name == root_text or not main_clause:
            main_clause = clause

    # Fallback if marker_head didn't directly match: first clause is if-clause, other is main-clause
    if not if_clause and len(clause_constructions) >= 2:
        if_clause = clause_constructions[0]
        main_clause = clause_constructions[1]

    if not if_clause or not main_clause:
        return {
            "conditional_type": None,
            "if_clause_verb": None,
            "main_clause_verb": None,
        }

    if_tense = if_clause.get("tense")
    if_construction = if_clause.get("construction")
    main_tense = main_clause.get("tense")
    main_construction = main_clause.get("construction")

    # 1. Third Conditional: if + Past Perfect, would/could/might + have + VBN (Modal Perfect)
    if if_tense == "Past Perfect" and main_construction == "Modal Perfect":
        return {
            "conditional_type": "Third Conditional",
            "if_clause_verb": if_clause.get("verb"),
            "main_clause_verb": main_clause.get("verb"),
        }

    # 2. First Conditional: if + Present Simple, will + base (Future Simple) or modal
    if if_tense == "Present Simple" and (
        main_tense == "Future Simple"
        or main_construction in {"Future Simple", "Modal + Base Verb"}
    ):
        return {
            "conditional_type": "First Conditional",
            "if_clause_verb": if_clause.get("verb"),
            "main_clause_verb": main_clause.get("verb"),
        }

    # 3. Second Conditional: if + Past Simple, would/could/might + base (Modal + Base Verb)
    if if_tense == "Past Simple" and main_construction == "Modal + Base Verb":
        return {
            "conditional_type": "Second Conditional",
            "if_clause_verb": if_clause.get("verb"),
            "main_clause_verb": main_clause.get("verb"),
        }

    # 4. Zero Conditional: if + Present Simple, Present Simple
    if if_tense == "Present Simple" and main_tense == "Present Simple":
        return {
            "conditional_type": "Zero Conditional",
            "if_clause_verb": if_clause.get("verb"),
            "main_clause_verb": main_clause.get("verb"),
        }

    # 5. Mixed Conditional: if + Past Perfect, would + base (Modal + Base Verb)
    if if_tense == "Past Perfect" and main_construction == "Modal + Base Verb":
        return {
            "conditional_type": "Mixed Conditional",
            "if_clause_verb": if_clause.get("verb"),
            "main_clause_verb": main_clause.get("verb"),
        }

    return {
        "conditional_type": None,
        "if_clause_verb": if_clause.get("verb"),
        "main_clause_verb": main_clause.get("verb"),
    }
