from typing import Any, Optional


def detect_verb_voice(
    verb_token: dict[str, Any],
    aux_tokens: list[dict[str, Any]],
    all_tokens: list[dict[str, Any]],
    construction_info: Optional[dict[str, Optional[str]]] = None,
) -> Optional[str]:
    """Deterministically determine the grammatical voice (Active vs Passive)
    for an individual lexical/main verb.
    """
    verb_text = verb_token.get("text", "")
    verb_tag = verb_token.get("tag", "")

    aux_deps = [a.get("dep", "") for a in aux_tokens]
    aux_texts_lower = [a.get("text", "").lower() for a in aux_tokens]

    # Check for passive evidence in dependencies
    has_auxpass = "auxpass" in aux_deps

    # Check for passive nominal subject governing this verb
    has_nsubjpass = any(
        t.get("dep") == "nsubjpass" and t.get("head") == verb_text
        for t in all_tokens
    )

    # Check if construction already explicitly identified a passive pattern
    construction = (construction_info or {}).get("construction") or ""
    is_passive_construction = "Passive" in construction

    # 1. Clear Passive Voice
    if has_auxpass or has_nsubjpass or is_passive_construction:
        return "Passive"

    # 2. Clear Active Voice Patterns
    # Active continuous (be + VBG, been + VBG)
    if verb_tag == "VBG":
        return "Active"

    # Active perfect (have/has/had + VBN without 'been' / auxpass)
    if any(h in aux_texts_lower for h in ["have", "has", "had"]) and "been" not in aux_texts_lower:
        return "Active"

    # Modal perfect / modal active (would have attended, will stay, did leave)
    if any(m in aux_texts_lower for m in ["would", "could", "might", "should", "will", "did", "do", "does"]):
        return "Active"

    # Simple tenses with active subjects (nsubj)
    has_active_subject = any(
        t.get("dep") in {"nsubj", "csubj"} and t.get("head") == verb_text
        for t in all_tokens
    )
    if has_active_subject and not has_auxpass and not has_nsubjpass:
        return "Active"

    # Simple finite active verbs (VBD, VBZ, VBP, VB) with no auxiliaries
    if not aux_tokens and verb_tag in {"VBD", "VBZ", "VBP", "VB"}:
        return "Active"

    return None


def detect_sentence_voice(
    clause_constructions: list[dict[str, Any]],
    parsed_data: dict[str, Any],
) -> Optional[str]:
    """Determine top-level voice for the sentence.
    Prefers root clause voice or overall unanimous voice.
    """
    if not clause_constructions:
        return None

    root_token_text = parsed_data.get("root", "")

    # Check the root verb voice first
    for clause in clause_constructions:
        if clause.get("verb") == root_token_text:
            root_voice = clause.get("voice")
            if root_voice:
                return root_voice

    # Otherwise if any clause is passive, check if root is passive
    voices = [c.get("voice") for c in clause_constructions if c.get("voice")]
    if all(v == "Active" for v in voices):
        return "Active"
    if all(v == "Passive" for v in voices):
        return "Passive"

    return voices[0] if voices else None
