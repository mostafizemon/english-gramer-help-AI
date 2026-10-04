from typing import Any, Optional


def get_auxiliaries_for_verb(
    verb_token: dict[str, Any], tokens: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract all auxiliary tokens that syntactically modify the given verb token."""
    verb_text = verb_token.get("text", "")

    aux_tokens: list[dict[str, Any]] = []

    for tok in tokens:
        dep = tok.get("dep", "")
        tag = tok.get("tag", "")
        head_text = tok.get("head", "")

        # Token is an auxiliary (aux/auxpass) or modal (tag MD) governing this verb
        if dep in {"aux", "auxpass"} or (tag == "MD" and dep not in {"ROOT"}):
            if head_text == verb_text:
                aux_tokens.append(tok)

    # Sort auxiliaries in original linear sentence order
    aux_tokens.sort(key=lambda t: t.get("index", 0))
    return aux_tokens


def detect_verb_construction(
    verb_token: dict[str, Any],
    aux_tokens: list[dict[str, Any]],
    all_tokens: list[dict[str, Any]],
) -> dict[str, Optional[str]]:
    """Determine the grammatical tense and construction for a lexical/main verb
    based on its auxiliary chain, part-of-speech tags, and morphological features.

    Returns:
        A dictionary with "tense" and "construction" (or None if ambiguous).
    """
    verb_text = verb_token.get("text", "")
    verb_tag = verb_token.get("tag", "")
    verb_lemma = verb_token.get("lemma", "").lower()
    verb_dep = verb_token.get("dep", "")
    verb_morph = verb_token.get("morph", {})

    aux_texts_lower = [a.get("text", "").lower() for a in aux_tokens]
    aux_lemmas_lower = [a.get("lemma", "").lower() for a in aux_tokens]
    aux_deps = [a.get("dep", "") for a in aux_tokens]

    has_passive_aux = "auxpass" in aux_deps

    # -------------------------------------------------------------------------
    # 1. Modal Perfect (would/could/might/should/must + have + VBN)
    # -------------------------------------------------------------------------
    if len(aux_texts_lower) >= 2 and aux_texts_lower[-1] in {"have", "'ve"}:
        first_aux = aux_texts_lower[0]
        if first_aux in {
            "would",
            "could",
            "might",
            "should",
            "must",
            "'d",
        } and verb_tag in {"VBN", "VBD", "VB"}:
            return {
                "tense": None,
                "construction": "Modal Perfect",
            }

    # -------------------------------------------------------------------------
    # 2. Modal + Base Verb (would/could/might/should/can/may/must + base)
    # -------------------------------------------------------------------------
    if (
        len(aux_texts_lower) == 1
        and aux_texts_lower[0]
        in {"would", "could", "might", "should", "must", "can", "may", "'d"}
    ):
        return {
            "tense": None,
            "construction": "Modal + Base Verb",
        }

    # -------------------------------------------------------------------------
    # 3. Future Tenses (will / shall)
    # -------------------------------------------------------------------------
    if aux_texts_lower and aux_texts_lower[0] in {"will", "shall", "'ll", "wo"}:
        # Future Perfect Continuous (will + have + been + VBG)
        if len(aux_texts_lower) >= 3 and aux_texts_lower[1:] == [
            "have",
            "been",
        ] and verb_tag == "VBG":
            return {
                "tense": "Future Perfect Continuous",
                "construction": "Future Perfect Continuous",
            }
        # Future Perfect (will + have + VBN)
        if (
            len(aux_texts_lower) >= 2
            and aux_texts_lower[1] == "have"
            and verb_tag in {"VBN", "VBD"}
        ):
            return {
                "tense": "Future Perfect",
                "construction": "Future Perfect",
            }
        # Future Continuous (will + be + VBG)
        if (
            len(aux_texts_lower) >= 2
            and aux_texts_lower[1] == "be"
            and verb_tag == "VBG"
        ):
            return {
                "tense": "Future Continuous",
                "construction": "Future Continuous",
            }
        # Future Simple (will + base verb)
        if len(aux_texts_lower) == 1 and verb_tag in {"VB", "VBP"}:
            return {
                "tense": "Future Simple",
                "construction": "Future Simple",
            }

    # -------------------------------------------------------------------------
    # 4. Perfect Continuous Tenses (have/has/had + been + VBG)
    # -------------------------------------------------------------------------
    if len(aux_texts_lower) >= 2 and "been" in aux_texts_lower and verb_tag == "VBG":
        if any(h in aux_texts_lower for h in ["had", "'d"]):
            return {
                "tense": "Past Perfect Continuous",
                "construction": "Past Perfect Continuous",
            }
        if any(h in aux_texts_lower for h in ["have", "has", "'ve", "'s"]):
            return {
                "tense": "Present Perfect Continuous",
                "construction": "Present Perfect Continuous",
            }

    # -------------------------------------------------------------------------
    # 5. Perfect Tenses (have/has/had + VBN or have/has/had + been + VBN passive)
    # -------------------------------------------------------------------------
    if any(h in aux_texts_lower for h in ["have", "has", "had", "'ve", "'s", "'d"]):
        is_past_perf = any(h in aux_texts_lower for h in ["had", "'d"])
        is_pres_perf = any(h in aux_texts_lower for h in ["have", "has", "'ve", "'s"])

        # Passive Perfect (has/have/had + been + VBN)
        if "been" in aux_texts_lower and verb_tag in {"VBN", "VBD"}:
            if is_past_perf:
                return {
                    "tense": "Past Perfect",
                    "construction": "Past Perfect Passive",
                }
            if is_pres_perf:
                return {
                    "tense": "Present Perfect",
                    "construction": "Present Perfect Passive",
                }

        # Active Perfect (has/have/had + VBN)
        if verb_tag in {"VBN", "VBD"} or "Perf" in verb_morph.get("Aspect", []):
            if is_past_perf:
                return {
                    "tense": "Past Perfect",
                    "construction": "Past Perfect",
                }
            if is_pres_perf:
                return {
                    "tense": "Present Perfect",
                    "construction": "Present Perfect",
                }

    # -------------------------------------------------------------------------
    # 6. Continuous Tenses (be + VBG or be + being + VBN passive)
    # -------------------------------------------------------------------------
    if any(
        b in aux_texts_lower
        for b in ["am", "is", "are", "was", "were", "'m", "'s", "'re"]
    ):
        is_past_be = any(b in aux_texts_lower for b in ["was", "were"])
        is_pres_be = any(
            b in aux_texts_lower for b in ["am", "is", "are", "'m", "'s", "'re"]
        )

        # Active Continuous (be + VBG)
        if verb_tag == "VBG":
            if is_past_be:
                return {
                    "tense": "Past Continuous",
                    "construction": "Past Continuous",
                }
            if is_pres_be:
                return {
                    "tense": "Present Continuous",
                    "construction": "Present Continuous",
                }

        # Passive Simple / Continuous (be + VBN)
        if verb_tag in {"VBN", "VBD"} or has_passive_aux:
            if "being" in aux_texts_lower:
                if is_past_be:
                    return {
                        "tense": "Past Continuous",
                        "construction": "Past Continuous Passive",
                    }
                if is_pres_be:
                    return {
                        "tense": "Present Continuous",
                        "construction": "Present Continuous Passive",
                    }
            else:
                if is_past_be:
                    return {
                        "tense": "Past Simple",
                        "construction": "Past Simple Passive",
                    }
                if is_pres_be:
                    return {
                        "tense": "Present Simple",
                        "construction": "Present Simple Passive",
                    }

    # -------------------------------------------------------------------------
    # 7. Do-Support (did / do / does + base verb)
    # -------------------------------------------------------------------------
    if any(d in aux_texts_lower for d in ["did", "do", "does", "ca", "wo"]):
        if "did" in aux_texts_lower:
            return {
                "tense": "Past Simple",
                "construction": "Past Simple",
            }
        if any(d in aux_texts_lower for d in ["do", "does"]):
            return {
                "tense": "Present Simple",
                "construction": "Present Simple",
            }

    # -------------------------------------------------------------------------
    # 8. Simple Tenses (No Auxiliaries)
    # -------------------------------------------------------------------------
    if not aux_tokens:
        # Past Simple (e.g. "He worked yesterday.", "I completed the report.")
        if verb_tag == "VBD" or (
            verb_lemma == "be" and verb_text.lower() in {"was", "were"}
        ):
            return {
                "tense": "Past Simple",
                "construction": "Past Simple",
            }

        # Present Simple 3rd Person Singular (e.g. "He works.", "She runs.", "It is.")
        if verb_tag == "VBZ" or (
            verb_lemma == "be" and verb_text.lower() in {"is", "'s"}
        ):
            return {
                "tense": "Present Simple",
                "construction": "Present Simple",
            }

        # Present Simple Non-3rd / Base (e.g. "I work.", "They run.", "I am.", "You are.")
        if verb_tag in {"VBP", "VB"} or (
            verb_lemma == "be"
            and verb_text.lower() in {"am", "are", "'m", "'re", "be"}
        ):
            return {
                "tense": "Present Simple",
                "construction": "Present Simple",
            }

    # Default fallback when construction cannot be determined with certainty
    return {
        "tense": None,
        "construction": None,
    }


def detect_clause_constructions(
    parsed_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify and analyze the verb group around each lexical/main verb in the sentence."""
    tokens = parsed_data.get("tokens", [])
    clause_constructions: list[dict[str, Any]] = []

    # Find all verbs (pos == "VERB" or pos == "AUX" when acting as main verb)
    verb_tokens = [
        t
        for t in tokens
        if (t.get("pos") == "VERB" and t.get("dep") not in {"aux", "auxpass"})
        or (t.get("pos") == "AUX" and t.get("dep") not in {"aux", "auxpass"})
    ]

    for verb in verb_tokens:
        aux_tokens = get_auxiliaries_for_verb(verb, tokens)
        aux_texts = [a.get("text", "") for a in aux_tokens]

        res = detect_verb_construction(verb, aux_tokens, tokens)

        clause_constructions.append(
            {
                "verb": verb.get("text", ""),
                "auxiliaries": aux_texts,
                "tense": res.get("tense"),
                "construction": res.get("construction"),
            }
        )

    return clause_constructions
