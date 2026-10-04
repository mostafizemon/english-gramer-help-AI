import logging
from typing import Any, Optional

from app.grammar.conditional_rules import detect_conditional_pattern
from app.grammar.tense_rules import (
    detect_verb_construction,
    get_auxiliaries_for_verb,
)
from app.grammar.voice_rules import detect_sentence_voice, detect_verb_voice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grammar Engine Service (Phase 2)
# ---------------------------------------------------------------------------


class GrammarEngine:
    """Deterministic, local English grammar engine that infers grammatical facts
    (tense/aspect, voice, conditional patterns, and clause-level verb constructions)
    directly from low-level NLP parser output without relying on external LLM calls.
    """

    def analyze(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Analyze a parsed sentence dictionary (produced by NLPParser.parse)
        and extract high-level deterministic grammatical facts.

        Args:
            parsed: The low-level parsed sentence dictionary.

        Returns:
            A structured, JSON-serializable dictionary containing tense, voice,
            conditional_type, and clause_constructions.
        """
        if not isinstance(parsed, dict):
            raise ValueError("Input to GrammarEngine.analyze must be a dictionary.")

        tokens = parsed.get("tokens", [])
        root_text = parsed.get("root", "")

        # 1. Identify all main / lexical verb tokens (including linking verbs)
        verb_tokens = [
            t
            for t in tokens
            if (t.get("pos") == "VERB" and t.get("dep") not in {"aux", "auxpass"})
            or (t.get("pos") == "AUX" and t.get("dep") not in {"aux", "auxpass"})
        ]

        # 2. Analyze verb groups for each main verb
        clause_constructions: list[dict[str, Any]] = []

        for verb in verb_tokens:
            aux_tokens = get_auxiliaries_for_verb(verb, tokens)
            aux_texts = [a.get("text", "") for a in aux_tokens]

            tense_info = detect_verb_construction(verb, aux_tokens, tokens)
            voice_info = detect_verb_voice(
                verb, aux_tokens, tokens, construction_info=tense_info
            )

            clause_constructions.append(
                {
                    "verb": verb.get("text", ""),
                    "auxiliaries": aux_texts,
                    "tense": tense_info.get("tense"),
                    "construction": tense_info.get("construction"),
                    "voice": voice_info,
                }
            )

        # 3. Detect conditional patterns
        conditional_res = detect_conditional_pattern(parsed, clause_constructions)
        conditional_type = conditional_res.get("conditional_type")

        # 4. Determine overall top-level tense
        top_level_tense: Optional[str] = None

        if len(clause_constructions) == 1:
            top_level_tense = clause_constructions[0].get("tense")
        elif conditional_type:
            if conditional_type == "Third Conditional":
                top_level_tense = "Past Perfect in the if-clause + Modal Perfect in the main clause"
            elif conditional_type == "First Conditional":
                top_level_tense = "Present Simple in the if-clause + Future Simple in the main clause"
            elif conditional_type == "Second Conditional":
                top_level_tense = "Past Simple in the if-clause + Modal construction in the main clause"
            elif conditional_type == "Zero Conditional":
                top_level_tense = "Present Simple"
            else:
                top_level_tense = None
        else:
            # Multi-clause non-conditional: check root clause tense
            root_clause = next(
                (c for c in clause_constructions if c.get("verb") == root_text),
                None,
            )
            if root_clause:
                top_level_tense = root_clause.get("tense")
            elif clause_constructions:
                top_level_tense = clause_constructions[0].get("tense")

        # 5. Determine overall top-level voice
        top_level_voice = detect_sentence_voice(clause_constructions, parsed)

        # Clean output clause constructions according to required structure
        formatted_clause_constructions = [
            {
                "verb": c["verb"],
                "auxiliaries": c["auxiliaries"],
                "tense": c["tense"],
                "construction": c["construction"],
            }
            for c in clause_constructions
        ]

        return {
            "tense": top_level_tense,
            "voice": top_level_voice,
            "conditional_type": conditional_type,
            "clause_constructions": formatted_clause_constructions,
        }


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

grammar_engine = GrammarEngine()
