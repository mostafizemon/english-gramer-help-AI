import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom NLP Parser Exceptions
# ---------------------------------------------------------------------------


class NLPParserError(Exception):
    """Base exception for all NLP parser errors."""


class NLPParserInputError(NLPParserError):
    """Raised when the input text is invalid (e.g. empty, non-string)."""


class NLPParserUnavailableError(NLPParserError):
    """Raised when spaCy or the required English model (en_core_web_sm) cannot be loaded."""


# ---------------------------------------------------------------------------
# Dependency & POS Constants
# ---------------------------------------------------------------------------

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
DIRECT_OBJECT_DEPS = {"dobj", "obj"}
INDIRECT_OBJECT_DEPS = {"iobj", "dative"}
PREPOSITION_OBJECT_DEPS = {"pobj"}
AUXILIARY_DEPS = {"aux", "auxpass"}

# Nominal dependent relations to form a cohesive noun phrase span
NOMINAL_CHILD_DEPS = {
    "det",
    "amod",
    "compound",
    "nummod",
    "quantmod",
    "poss",
    "case",
    "nmod",
    "predet",
}

# Subordinating / coordinating conjunction tags and dependency markers
CONJUNCTION_POS = {"CCONJ", "SCONJ"}
CONJUNCTION_DEPS = {"cc", "mark"}


# ---------------------------------------------------------------------------
# NLP Parser Service
# ---------------------------------------------------------------------------


class NLPParser:
    """Deterministic natural language processing parser for English sentences
    using spaCy dependency parsing and part-of-speech tagging.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Optional[Any] = None

    def _get_nlp(self) -> Any:
        """Lazily load the spaCy model on first use.

        The model is NOT loaded at module import time, ensuring that application
        startup remains decoupled from model loading until parse() is invoked.
        """
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise NLPParserUnavailableError(
                    "spaCy is not installed. Please install it using: pip install 'spacy>=3.8,<4'"
                ) from exc

            try:
                self._nlp = spacy.load(self.model_name)
            except Exception as exc:
                raise NLPParserUnavailableError(
                    f"Failed to load spaCy model '{self.model_name}'. "
                    f"Please download it using: python -m spacy download {self.model_name}"
                ) from exc

        return self._nlp

    @staticmethod
    def _validate_input(sentence: Any) -> str:
        """Validate and sanitize input sentence string."""
        if not isinstance(sentence, str):
            raise NLPParserInputError("Input sentence must be a string.")

        cleaned = sentence.strip()
        if not cleaned:
            raise NLPParserInputError("Input sentence cannot be empty or whitespace.")

        return cleaned

    @staticmethod
    def _deduplicate_preserve_order(items: list[str]) -> list[str]:
        """Deduplicate a list of strings while preserving first-occurrence order."""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @staticmethod
    def _format_token(token: Any) -> dict[str, Any]:
        """Convert a spaCy token into a clean, JSON-serializable dictionary."""
        morph_dict: dict[str, list[str]] = {}
        for key, value in token.morph.to_dict().items():
            if isinstance(value, str):
                morph_dict[key] = [v.strip() for v in value.split(",") if v.strip()]
            elif isinstance(value, (list, tuple)):
                morph_dict[key] = list(value)
            else:
                morph_dict[key] = [str(value)]

        return {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "index": token.i,
            "morph": morph_dict,
        }

    @staticmethod
    def _extract_nominal_span(token: Any, doc: Any) -> str:
        """Extract a clean, contiguous noun/pronoun phrase span for a given head token."""
        # 1. Prefer predefined spaCy noun chunks if token is the chunk root
        for chunk in doc.noun_chunks:
            if chunk.root == token:
                clean_chunk = chunk.text.strip(" ,.!?;:\"'()[]{}")
                if clean_chunk:
                    return clean_chunk

        # 2. Subtree expansion for nominal modifiers
        span_tokens = [
            t
            for t in token.subtree
            if t == token or (t.dep_ in NOMINAL_CHILD_DEPS and t.head == token)
        ]
        span_tokens.sort(key=lambda t: t.i)

        # Check if collected tokens form a contiguous span with the head token
        indices = [t.i for t in span_tokens]
        if indices and indices[-1] - indices[0] + 1 == len(indices):
            clean_span = doc[indices[0] : indices[-1] + 1].text.strip(
                " ,.!?;:\"'()[]{}"
            )
            if clean_span:
                return clean_span

        return token.text.strip(" ,.!?;:\"'()[]{}")

    def parse(self, sentence: str) -> dict[str, Any]:
        """Parse an English sentence and extract deterministic low-level grammatical facts.

        Args:
            sentence: The raw English input sentence string.

        Returns:
            A structured, JSON-serializable dictionary containing tokens, grammatical
            roles, parts of speech, and syntactic dependencies.

        Raises:
            NLPParserInputError: If the input is not a non-empty string.
            NLPParserUnavailableError: If spaCy or the language model is not available.
        """
        cleaned_text = self._validate_input(sentence)
        nlp = self._get_nlp()
        doc = nlp(cleaned_text)

        tokens: list[dict[str, Any]] = [self._format_token(t) for t in doc]

        subjects_raw: list[str] = []
        main_verbs_raw: list[str] = []
        auxiliary_verbs_raw: list[str] = []
        direct_objects_raw: list[str] = []
        indirect_objects_raw: list[str] = []
        objects_of_prepositions_raw: list[str] = []
        prepositions_raw: list[str] = []
        determiners_raw: list[str] = []
        nouns_raw: list[str] = []
        pronouns_raw: list[str] = []
        adjectives_raw: list[str] = []
        adverbs_raw: list[str] = []
        conjunctions_raw: list[str] = []

        root_token_text = ""

        for token in doc:
            dep = token.dep_
            pos = token.pos_
            tag = token.tag_

            # Root token
            if dep == "ROOT" and not root_token_text:
                root_token_text = token.text

            # 1. Subjects (nsubj, nsubjpass, csubj, csubjpass)
            if dep in SUBJECT_DEPS:
                subjects_raw.append(self._extract_nominal_span(token, doc))

            # 2. Auxiliary vs Main Verbs
            if dep in AUXILIARY_DEPS or tag == "MD":
                auxiliary_verbs_raw.append(token.text)
            elif pos == "VERB" or (pos == "AUX" and dep not in AUXILIARY_DEPS):
                main_verbs_raw.append(token.text)

            # 3. Direct Objects (dobj, obj)
            if dep in DIRECT_OBJECT_DEPS:
                direct_objects_raw.append(self._extract_nominal_span(token, doc))

            # 4. Indirect Objects (iobj, dative)
            if dep in INDIRECT_OBJECT_DEPS:
                indirect_objects_raw.append(self._extract_nominal_span(token, doc))

            # 5. Objects of Prepositions (pobj)
            if dep in PREPOSITION_OBJECT_DEPS:
                objects_of_prepositions_raw.append(
                    self._extract_nominal_span(token, doc)
                )

            # 6. Prepositions (ADP or prep dependency, excluding infinitival 'to')
            if pos == "ADP" or (dep == "prep" and pos != "PART" and tag != "TO"):
                prepositions_raw.append(token.text)

            # 7. Parts of Speech Classifications
            if pos == "DET" or tag in {"DT", "WDT", "PDT"}:
                determiners_raw.append(token.text)

            if pos in {"NOUN", "PROPN"}:
                nouns_raw.append(token.text)

            if pos == "PRON" or tag in {"PRP", "PRP$", "WP", "WP$"}:
                pronouns_raw.append(token.text)

            if pos == "ADJ" or tag in {"JJ", "JJR", "JJS"}:
                adjectives_raw.append(token.text)

            # Adverbs: lexical adverbs and interrogative/wh-adverbs functioning adverbially
            is_adverbial = (
                pos == "ADV"
                or tag in {"RB", "RBR", "RBS", "WRB"}
                or (dep == "advmod" and tag != "IN")
            )
            if is_adverbial:
                adverbs_raw.append(token.text)

            # Conjunctions: coordinating (CCONJ/cc) and genuine subordinating markers (mark/SCONJ)
            # Do NOT classify wh-adverbs (tag WRB / dep advmod) as conjunctions
            is_wh_adverb = tag == "WRB" or (dep == "advmod" and pos == "SCONJ")
            if not is_wh_adverb:
                if (
                    pos == "CCONJ"
                    or dep == "cc"
                    or tag == "CC"
                    or dep == "mark"
                    or (pos == "SCONJ" and dep not in {"advmod", "npadvmod"})
                ):
                    conjunctions_raw.append(token.text)

        sentences: list[str] = [s.text.strip() for s in doc.sents if s.text.strip()]

        return {
            "text": cleaned_text,
            "tokens": tokens,
            "subjects": self._deduplicate_preserve_order(subjects_raw),
            "main_verbs": self._deduplicate_preserve_order(main_verbs_raw),
            "auxiliary_verbs": self._deduplicate_preserve_order(auxiliary_verbs_raw),
            "direct_objects": self._deduplicate_preserve_order(direct_objects_raw),
            "indirect_objects": self._deduplicate_preserve_order(indirect_objects_raw),
            "objects_of_prepositions": self._deduplicate_preserve_order(
                objects_of_prepositions_raw
            ),
            "prepositions": self._deduplicate_preserve_order(prepositions_raw),
            "determiners": self._deduplicate_preserve_order(determiners_raw),
            "nouns": self._deduplicate_preserve_order(nouns_raw),
            "pronouns": self._deduplicate_preserve_order(pronouns_raw),
            "adjectives": self._deduplicate_preserve_order(adjectives_raw),
            "adverbs": self._deduplicate_preserve_order(adverbs_raw),
            "conjunctions": self._deduplicate_preserve_order(conjunctions_raw),
            "root": root_token_text,
            "sentences": sentences,
        }


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

nlp_parser = NLPParser()
