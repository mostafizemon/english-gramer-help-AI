from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class WordAnalysisRequest(BaseModel):
    word: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The English word to analyze",
    )


class GrammarAnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The English word or sentence to analyze",
    )


# ---------------------------------------------------------------------------
# Word Analysis Component Schemas
# ---------------------------------------------------------------------------


class WordOverview(BaseModel):
    word: str = Field(..., description="The queried word")
    normalized_word: str = Field(..., description="Lowercase normalized word form")
    base_form: Optional[str] = Field(None, description="Base/lemma form of the word")
    ipa: Optional[str] = Field(None, description="International Phonetic Alphabet pronunciation")
    parts_of_speech: list[str] = Field(default_factory=list, description="List of parts of speech (e.g. Verb, Noun)")
    cefr_level: Optional[str] = Field(None, description="CEFR difficulty level (e.g. A1, A2, B1, B2, C1, C2)")
    frequency_label: Optional[str] = Field(None, description="Frequency indicator (e.g. High Frequency, Common, Academic)")
    is_irregular_verb: Optional[bool] = Field(None, description="Whether the word is an irregular verb (null if not a verb)")


class BanglaMeaning(BaseModel):
    meaning: str = Field(..., description="Primary Bangla translation/meaning")
    explanation: str = Field(..., description="Detailed Bangla explanation of the meaning")


class EnglishSense(BaseModel):
    sense: str = Field(..., description="Sense title/label (e.g. Physical Locomotion)")
    domain: Optional[str] = Field(None, description="Domain/subject area (e.g. Physical, Business, Technology)")
    grammatical_type: Optional[str] = Field(None, description="Grammatical sub-classification (e.g. Transitive, Intransitive, Countable)")
    definition: str = Field(..., description="Clear English definition for this sense")


class UsageExample(BaseModel):
    level: str = Field(..., description="Difficulty level (e.g. Beginner, Intermediate, Advanced)")
    cefr_level: Optional[str] = Field(None, description="CEFR level for this example sentence")
    sentence: str = Field(..., description="English example sentence")
    bangla_translation: str = Field(..., description="Accurate natural Bangla translation")
    target_form: str = Field(..., description="The specific form of the word used in the sentence")
    grammar_note: str = Field(..., description="Educational note explaining the grammar/usage in this example")


class ContextUsage(BaseModel):
    context: str = Field(..., description="Context name (e.g. Physical, Business, Technology, Social)")
    sentence: str = Field(..., description="English sentence illustrating usage in this context")
    bangla_explanation: str = Field(..., description="Bangla explanation of how the word functions in this context")
    target_form: str = Field(..., description="The form of the word used in this context")


class SynonymGroup(BaseModel):
    context: str = Field(..., description="The specific sense/context for this group of synonyms")
    words: list[str] = Field(default_factory=list, description="List of contextual synonyms")


class IdiomOrPhrasalVerb(BaseModel):
    phrase: str = Field(..., description="The idiom, phrasal verb, or common expression")
    type: str = Field(..., description="Type (e.g. Phrasal Verb, Idiom, Expression)")
    english_meaning: str = Field(..., description="Clear English definition of the phrase")
    bangla_meaning: str = Field(..., description="Bangla translation/meaning of the phrase")
    example_sentence: str = Field(..., description="English example sentence using the phrase")
    example_bangla_translation: str = Field(..., description="Bangla translation of the example sentence")


class VerbForms(BaseModel):
    base_v1: Optional[str] = Field(None, description="Base form (V1)")
    past_v2: Optional[str] = Field(None, description="Simple Past form (V2)")
    past_participle_v3: Optional[str] = Field(None, description="Past Participle form (V3)")
    ing_form: Optional[str] = Field(None, description="Present Participle / Gerund form (-ing)")
    third_person_singular: Optional[str] = Field(None, description="Third person singular present form (-s/-es)")


class TenseUsage(BaseModel):
    tense: str = Field(..., description="Name of the tense (e.g. Present Simple, Past Continuous)")
    formula: str = Field(..., description="Grammar structure/formula (e.g. Subject + V1(s/es) + Object)")
    example: str = Field(..., description="Example sentence demonstrating this tense with the word")
    bangla_explanation: str = Field(..., description="Bangla explanation of the tense usage and meaning")


class GrammarRole(BaseModel):
    role: str = Field(..., description="Specific grammatical role (e.g. Gerund as Subject, Present Participle as Adjective)")
    part_of_speech: str = Field(..., description="Part of speech associated with this role")
    example_sentence: str = Field(..., description="English sentence demonstrating this grammatical role")
    english_explanation: str = Field(..., description="English explanation of the grammatical function")
    bangla_explanation: str = Field(..., description="Bangla explanation of the grammatical function")


class CommonMistake(BaseModel):
    incorrect: str = Field(..., description="Common incorrect usage sentence or phrase")
    correct: str = Field(..., description="Corrected sentence or phrase")
    explanation: str = Field(..., description="English explanation of why the mistake occurs and how to fix it")
    bangla_explanation: str = Field(..., description="Bangla explanation of the mistake for Bengali learners")


class DialogueLine(BaseModel):
    speaker: str = Field(..., description="Speaker identifier (e.g. Person A, Alex, Teacher)")
    text: str = Field(..., description="Dialogue line text")
    target_form: Optional[str] = Field(None, description="The form of the target word used in this line, if any")


class RealConversation(BaseModel):
    context: str = Field(..., description="Scenario/setting description for the conversation")
    lines: list[DialogueLine] = Field(default_factory=list, description="Turn-by-turn dialogue lines")


class QuizOption(BaseModel):
    text: str = Field(..., description="Option choice text")
    is_correct: bool = Field(..., description="Whether this option is the correct answer")


class QuickQuiz(BaseModel):
    question: str = Field(..., description="Multiple choice quiz question testing comprehension or grammar")
    options: list[QuizOption] = Field(..., description="List of quiz answer options (typically 4 options)")
    explanation: str = Field(..., description="English explanation of the correct answer")
    bangla_explanation: str = Field(..., description="Bangla explanation of the quiz question and answer")


class RelatedWord(BaseModel):
    word: str = Field(..., description="Related word from the word family or derivation")
    relation_or_part_of_speech: str = Field(..., description="Relationship or part of speech (e.g. noun form, adjective, opposite)")
    meaning: Optional[str] = Field(None, description="Brief meaning or Bangla translation")


class StoryUsage(BaseModel):
    sentence: str = Field(..., description="Sentence from the story containing the word")
    target_form: str = Field(..., description="The specific form of the word used in this sentence")
    grammar_role: str = Field(..., description="Grammatical role in this context (e.g. Main Verb, Gerund)")
    tense: Optional[str] = Field(None, description="Tense if applicable")
    english_explanation: str = Field(..., description="English explanation of how the word works in this sentence")
    bangla_explanation: str = Field(..., description="Bangla explanation of the usage in this story sentence")


class LearningStory(BaseModel):
    title: str = Field(..., description="Engaging title for the story")
    story: str = Field(..., description="Short story paragraph(s) naturally integrating the word in various forms")
    usages: list[StoryUsage] = Field(default_factory=list, description="Breakdown of target word usages within the story")


# ---------------------------------------------------------------------------
# Top-Level Word Analysis Response Model
# ---------------------------------------------------------------------------


class WordAnalysisResponse(BaseModel):
    overview: WordOverview
    bangla_meanings: list[BanglaMeaning] = Field(default_factory=list)
    english_senses: list[EnglishSense] = Field(default_factory=list)
    grammar_roles: list[GrammarRole] = Field(default_factory=list)
    usage_examples: list[UsageExample] = Field(default_factory=list)
    different_contexts: list[ContextUsage] = Field(default_factory=list)
    synonym_groups: list[SynonymGroup] = Field(default_factory=list)
    antonyms: list[str] = Field(default_factory=list)
    idioms_and_phrasal_verbs: list[IdiomOrPhrasalVerb] = Field(default_factory=list)
    verb_forms: Optional[VerbForms] = None
    tense_usages: list[TenseUsage] = Field(default_factory=list)
    common_mistakes: list[CommonMistake] = Field(default_factory=list)
    real_conversation: RealConversation
    quick_quiz: QuickQuiz
    related_words: list[RelatedWord] = Field(default_factory=list)
    story: LearningStory


# ---------------------------------------------------------------------------
# Sentence Analysis Component Schemas
# ---------------------------------------------------------------------------


class SentencePhrase(BaseModel):
    text: str = Field(..., description="The phrase text from the sentence")
    type: str = Field(..., description="Type of phrase (e.g. Prepositional Phrase, Noun Phrase, Verb Phrase)")
    function: str = Field(..., description="Grammatical function (e.g. Adverbial of Place, Subject Complement)")
    bangla_explanation: str = Field(..., description="Bangla explanation of the phrase role and function")


class SentenceClause(BaseModel):
    text: str = Field(..., description="The clause text")
    type: str = Field(..., description="Type of clause (e.g. Independent Clause, Dependent Adverbial Clause, Relative Clause)")
    function: str = Field(..., description="Grammatical function in the sentence")
    bangla_explanation: str = Field(..., description="Bangla explanation of the clause function")


class SentenceGrammarExplanation(BaseModel):
    topic: str = Field(..., description="Grammar topic or rule (e.g. Subject-Verb Agreement, Present Continuous Usage)")
    english_explanation: str = Field(..., description="English explanation of the grammar concept")
    bangla_explanation: str = Field(..., description="Bangla pedagogical explanation of the grammar concept")


class SentenceMistake(BaseModel):
    issue: str = Field(..., description="Grammar issue name (e.g. Tense Error, Preposition Error)")
    original: str = Field(..., description="The incorrect segment or original faulty phrasing")
    correction: str = Field(..., description="Corrected segment or phrasing")
    english_explanation: str = Field(..., description="English explanation of the error and correction")
    bangla_explanation: str = Field(..., description="Bangla explanation of the error for Bengali learners")


# ---------------------------------------------------------------------------
# Top-Level Sentence Analysis Response Model
# ---------------------------------------------------------------------------


class SentenceAnalysisResponse(BaseModel):
    original_sentence: str = Field(..., description="The submitted sentence")
    corrected_sentence: Optional[str] = Field(None, description="Corrected sentence if original contains grammatical errors, or null if correct")
    is_grammatically_correct: bool = Field(..., description="Whether the original sentence is grammatically correct")
    bangla_translation: str = Field(..., description="Accurate and natural Bangla translation of the sentence")
    sentence_type: str = Field(..., description="Structural sentence type (Simple, Compound, Complex, Compound-Complex)")
    sentence_purpose: Optional[str] = Field(None, description="Functional sentence purpose (Declarative, Interrogative, Imperative, Exclamatory)")
    tense: Optional[str] = Field(None, description="Primary tense of the sentence (e.g. Present Continuous, Past Simple, or 'Mixed: ...' for multi-clause)")
    voice: Optional[str] = Field(None, description="Voice of the sentence (Active, Passive, or null if inapplicable)")
    subject: list[str] = Field(default_factory=list, description="Subject word(s) or noun phrase(s)")
    main_verbs: list[str] = Field(default_factory=list, description="Main verb(s)")
    auxiliary_verbs: list[str] = Field(default_factory=list, description="Auxiliary/modal verbs (e.g. is, have, can, will)")
    objects: list[str] = Field(default_factory=list, description="Direct/indirect object(s)")
    complements: list[str] = Field(default_factory=list, description="Subject/object complement(s)")
    nouns: list[str] = Field(default_factory=list, description="Nouns in the sentence")
    pronouns: list[str] = Field(default_factory=list, description="Pronouns in the sentence")
    adjectives: list[str] = Field(default_factory=list, description="Adjectives in the sentence")
    adverbs: list[str] = Field(default_factory=list, description="Adverbs in the sentence")
    prepositions: list[str] = Field(default_factory=list, description="Prepositions in the sentence")
    conjunctions: list[str] = Field(default_factory=list, description="Conjunctions in the sentence")
    determiners: list[str] = Field(default_factory=list, description="Determiners/articles (e.g. the, a, this, their)")
    phrases: list[SentencePhrase] = Field(default_factory=list, description="Key phrases identified in the sentence")
    clauses: list[SentenceClause] = Field(default_factory=list, description="Clauses identified in the sentence")
    structure: str = Field(..., description="Syntactic structural formula (e.g. Subject + Auxiliary Verb + Main Verb (-ing) + Prepositional Phrase)")
    grammar_explanations: list[SentenceGrammarExplanation] = Field(default_factory=list, description="Detailed grammar rules and pedagogical explanations")
    common_mistakes: list[SentenceMistake] = Field(default_factory=list, description="Identified errors or common mistakes associated with this sentence pattern")
    natural_alternatives: list[str] = Field(default_factory=list, description="Natural alternative ways to express the same idea in English")


# ---------------------------------------------------------------------------
# Universal Analysis Response Model
# ---------------------------------------------------------------------------


class GrammarAnalyzeResponse(BaseModel):
    input_type: Literal["word", "sentence"] = Field(..., description="Detected input type ('word' or 'sentence')")
    analysis: Union[WordAnalysisResponse, SentenceAnalysisResponse] = Field(..., description="Detailed word or sentence grammar analysis")
