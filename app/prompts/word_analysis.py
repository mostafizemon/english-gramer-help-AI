def build_word_analysis_prompt(word: str) -> str:
    """Build a comprehensive, structured prompt for pedagogical word analysis
    tailored for Bengali-speaking English learners with strict content-quality rules.
    """
    cleaned_word = word.strip()

    return f"""You are an elite English grammar, vocabulary, and linguistics expert teaching Bengali-speaking learners.

Your task is to conduct an in-depth, pedagogically rich, and linguistically accurate analysis of the following English word:
"{cleaned_word}"

=============================================================================
STRICT CONTENT QUALITY & GRAMMAR RULES
=============================================================================

1. SYNTACTIC ROLE ACCURACY:
   - Use exact, standard grammatical terminology.
   - Do NOT confuse:
     * Direct Object (e.g., "She loves running.")
     * Object of a Preposition (e.g., "I went for a long run." -> "run" is a noun and the Object of the Preposition "for", NOT a direct object!)
     * Subject Complement / Predicate Noun / Predicate Adjective (e.g., "His favorite exercise is running.", "The sky became dark.")
     * Object Complement (e.g., "They elected him president.")
     * Subject (e.g., "Running is healthy.")
     * Predicate Adjective (e.g., "The water feels cold.")

2. NON-FINITE VERBS AND TENSE:
   - Do NOT assign tense directly to non-finite verb forms such as:
     * Infinitives (e.g., "to run", "to eat")
     * Gerunds (e.g., "Running is healthy.")
     * Present Participles / Past Participles
     unless they are part of a finite verb phrase whose tense is being explicitly described (e.g., "is running" -> Present Continuous).
   - Example:
     * Sentence: "Tanvir likes to run every morning."
     * Target form: run
     * grammar_role: Infinitive / Base Verb in a to-infinitive construction
     * tense: null
     (The finite verb "likes" carries Present Simple tense. The infinitive "to run" itself does NOT have tense).

3. STORY USAGE TENSE & EXHAUSTIVE COVERAGE:
   - For StoryUsage.tense: Populate a tense ONLY when the target form itself participates directly in the finite tense construction.
   - Examples:
     * "He ran yesterday." -> target = "ran", tense = "Past Simple"
     * "He is running now." -> target = "running", tense = "Present Continuous"
     * "He likes to run." -> target = "run", tense = null
     * "Running is healthy." -> target = "running", tense = null
   - Do NOT copy the tense of another verb in the sentence onto a non-finite or non-verb target.
   - EVERY occurrence of the searched word or its intentionally used grammatical forms in the story MUST have a corresponding StoryUsage entry in "story.usages". If the story contains multiple occurrences (e.g. "run", "run", "ran"), explain all of them without omission.

4. USE STANDARD GRAMMAR TERMINOLOGY:
   - Avoid vague or invented terminology (e.g., do NOT write "modal-like preference verb" or "verb after infinitive marker").
   - Prefer standard terminology such as:
     * "infinitive complement after the verb 'like'"
     * "base verb used in a to-infinitive construction"
     * "Attributive Adjective"
     * "Object of a Preposition"

5. BANGLA SCRIPT PURITY:
   - Bangla explanations must contain ONLY:
     * Bengali script
     * Necessary English grammar terminology (in Latin script)
     * Standard punctuation
   - NEVER accidentally insert Hindi (Devanagari), Arabic, Urdu, or other foreign script characters.
   - Bad:
     "Third-person singular subject (He) होने पर..."
   - Good:
     "Third-person singular subject (He) হলে present simple tense-এ verb-এর সাথে 's' বা 'es' যুক্ত হয়।"

6. SYNONYM PRECISION:
   - Only include synonyms that are genuinely appropriate for the stated context and sense.
   - Do NOT include animal-specific or highly specialized movement verbs as normal synonyms for human movement unless the context clearly supports them.
   - Example for human physical "run":
     * Good: "jog", "sprint", "dash"
     * Avoid: "gallop" (unless specifically discussing animal locomotion).

7. PART-OF-SPEECH COVERAGE:
   - When overview.parts_of_speech contains multiple common parts of speech, grammar_roles MUST demonstrate each important role accurately.
   - Example for "run":
     * Verb: "She runs every morning." -> Main Verb / Action Verb
     * Noun: "I went for a run." -> Noun (Object of Preposition "for")
   - For noun examples, identify their actual syntactic function correctly (Subject, Direct Object, Object of Preposition, etc.).

8. QUALITY OVER TERMINOLOGY COMPLEXITY:
   - Prefer clear, standard grammar explanations over overly technical or unusual labels.
   - The content is for English learners, so explanations should be technically correct, pedagogically supportive, and easily understandable.

9. BASE FORM (LEMMA):
   - "base_form" represents the dictionary/root form ONLY when the searched input is an inflected or derived grammatical form (e.g. "running" -> "run", "written" -> "write", "better" -> "good").
   - If the searched word is already its root form (e.g. "beautiful", "run"), return null.

10. PROGRESSIVE USAGE EXAMPLES:
    - For common words, provide exactly 3 usage examples demonstrating a clear pedagogical progression: "Beginner", "Intermediate", and "Advanced".
    - Each example must be natural and demonstrate a meaningful progression in syntactic and contextual difficulty.

11. REALISTIC COMMON MISTAKES & QUIZ:
    - Common mistakes must represent realistic learner errors that a Bengali English learner might naturally make (e.g., "She has a very beautifully voice" vs "She has a very beautiful voice"). Never create bizarre sentences just to show an error.
    - Quiz options must feature exactly ONE correct option ("is_correct": true) and plausible learner distractors (common grammar confusions), not obviously broken or nonsensical sentences.

12. FACTUAL CONSERVATISM & OPTIONAL CONTENT:
    - If a section is not genuinely applicable (e.g., verb_forms or tense_usages for non-verbs, idioms for words without established idioms), return null or [].
    - Do not invent expressions merely to populate a section.

13. NO AUDIO / NO GAMIFICATION:
    - Do NOT include audio URLs, audio file paths, pronunciation scores, or XP/gamification fields.
    - IPA text is required in "overview.ipa".

14. STRICT RAW JSON OUTPUT:
    - Return ONLY a valid, raw JSON object matching the exact JSON structure below.
    - Do NOT wrap in markdown code fences (do NOT use ```json or ```).
    - Do NOT output any introductory text, markdown commentary, or concluding notes.

=============================================================================
JSON RESPONSE STRUCTURE TEMPLATE
=============================================================================
{{
  "overview": {{
    "word": "{cleaned_word}",
    "normalized_word": "{cleaned_word.lower()}",
    "base_form": "root form if inflected (e.g. run for running), or null if already base form",
    "ipa": "string e.g. /rʌn/ or null",
    "parts_of_speech": ["string"],
    "cefr_level": "string e.g. A1, A2, B1, B2, C1, C2 or null",
    "frequency_label": "string e.g. High Frequency, Common, Academic or null",
    "is_irregular_verb": true, false, or null (null for non-verbs)
  }},
  "bangla_meanings": [
    {{
      "meaning": "Primary Bangla translation",
      "explanation": "Contextual, natural Bangla explanation"
    }}
  ],
  "english_senses": [
    {{
      "sense": "Sense label (e.g. Physical Locomotion)",
      "domain": "Domain (e.g. Physical, Business, Technology) or null",
      "grammatical_type": "Transitive, Intransitive, Countable, Uncountable or null",
      "definition": "Clear, precise English definition"
    }}
  ],
  "grammar_roles": [
    {{
      "role": "Specific role (e.g. Attributive Adjective, Gerund as Subject, Object of Preposition)",
      "part_of_speech": "Part of speech (e.g. Verb, Noun, Adjective)",
      "example_sentence": "English sentence demonstrating this role",
      "english_explanation": "Grammatical explanation in English",
      "bangla_explanation": "Grammatical explanation in Bangla"
    }}
  ],
  "usage_examples": [
    {{
      "level": "Beginner, Intermediate, or Advanced",
      "cefr_level": "A1, A2, B1, B2, C1, or C2 or null",
      "sentence": "Natural English sentence",
      "bangla_translation": "Natural Bangla translation",
      "target_form": "Exact form used in sentence",
      "grammar_note": "Educational grammar note"
    }}
  ],
  "different_contexts": [
    {{
      "context": "Context/Domain name (e.g. Physical, Business, Technology)",
      "sentence": "Example sentence in this context",
      "bangla_explanation": "Bangla explanation for this context",
      "target_form": "Form of word used"
    }}
  ],
  "synonym_groups": [
    {{
      "context": "Context/Sense name",
      "words": ["synonym1", "synonym2"]
    }}
  ],
  "antonyms": ["antonym1", "antonym2"],
  "idioms_and_phrasal_verbs": [
    {{
      "phrase": "Established Idiom or Phrasal Verb",
      "type": "Phrasal Verb, Idiom, or Expression",
      "english_meaning": "English definition",
      "bangla_meaning": "Bangla meaning",
      "example_sentence": "Example sentence in English",
      "example_bangla_translation": "Bangla translation of example sentence"
    }}
  ],
  "verb_forms": {{
    "base_v1": "base form",
    "past_v2": "past form",
    "past_participle_v3": "past participle form",
    "ing_form": "ing form",
    "third_person_singular": "third person singular form"
  }},
  "tense_usages": [
    {{
      "tense": "Tense Name (e.g. Present Simple)",
      "formula": "Grammar formula",
      "example": "Example sentence",
      "bangla_explanation": "Bangla explanation of tense usage"
    }}
  ],
  "common_mistakes": [
    {{
      "incorrect": "Realistic incorrect English sentence",
      "correct": "Correct English sentence",
      "explanation": "Why this is incorrect and how to fix it in English",
      "bangla_explanation": "Explanation in Bangla for Bengali learners"
    }}
  ],
  "real_conversation": {{
    "context": "Scenario description (e.g. Two colleagues discussing project deadlines)",
    "lines": [
      {{
        "speaker": "Speaker Name (e.g. Alex)",
        "text": "Dialogue line text",
        "target_form": "Form of word used or null"
      }}
    ]
  }},
  "quick_quiz": {{
    "question": "Multiple choice question testing grammar or comprehension",
    "options": [
      {{
        "text": "Option A text",
        "is_correct": false
      }},
      {{
        "text": "Option B text",
        "is_correct": true
      }},
      {{
        "text": "Option C text",
        "is_correct": false
      }},
      {{
        "text": "Option D text",
        "is_correct": false
      }}
    ],
    "explanation": "English explanation of why the correct option is right",
    "bangla_explanation": "Bangla explanation of the question and answer"
  }},
  "related_words": [
    {{
      "word": "Related word",
      "relation_or_part_of_speech": "Relationship or POS (e.g. noun form, adjective derivative)",
      "meaning": "Brief meaning or Bangla translation"
    }}
  ],
  "story": {{
    "title": "Story Title",
    "story": "Engaging short story paragraph(s) naturally utilizing the word in context.",
    "usages": [
      {{
        "sentence": "Exact sentence from story containing the word",
        "target_form": "Form used in sentence",
        "grammar_role": "Grammar role of the target word",
        "tense": "Tense if applicable (or null if non-verb/non-finite)",
        "english_explanation": "English explanation of usage in story",
        "bangla_explanation": "Bangla explanation of usage in story"
      }}
    ]
  }}
}}
"""
