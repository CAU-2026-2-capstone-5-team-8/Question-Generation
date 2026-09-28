"""Deterministic, minimal prompt rendering for the versioned generation prompt."""

import json
from dataclasses import dataclass

from question_generation.config import (
    DEFAULT_OUTPUT_LANGUAGE,
    GROUNDED_PROMPT_VERSION,
    GROUNDED_REVISION_PROMPT_VERSION,
    PROMPT_VERSION,
    REVISION_PROMPT_VERSION,
)
from question_generation.schemas import GeneratedQuestion, GenerationGrounding, QuestionSpec


@dataclass(frozen=True)
class PromptPayload:
    system_instruction: str
    contents: str
    output_language: str = DEFAULT_OUTPUT_LANGUAGE
    prompt_version: str = PROMPT_VERSION

    def render(self) -> str:
        return (
            f"prompt_version: {self.prompt_version}\n\n"
            f"[system]\n{self.system_instruction}\n\n[user]\n{self.contents}\n"
        )


SYSTEM_INSTRUCTION = """You author one multiple-choice diagnostic question from an already-selected
QuestionSpec. The specification is authoritative: do not select another concept, question type,
operation, or difficulty. Produce exactly four plausible, mutually distinct choices with exactly
one correct answer. Do not use answer-length, grammar, absolutes, or
meta-language as clues. Do not use TODOs or placeholders. Do not claim that a named book, page,
chapter, or source says anything. Evidence metadata is audit context only, not citable source text.
Do not invent quotations, page references, book-specific facts, or unseen prose. For
vocabulary/recognize Level 1, test concise definition or identification of the primary concept. For
background_knowledge/recall Level 1, test prerequisite recall or basic understanding of the primary
concept. The explanation must identify why the correct choice is correct without referring to a
nonexistent source."""

REVISION_INSTRUCTION = """You revise one previously generated diagnostic question in response to
reviewer feedback. The QuestionSpec remains authoritative. Return a complete replacement question,
not a patch or commentary. Address every applicable feedback point while preserving the target
concept, question type, cognitive operation, difficulty, and configured output-language policy. Do
not copy an identified factual, terminology, ambiguity, or wording error into the revision. The
previous question and feedback are revision context only and are not citable source evidence."""

GROUNDED_SYSTEM_INSTRUCTION = """You author one passage-grounded multiple-choice comprehension
question from an already-selected QuestionSpec. The QuestionSpec is authoritative: do not change
the topic, concept, question type, operation, or difficulty. Use only the provided passage as the
factual basis. A reader must be able to determine the correct answer without external knowledge,
but the question must require understanding and applying the passage rather than merely locating an
identical string. Do not invent book, page, chapter, quotation, or source claims. Return only the
question sentence in stem; do not copy or translate the passage into stem because the application
will display the exact source passage separately. Produce exactly four plausible, mutually distinct
choices with exactly one correct answer. Distractors must be plausible in the passage context. Do
not use answer-length, grammar, absolutes, meta-language, TODOs, or placeholders as clues. The
explanation must justify the answer from the supplied passage and must not rely on outside facts."""

GROUNDED_REVISION_INSTRUCTION = """You revise one previously generated passage-grounded question
in response to reviewer feedback. Return a complete replacement question, not a patch or
commentary. Preserve the authoritative QuestionSpec and use only the provided passage as the
factual basis. Do not modify, rewrite, translate, summarize, or repeat the passage in the returned
stem. Address every applicable feedback point. The revised question must require applying a rule
from the passage to a new situation; it must not be answerable by copying an example or matching an
identical string from the passage. Do not require outside knowledge. Preserve comprehension/apply
at Level 2. Produce exactly four plausible, mutually distinct choices with exactly one correct
answer, and explain the answer only from the passage. The previous question and reviewer feedback
are revision context, not additional factual evidence."""


def _language_policy(output_language: str) -> str:
    if not output_language or output_language != output_language.strip():
        raise ValueError("output_language must be nonblank and trimmed")
    if output_language == "ko-KR":
        return """The authoritative output language is ko-KR. Write the stem, every choice, and the
explanation in Korean. Keep canonical concept and topic identifiers exactly as provided; do not
translate or alter identity fields in the structured response. Use natural Korean translations for
technical terms. At first mention, or when a translation is ambiguous, you may write a term as
Korean (English term), for example 프로세스(process). Do not add English to every technical term
unnecessarily, and do not change the meaning of technical terminology."""
    return f"""The authoritative output language is {output_language}. Write the stem, every choice,
and the explanation in the language identified by that locale. Keep canonical concept and topic
identifiers exactly as provided; do not translate or alter identity fields in the structured
response. Use natural localized technical terminology without changing its meaning."""


def _audit_context(spec: QuestionSpec, output_language: str) -> dict[str, object]:
    return {
        "question_spec_id": spec.question_id,
        "topic_id": spec.topic_id,
        "question_type": spec.question_type,
        "cognitive_operation": spec.cognitive_operation,
        "primary_concept": spec.primary_concept,
        "related_concepts": spec.related_concepts,
        "prerequisite_concepts": spec.prerequisite_concepts,
        "target_difficulty": spec.target_difficulty,
        "difficulty_rationale": spec.difficulty_rationale,
        "evidence_summary": spec.evidence_summary,
        "supporting_book_ids": spec.supporting_book_ids,
        "supporting_evidence": [
            {
                "concept_id": item.concept_id,
                "book_id": item.book_id,
                "evidence_type": item.evidence_type,
                "evidence_id": item.evidence_id,
            }
            for item in spec.supporting_evidence
        ],
        "source_document_ids": spec.source_document_ids,
        "question_spec_version": spec.question_spec_version,
        "config_version": spec.config_version,
        "config_hash": spec.config_hash,
        "output_language": output_language,
    }


def build_prompt(
    spec: QuestionSpec,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> PromptPayload:
    """Render only the target and compact audit metadata; never send whole topic/book content."""

    audit_context = _audit_context(spec, output_language)
    contents = (
        "Generate exactly one question for the following authoritative target. "
        "Echo all target identity fields exactly.\n\n"
        + json.dumps(audit_context, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return PromptPayload(
        system_instruction=f"{SYSTEM_INSTRUCTION}\n\n{_language_policy(output_language)}",
        contents=contents,
        output_language=output_language,
    )


def build_grounded_prompt(
    spec: QuestionSpec,
    grounding: GenerationGrounding,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> PromptPayload:
    """Render one authoritative target with the exact bounded grounding passage."""

    context = {
        "authoritative_target": _audit_context(spec, output_language),
        "grounding": {
            "grounding_version": grounding.grounding_version,
            "source_document_id": grounding.source_document_id,
            "document_content_hash": grounding.document_content_hash,
            "passage_extraction_policy": grounding.passage_extraction_policy,
            "passage_hash": grounding.passage_hash,
            "passage_text": grounding.passage_text,
        },
    }
    contents = (
        "Generate exactly one question for the authoritative target. Echo all target identity "
        "fields exactly and ground the answer only in grounding.passage_text.\n\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return PromptPayload(
        system_instruction=(
            f"{GROUNDED_SYSTEM_INSTRUCTION}\n\n{_language_policy(output_language)}"
        ),
        contents=contents,
        output_language=output_language,
        prompt_version=GROUNDED_PROMPT_VERSION,
    )


def build_revision_prompt(
    spec: QuestionSpec,
    previous: GeneratedQuestion,
    feedback: str,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> PromptPayload:
    """Render a versioned replacement request from one prior question and reviewer feedback."""

    if not feedback or feedback != feedback.strip():
        raise ValueError("revision feedback must be nonblank and trimmed")
    if len(feedback) > 8000:
        raise ValueError("revision feedback must not exceed 8000 characters")
    revision_context = {
        "authoritative_target": _audit_context(spec, output_language),
        "previous_question": {
            "generated_question_id": previous.generated_question_id,
            "stem": previous.stem,
            "choices": previous.choices,
            "correct_choice_index": previous.correct_choice_index,
            "explanation": previous.explanation,
        },
        "reviewer_feedback": feedback,
    }
    contents = (
        "Generate exactly one revised question for the authoritative target. "
        "Echo all target identity fields exactly.\n\n"
        + json.dumps(revision_context, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return PromptPayload(
        system_instruction=(
            f"{SYSTEM_INSTRUCTION}\n\n{REVISION_INSTRUCTION}\n\n{_language_policy(output_language)}"
        ),
        contents=contents,
        output_language=output_language,
        prompt_version=REVISION_PROMPT_VERSION,
    )


def build_grounded_revision_prompt(
    spec: QuestionSpec,
    grounding: GenerationGrounding,
    previous: GeneratedQuestion,
    feedback: str,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> PromptPayload:
    """Render a grounded replacement request while preserving the exact source passage."""

    if not feedback or feedback != feedback.strip():
        raise ValueError("revision feedback must be nonblank and trimmed")
    if len(feedback) > 8000:
        raise ValueError("revision feedback must not exceed 8000 characters")
    labels = ("지문", "질문") if output_language == "ko-KR" else ("Passage", "Question")
    expected_prefix = f"{labels[0]}:\n{grounding.passage_text}\n\n{labels[1]}:\n"
    if not previous.stem.startswith(expected_prefix):
        raise ValueError("previous grounded stem does not preserve the exact passage")
    previous_question = previous.stem[len(expected_prefix) :]
    if not previous_question.strip():
        raise ValueError("previous grounded question must be nonblank")
    revision_context = {
        "authoritative_target": _audit_context(spec, output_language),
        "grounding": {
            "grounding_version": grounding.grounding_version,
            "source_document_id": grounding.source_document_id,
            "document_content_hash": grounding.document_content_hash,
            "passage_extraction_policy": grounding.passage_extraction_policy,
            "passage_hash": grounding.passage_hash,
            "passage_text": grounding.passage_text,
        },
        "previous_question": {
            "generated_question_id": previous.generated_question_id,
            "question": previous_question,
            "choices": previous.choices,
            "correct_choice_index": previous.correct_choice_index,
            "explanation": previous.explanation,
        },
        "reviewer_feedback": feedback,
    }
    contents = (
        "Generate exactly one revised passage-grounded question for the authoritative target. "
        "Echo all target identity fields exactly and ground the answer only in "
        "grounding.passage_text.\n\n"
        + json.dumps(revision_context, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return PromptPayload(
        system_instruction=(
            f"{GROUNDED_SYSTEM_INSTRUCTION}\n\n{GROUNDED_REVISION_INSTRUCTION}\n\n"
            f"{_language_policy(output_language)}"
        ),
        contents=contents,
        output_language=output_language,
        prompt_version=GROUNDED_REVISION_PROMPT_VERSION,
    )
