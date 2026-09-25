"""Deterministic, minimal prompt rendering for question-generation-prompt-v1."""

import json
from dataclasses import dataclass

from question_generation.config import PROMPT_VERSION
from question_generation.schemas import QuestionSpec


@dataclass(frozen=True)
class PromptPayload:
    system_instruction: str
    contents: str
    prompt_version: str = PROMPT_VERSION

    def render(self) -> str:
        return (
            f"prompt_version: {self.prompt_version}\n\n"
            f"[system]\n{self.system_instruction}\n\n[user]\n{self.contents}\n"
        )


SYSTEM_INSTRUCTION = """You author one multiple-choice diagnostic question from an already-selected
QuestionSpec. The specification is authoritative: do not select another concept, question type,
operation, or difficulty. Write in clear English. Produce exactly four plausible, mutually distinct
choices with exactly one correct answer. Do not use answer-length, grammar, absolutes, or
meta-language as clues. Do not use TODOs or placeholders. Do not claim that a named book, page,
chapter, or source says anything. Evidence metadata is audit context only, not citable source text.
Do not invent quotations, page references, book-specific facts, or unseen prose. For
vocabulary/recognize Level 1, test concise definition or identification of the primary concept. For
background_knowledge/recall Level 1, test prerequisite recall or basic understanding of the primary
concept. The explanation must identify why the correct choice is correct without referring to a
nonexistent source."""


def build_prompt(spec: QuestionSpec) -> PromptPayload:
    """Render only the target and compact audit metadata; never send whole topic/book content."""

    audit_context = {
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
    }
    contents = (
        "Generate exactly one question for the following authoritative target. "
        "Echo all target identity fields exactly.\n\n"
        + json.dumps(audit_context, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return PromptPayload(system_instruction=SYSTEM_INSTRUCTION, contents=contents)
