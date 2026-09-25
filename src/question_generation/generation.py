"""Provider-independent generation orchestration and deterministic finalization."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from question_generation.config import (
    GENERATED_QUESTION_VERSION,
    GENERATION_CONFIG_VERSION,
    PROMPT_VERSION,
)
from question_generation.prompt import PromptPayload, build_prompt
from question_generation.schemas import (
    GeneratedQuestion,
    ProviderQuestion,
    QuestionSpec,
    TokenUsage,
)
from question_generation.validation import validate_provider_question, validate_supported_spec


@dataclass(frozen=True)
class ProviderGeneration:
    output: ProviderQuestion
    model: str
    usage: TokenUsage


class QuestionGenerator(Protocol):
    def generate(self, prompt: PromptPayload) -> ProviderGeneration:
        """Return one already structured provider response."""


class FakeQuestionGenerator:
    """Network-free deterministic provider used by tests and local contract checks."""

    def __init__(
        self,
        output: ProviderQuestion,
        *,
        model: str = "fake-question-generator-v1",
        usage: TokenUsage | None = None,
    ) -> None:
        self.output = output
        self.model = model
        self.usage = usage or TokenUsage()
        self.prompts: list[PromptPayload] = []

    def generate(self, prompt: PromptPayload) -> ProviderGeneration:
        self.prompts.append(prompt)
        return ProviderGeneration(output=self.output, model=self.model, usage=self.usage)


def _generated_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "gq_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def generate_question(
    spec: QuestionSpec,
    *,
    artifact_hash: str,
    generator: QuestionGenerator,
) -> GeneratedQuestion:
    validate_supported_spec(spec)
    prompt = build_prompt(spec)
    provider = generator.generate(prompt)
    validate_provider_question(provider.output, spec)

    evidence_ids = list(dict.fromkeys(item.evidence_id for item in spec.supporting_evidence))
    identity_and_output: dict[str, object] = {
        "generated_question_version": GENERATED_QUESTION_VERSION,
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
        "stem": provider.output.stem,
        "choices": provider.output.choices,
        "correct_choice_index": provider.output.correct_choice_index,
        "explanation": provider.output.explanation,
        "generation_model": provider.model,
        "prompt_version": PROMPT_VERSION,
        "generation_config_version": GENERATION_CONFIG_VERSION,
        "question_spec_version": spec.question_spec_version,
        "question_spec_config_version": spec.config_version,
        "question_spec_config_hash": spec.config_hash,
        "supporting_book_ids": spec.supporting_book_ids,
        "supporting_evidence_ids": evidence_ids,
        "source_document_ids": spec.source_document_ids,
        "input_artifact_hash": artifact_hash,
    }
    return GeneratedQuestion(
        generated_question_id=_generated_id(identity_and_output),
        **identity_and_output,
        usage=provider.usage,
    )
