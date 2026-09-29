"""Provider-independent generation orchestration and deterministic finalization."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from question_generation.config import (
    DEFAULT_OUTPUT_LANGUAGE,
    DISPLAY_GROUNDED_GENERATED_QUESTION_VERSION,
    GENERATED_QUESTION_VERSION,
    GENERATION_CONFIG_VERSION,
    GROUNDED_GENERATED_QUESTION_VERSION,
)
from question_generation.errors import InputContractError
from question_generation.prompt import (
    PromptPayload,
    build_display_grounded_prompt,
    build_display_grounded_revision_prompt,
    build_grounded_prompt,
    build_grounded_revision_prompt,
    build_prompt,
    build_revision_prompt,
)
from question_generation.schemas import (
    GeneratedQuestion,
    GeneratedQuestionV4,
    GenerationGrounding,
    GenerationGroundingV2,
    ProviderQuestion,
    QuestionSpec,
    TokenUsage,
)
from question_generation.validation import (
    validate_grounding_for_spec,
    validate_provider_question,
    validate_supported_spec,
)


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
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestion:
    validate_supported_spec(spec)
    prompt = build_prompt(spec, output_language=output_language)
    return _generate_from_prompt(
        spec,
        artifact_hash=artifact_hash,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
    )


def generate_grounded_question(
    spec: QuestionSpec,
    *,
    blueprint_hash: str,
    canonical_file_hashes: dict[str, str],
    grounding: GenerationGrounding,
    grounding_artifact_hash: str,
    generator: QuestionGenerator,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestion:
    """Generate the narrow comprehension/apply slice from an exact grounded passage."""

    validate_grounding_for_spec(
        grounding,
        spec,
        blueprint_hash=blueprint_hash,
        canonical_file_hashes=canonical_file_hashes,
    )
    prompt = build_grounded_prompt(spec, grounding, output_language=output_language)
    return _generate_from_prompt(
        spec,
        artifact_hash=grounding_artifact_hash,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
        generated_question_version=GROUNDED_GENERATED_QUESTION_VERSION,
        grounding=grounding,
    )


def generate_display_grounded_question(
    spec: QuestionSpec,
    *,
    blueprint_hash: str,
    canonical_file_hashes: dict[str, str],
    grounding: GenerationGroundingV2,
    grounding_artifact_hash: str,
    generator: QuestionGenerator,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestionV4:
    """Generate a v4 question with display passage and question text kept separate."""

    validate_grounding_for_spec(
        grounding,
        spec,
        blueprint_hash=blueprint_hash,
        canonical_file_hashes=canonical_file_hashes,
    )
    prompt = build_display_grounded_prompt(spec, grounding, output_language=output_language)
    return _generate_display_grounded_from_prompt(
        spec,
        grounding=grounding,
        grounding_artifact_hash=grounding_artifact_hash,
        passage=grounding.display_passage_text,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
    )


def _generate_display_grounded_from_prompt(
    spec: QuestionSpec,
    *,
    grounding: GenerationGroundingV2,
    grounding_artifact_hash: str,
    passage: str,
    generator: QuestionGenerator,
    prompt: PromptPayload,
    output_language: str,
) -> GeneratedQuestionV4:
    provider = generator.generate(prompt)
    validate_provider_question(provider.output, spec, grounding=grounding)

    evidence_ids = list(dict.fromkeys(item.evidence_id for item in spec.supporting_evidence))
    identity_and_output: dict[str, object] = {
        "generated_question_version": DISPLAY_GROUNDED_GENERATED_QUESTION_VERSION,
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
        "passage": passage,
        "stem": provider.output.stem,
        "choices": provider.output.choices,
        "correct_choice_index": provider.output.correct_choice_index,
        "explanation": provider.output.explanation,
        "generation_model": provider.model,
        "output_language": output_language,
        "prompt_version": prompt.prompt_version,
        "generation_config_version": GENERATION_CONFIG_VERSION,
        "question_spec_version": spec.question_spec_version,
        "question_spec_config_version": spec.config_version,
        "question_spec_config_hash": spec.config_hash,
        "supporting_book_ids": spec.supporting_book_ids,
        "supporting_evidence_ids": evidence_ids,
        "source_document_ids": spec.source_document_ids,
        "grounding_version": grounding.grounding_version,
        "source_passage_extraction_policy": grounding.source_passage_extraction_policy,
        "source_passage_hash": grounding.source_passage_hash,
        "display_passage_hash": grounding.display_passage_hash,
        "display_normalization_policy": grounding.display_normalization_policy,
        "input_artifact_hash": grounding_artifact_hash,
    }
    return GeneratedQuestionV4(
        generated_question_id=_generated_id(identity_and_output),
        **identity_and_output,
        usage=provider.usage,
    )


def _validate_revision_source(
    spec: QuestionSpec,
    previous: GeneratedQuestion | GeneratedQuestionV4,
    *,
    artifact_hash: str,
    output_language: str,
) -> None:
    expected = {
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
        "question_spec_version": spec.question_spec_version,
        "question_spec_config_version": spec.config_version,
        "question_spec_config_hash": spec.config_hash,
        "supporting_book_ids": spec.supporting_book_ids,
        "supporting_evidence_ids": list(
            dict.fromkeys(item.evidence_id for item in spec.supporting_evidence)
        ),
        "source_document_ids": spec.source_document_ids,
        "input_artifact_hash": artifact_hash,
        "output_language": output_language,
    }
    mismatches = [name for name, value in expected.items() if getattr(previous, name) != value]
    if mismatches:
        raise InputContractError(
            "previous question does not match revision target: " + ", ".join(mismatches)
        )


def revise_question(
    spec: QuestionSpec,
    *,
    artifact_hash: str,
    previous: GeneratedQuestion,
    feedback: str,
    generator: QuestionGenerator,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestion:
    """Generate a validated replacement without changing the authoritative QuestionSpec."""

    validate_supported_spec(spec)
    _validate_revision_source(
        spec,
        previous,
        artifact_hash=artifact_hash,
        output_language=output_language,
    )
    prompt = build_revision_prompt(
        spec,
        previous,
        feedback,
        output_language=output_language,
    )
    return _generate_from_prompt(
        spec,
        artifact_hash=artifact_hash,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
    )


def revise_grounded_question(
    spec: QuestionSpec,
    *,
    blueprint_hash: str,
    canonical_file_hashes: dict[str, str],
    grounding: GenerationGrounding,
    grounding_artifact_hash: str,
    previous: GeneratedQuestion,
    feedback: str,
    generator: QuestionGenerator,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestion:
    """Revise one grounded v3 question without changing its passage or provenance."""

    validate_grounding_for_spec(
        grounding,
        spec,
        blueprint_hash=blueprint_hash,
        canonical_file_hashes=canonical_file_hashes,
    )
    _validate_revision_source(
        spec,
        previous,
        artifact_hash=grounding_artifact_hash,
        output_language=output_language,
    )
    if previous.generated_question_version != GROUNDED_GENERATED_QUESTION_VERSION:
        raise InputContractError("grounded revision requires generated-question-v3 input")
    labels = ("지문", "질문") if output_language == "ko-KR" else ("Passage", "Question")
    expected_prefix = f"{labels[0]}:\n{grounding.passage_text}\n\n{labels[1]}:\n"
    if (
        not previous.stem.startswith(expected_prefix)
        or not previous.stem[len(expected_prefix) :].strip()
    ):
        raise InputContractError("previous grounded stem does not preserve the exact passage")
    prompt = build_grounded_revision_prompt(
        spec,
        grounding,
        previous,
        feedback,
        output_language=output_language,
    )
    return _generate_from_prompt(
        spec,
        artifact_hash=grounding_artifact_hash,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
        generated_question_version=GROUNDED_GENERATED_QUESTION_VERSION,
        grounding=grounding,
    )


def revise_display_grounded_question(
    spec: QuestionSpec,
    *,
    blueprint_hash: str,
    canonical_file_hashes: dict[str, str],
    grounding: GenerationGroundingV2,
    grounding_artifact_hash: str,
    previous: GeneratedQuestion | GeneratedQuestionV4,
    feedback: str,
    generator: QuestionGenerator,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> GeneratedQuestionV4:
    """Revise v4 output while preserving its validated display passage and provenance."""

    validate_grounding_for_spec(
        grounding,
        spec,
        blueprint_hash=blueprint_hash,
        canonical_file_hashes=canonical_file_hashes,
    )
    if previous.generated_question_version != DISPLAY_GROUNDED_GENERATED_QUESTION_VERSION:
        raise InputContractError("display-grounded revision requires generated-question-v4 input")
    if not isinstance(previous, GeneratedQuestionV4):
        raise InputContractError("invalid generated-question-v4 revision input")
    _validate_revision_source(
        spec,
        previous,
        artifact_hash=grounding_artifact_hash,
        output_language=output_language,
    )
    expected_grounding = {
        "grounding_version": grounding.grounding_version,
        "source_passage_extraction_policy": grounding.source_passage_extraction_policy,
        "source_passage_hash": grounding.source_passage_hash,
        "display_passage_hash": grounding.display_passage_hash,
        "display_normalization_policy": grounding.display_normalization_policy,
        "source_document_ids": [grounding.source_document_id],
        "passage": grounding.display_passage_text,
    }
    grounding_mismatches = [
        name for name, value in expected_grounding.items() if getattr(previous, name) != value
    ]
    if grounding_mismatches:
        raise InputContractError(
            "previous question does not match display grounding: " + ", ".join(grounding_mismatches)
        )
    prompt = build_display_grounded_revision_prompt(
        spec,
        grounding,
        previous,
        feedback,
        output_language=output_language,
    )
    return _generate_display_grounded_from_prompt(
        spec,
        grounding=grounding,
        grounding_artifact_hash=grounding_artifact_hash,
        passage=previous.passage,
        generator=generator,
        prompt=prompt,
        output_language=output_language,
    )


def _generate_from_prompt(
    spec: QuestionSpec,
    *,
    artifact_hash: str,
    generator: QuestionGenerator,
    prompt: PromptPayload,
    output_language: str,
    generated_question_version: str = GENERATED_QUESTION_VERSION,
    grounding: GenerationGrounding | None = None,
) -> GeneratedQuestion:
    provider = generator.generate(prompt)
    validate_provider_question(provider.output, spec, grounding=grounding)

    evidence_ids = list(dict.fromkeys(item.evidence_id for item in spec.supporting_evidence))
    stem = provider.output.stem
    if grounding is not None:
        labels = ("지문", "질문") if output_language == "ko-KR" else ("Passage", "Question")
        stem = f"{labels[0]}:\n{grounding.passage_text}\n\n{labels[1]}:\n{provider.output.stem}"
    identity_and_output: dict[str, object] = {
        "generated_question_version": generated_question_version,
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
        "stem": stem,
        "choices": provider.output.choices,
        "correct_choice_index": provider.output.correct_choice_index,
        "explanation": provider.output.explanation,
        "generation_model": provider.model,
        "output_language": output_language,
        "prompt_version": prompt.prompt_version,
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
