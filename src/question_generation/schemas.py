"""Versioned input, provider-output, final-output, and review contracts."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["vocabulary", "background_knowledge", "comprehension"]
CognitiveOperation = Literal[
    "recognize", "recall", "compare", "relate", "apply", "integrate", "infer"
]
ConceptRole = Literal["covered", "prerequisite"]
TargetDifficulty = Literal[1, 2, 3]


def _nonblank(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("value must be nonblank")
    if value != value.strip():
        raise ValueError("value must be trimmed")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AssessmentEvidenceRef(StrictModel):
    concept_id: str
    book_id: str
    evidence_type: str
    evidence_id: str
    mention_count: int = Field(ge=1)

    _validate_strings = field_validator("concept_id", "book_id", "evidence_type", "evidence_id")(
        _nonblank
    )


class QuestionSpec(StrictModel):
    """Local copy of ML's question-spec-v1 contract; no cross-repository import."""

    question_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: QuestionType
    cognitive_operation: CognitiveOperation
    concept_role: ConceptRole
    primary_concept: str
    related_concepts: list[str]
    prerequisite_concepts: list[str]
    target_difficulty: TargetDifficulty
    difficulty_version: str
    difficulty_rationale: str
    prerequisite_depth: int | None = Field(default=None, ge=0)
    primary_concept_count: int = Field(ge=1)
    related_concept_count: int = Field(ge=0)
    relationship_reasoning_required: bool
    multi_step_required: bool
    abstraction_level: Literal["concrete", "relational", "abstract"] | None = None
    source_text_complexity: float | None = Field(default=None, ge=0, le=1)
    assessment_priority: float = Field(ge=0, le=1)
    relation_candidate_source: Literal["configured_pair"] | None = None
    evidence_summary: str
    supporting_book_ids: list[str] = Field(min_length=1)
    supporting_evidence: list[AssessmentEvidenceRef] = Field(min_length=1)
    source_document_ids: list[str]
    question_spec_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _validate_strings = field_validator(
        "topic_id",
        "primary_concept",
        "difficulty_version",
        "difficulty_rationale",
        "evidence_summary",
        "question_spec_version",
        "config_version",
    )(_nonblank)

    @field_validator(
        "related_concepts", "prerequisite_concepts", "supporting_book_ids", "source_document_ids"
    )
    @classmethod
    def list_strings_must_be_nonblank(cls, values: list[str]) -> list[str]:
        for value in values:
            _nonblank(value)
        return values

    @model_validator(mode="after")
    def semantics_match_ml_contract(self) -> "QuestionSpec":
        if self.primary_concept_count != 1:
            raise ValueError("primary_concept_count must be 1")
        if self.related_concept_count != len(self.related_concepts):
            raise ValueError("related_concept_count must match related_concepts")
        if self.primary_concept in self.related_concepts or len(self.related_concepts) != len(
            set(self.related_concepts)
        ):
            raise ValueError("related concepts must be unique and differ from primary")
        if len(self.supporting_book_ids) != len(set(self.supporting_book_ids)):
            raise ValueError("supporting_book_ids must be unique")
        valid_concepts = {self.primary_concept, *self.related_concepts}
        if any(
            item.book_id not in self.supporting_book_ids or item.concept_id not in valid_concepts
            for item in self.supporting_evidence
        ):
            raise ValueError("supporting evidence must match question concepts and books")
        expected_role = (
            "prerequisite" if self.question_type == "background_knowledge" else "covered"
        )
        if self.concept_role != expected_role:
            raise ValueError("question type does not match concept role")
        if self.question_type == "comprehension":
            if not self.source_document_ids or self.source_text_complexity is None:
                raise ValueError("comprehension requires analyzed prose")
            prose_types = {"preface", "introduction", "preview", "sample_chapter", "other"}
            anchored = {
                item.evidence_id
                for item in self.supporting_evidence
                if item.evidence_type in prose_types
            }
            if not set(self.source_document_ids) <= anchored:
                raise ValueError("comprehension source documents require matching prose references")
        elif self.source_document_ids or self.source_text_complexity is not None:
            raise ValueError("non-comprehension specs must not claim prose grounding")
        if self.target_difficulty == 1 and (
            self.related_concepts
            or self.relationship_reasoning_required
            or self.multi_step_required
        ):
            raise ValueError("Level 1 cannot require relation or multi-step reasoning")
        if self.target_difficulty == 2 and self.multi_step_required:
            raise ValueError("Level 2 must not require multiple steps")
        if self.target_difficulty == 3 and not self.multi_step_required:
            raise ValueError("Level 3 requires a multi-step target")
        if bool(self.related_concepts) != (self.relation_candidate_source is not None):
            raise ValueError("related concepts require an explicit candidate source")
        return self


class AssessmentBlueprintEnvelope(BaseModel):
    """Fields needed to select and provenance-check a spec in an ML blueprint artifact."""

    model_config = ConfigDict(extra="allow", strict=True)

    topic_id: str
    question_specs: list[QuestionSpec]
    blueprint_version: Literal["assessment-blueprint-v1"]
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canonical_file_hashes: dict[str, str]
    book_profiles_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def provenance_must_match_specs(self) -> "AssessmentBlueprintEnvelope":
        if not self.question_specs:
            raise ValueError("blueprint contains no question_specs")
        if any(
            item.topic_id != self.topic_id
            or item.config_version != self.config_version
            or item.config_hash != self.config_hash
            for item in self.question_specs
        ):
            raise ValueError("blueprint and QuestionSpec provenance must match")
        required = {"books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl"}
        if set(self.canonical_file_hashes) != required or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in self.canonical_file_hashes.values()
        ):
            raise ValueError("canonical_file_hashes must cover the four canonical JSONL inputs")
        return self


class ProviderQuestion(StrictModel):
    """The only schema Gemini is allowed to return."""

    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: Literal["vocabulary", "background_knowledge"]
    cognitive_operation: Literal["recognize", "recall"]
    primary_concept: str
    related_concepts: list[str]
    target_difficulty: int = Field(ge=1, le=3)
    stem: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_choice_index: int = Field(ge=0, le=3)
    explanation: str

    _validate_strings = field_validator("topic_id", "primary_concept", "stem", "explanation")(
        _nonblank
    )

    @field_validator("related_concepts", "choices")
    @classmethod
    def provider_lists_must_contain_nonblank_strings(cls, values: list[str]) -> list[str]:
        for value in values:
            _nonblank(value)
        return values


class TokenUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GeneratedQuestion(StrictModel):
    generated_question_id: str = Field(pattern=r"^gq_[0-9a-f]{32}$")
    generated_question_version: str
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: Literal["vocabulary", "background_knowledge"]
    cognitive_operation: Literal["recognize", "recall"]
    primary_concept: str
    related_concepts: list[str]
    prerequisite_concepts: list[str]
    target_difficulty: Literal[1]
    difficulty_rationale: str
    evidence_summary: str
    stem: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_choice_index: int = Field(ge=0, le=3)
    explanation: str
    generation_model: str
    output_language: str
    prompt_version: str
    generation_config_version: str
    question_spec_version: str
    question_spec_config_version: str
    question_spec_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    supporting_book_ids: list[str]
    supporting_evidence_ids: list[str]
    source_document_ids: list[str]
    input_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    usage: TokenUsage

    _validate_strings = field_validator(
        "generated_question_version",
        "topic_id",
        "primary_concept",
        "difficulty_rationale",
        "evidence_summary",
        "stem",
        "explanation",
        "generation_model",
        "output_language",
        "prompt_version",
        "generation_config_version",
        "question_spec_version",
        "question_spec_config_version",
    )(_nonblank)

    @field_validator(
        "related_concepts",
        "prerequisite_concepts",
        "choices",
        "supporting_book_ids",
        "supporting_evidence_ids",
        "source_document_ids",
    )
    @classmethod
    def output_lists_must_contain_nonblank_strings(cls, values: list[str]) -> list[str]:
        for value in values:
            _nonblank(value)
        return values

    @model_validator(mode="after")
    def choices_and_provenance_must_be_unique(self) -> "GeneratedQuestion":
        normalized = [" ".join(value.casefold().split()) for value in self.choices]
        if len(set(normalized)) != 4:
            raise ValueError("choices must be unique after normalization")
        for values, label in (
            (self.supporting_book_ids, "supporting_book_ids"),
            (self.supporting_evidence_ids, "supporting_evidence_ids"),
            (self.source_document_ids, "source_document_ids"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


ReviewStatus = Literal["approve", "reject", "needs_revision"]


class HumanQuestionReview(StrictModel):
    generated_question_id: str = Field(pattern=r"^gq_[0-9a-f]{32}$")
    status: ReviewStatus
    correct: bool
    concept_alignment: int = Field(ge=1, le=5)
    difficulty_appropriate: bool
    distractor_quality: int = Field(ge=1, le=5)
    explanation_quality: int = Field(ge=1, le=5)
    notes: str = ""


class FixtureBundle(StrictModel):
    """Small committed contract fixture derived from real ML artifacts."""

    fixture_version: Literal["ml-question-spec-fixture-v1"]
    source_repository: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_artifact: str
    source_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    question_specs: list[QuestionSpec] = Field(min_length=2)
    notes: str


def model_schema_for_audit(model: type[BaseModel]) -> dict[str, Any]:
    """Expose schema generation without coupling callers to Pydantic internals."""

    return model.model_json_schema()
