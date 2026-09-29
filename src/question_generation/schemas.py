"""Versioned input, provider-output, final-output, and review contracts."""

import hashlib
import json
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


class GenerationGrounding(StrictModel):
    """Local copy of ML's generation-grounding-v1 artifact contract."""

    schema_version: Literal[1]
    grounding_version: Literal["generation-grounding-v1"]
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    question_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    topic_id: str
    question_type: Literal["comprehension"]
    cognitive_operation: Literal["apply"]
    target_difficulty: Literal[2]
    primary_concept: str
    related_concepts: list[str]
    source_document_id: str
    book_id: str
    document_type: Literal["preface", "introduction", "preview", "sample_chapter", "other"]
    document_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passage_text: str = Field(min_length=600, max_length=1800)
    passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passage_extraction_policy: Literal["first-concept-sentence-window-v1"]
    source_id: str
    provider: str
    source_type: str
    source_url: str
    source_retrieved_at: str
    license: str
    rights_note: str | None = None
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    edition_relation: Literal["exact", "same_work", "unspecified"]
    canonical_file_hashes: dict[str, str]
    blueprint_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _validate_strings = field_validator(
        "topic_id",
        "primary_concept",
        "source_document_id",
        "book_id",
        "source_id",
        "provider",
        "source_type",
        "source_url",
        "source_retrieved_at",
        "license",
    )(_nonblank)

    @field_validator("canonical_file_hashes")
    @classmethod
    def canonical_hashes_must_cover_exact_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl"}
        if set(value) != required or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in value.values()
        ):
            raise ValueError("canonical file hashes must cover all four JSONL inputs")
        return value

    @model_validator(mode="after")
    def passage_and_target_must_be_consistent(self) -> "GenerationGrounding":
        actual = "sha256:" + hashlib.sha256(self.passage_text.encode("utf-8")).hexdigest()
        if self.passage_hash != actual:
            raise ValueError("passage hash does not match passage text")
        if self.related_concepts:
            raise ValueError("generation-grounding-v1 supports only single-concept apply targets")
        return self


_REVIEWED_DISPLAY_REPLACEMENTS = {
    (
        "doc_de934d33d551223812e8",
        "sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0",
    ): (
        ("DeﬁnitionAnm×n", "Definition An m×n"),
        ("withm rows\nandn columns", "with m rows\nand n columns"),
        ("anentry", "an entry"),
        (
            "has2 rows and3 columns and so is a2×3 matrix",
            "has 2 rows and 3 columns and so is a 2×3 matrix",
        ),
        ("two-by-\nthree", "two-by-three"),
        ("stated ﬁrst", "stated first"),
        ("row and ﬁrst column", "row and first column"),
        ("isa2,1 =3", "is a2,1 = 3"),
    ),
}


def _reviewed_display_passage(
    source_passage_text: str,
    *,
    source_document_id: str,
    source_passage_hash: str,
) -> str:
    actual_hash = "sha256:" + hashlib.sha256(source_passage_text.encode("utf-8")).hexdigest()
    if source_passage_hash != actual_hash:
        raise ValueError("source passage hash does not match source passage text")
    replacements = _REVIEWED_DISPLAY_REPLACEMENTS.get((source_document_id, source_passage_hash))
    if replacements is None:
        raise ValueError("display policy has no reviewed rules for this source passage")
    display_passage = source_passage_text
    for original, replacement in replacements:
        if display_passage.count(original) != 1:
            raise ValueError("reviewed display policy input does not match its source fragment")
        display_passage = display_passage.replace(original, replacement, 1)
    return display_passage


class GenerationGroundingV2(StrictModel):
    """Local copy of ML's source/display-separated generation-grounding-v2 contract."""

    schema_version: Literal[2]
    grounding_version: Literal["generation-grounding-v2"]
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    question_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    topic_id: str
    question_type: Literal["comprehension"]
    cognitive_operation: Literal["apply"]
    target_difficulty: Literal[2]
    primary_concept: str
    related_concepts: list[str]
    source_document_id: str
    book_id: str
    document_type: Literal["preface", "introduction", "preview", "sample_chapter", "other"]
    document_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_passage_text: str = Field(min_length=600, max_length=1800)
    source_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_passage_extraction_policy: Literal["first-concept-sentence-window-v1"]
    display_passage_text: str = Field(min_length=600, max_length=2000)
    display_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    display_normalization_policy: Literal["pdf-display-normalization-v1"]
    source_id: str
    provider: str
    source_type: str
    source_url: str
    source_retrieved_at: str
    license: str
    rights_note: str | None = None
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    edition_relation: Literal["exact", "same_work", "unspecified"]
    canonical_file_hashes: dict[str, str]
    blueprint_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _validate_strings = field_validator(
        "topic_id",
        "primary_concept",
        "source_document_id",
        "book_id",
        "source_id",
        "provider",
        "source_type",
        "source_url",
        "source_retrieved_at",
        "license",
    )(_nonblank)

    @field_validator("canonical_file_hashes")
    @classmethod
    def canonical_hashes_must_cover_exact_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl"}
        if set(value) != required or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in value.values()
        ):
            raise ValueError("canonical file hashes must cover all four JSONL inputs")
        return value

    @model_validator(mode="after")
    def source_display_and_target_must_be_consistent(self) -> "GenerationGroundingV2":
        expected_display = _reviewed_display_passage(
            self.source_passage_text,
            source_document_id=self.source_document_id,
            source_passage_hash=self.source_passage_hash,
        )
        if self.display_passage_text != expected_display:
            raise ValueError("display passage does not match the deterministic display policy")
        display_hash = (
            "sha256:" + hashlib.sha256(self.display_passage_text.encode("utf-8")).hexdigest()
        )
        if self.display_passage_hash != display_hash:
            raise ValueError("display passage hash does not match display passage text")
        if self.related_concepts:
            raise ValueError("generation-grounding-v2 supports only single-concept apply targets")
        return self


class ProviderQuestion(StrictModel):
    """The only schema Gemini is allowed to return."""

    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: Literal["vocabulary", "background_knowledge", "comprehension"]
    cognitive_operation: Literal["recognize", "recall", "apply"]
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
    question_type: Literal["vocabulary", "background_knowledge", "comprehension"]
    cognitive_operation: Literal["recognize", "recall", "apply"]
    primary_concept: str
    related_concepts: list[str]
    prerequisite_concepts: list[str]
    target_difficulty: Literal[1, 2]
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
        if self.generated_question_version == "generated-question-v2":
            if (
                (self.question_type, self.cognitive_operation)
                not in {
                    ("vocabulary", "recognize"),
                    ("background_knowledge", "recall"),
                }
                or self.target_difficulty != 1
                or self.related_concepts
                or self.source_document_ids
            ):
                raise ValueError("generated-question-v2 contains an unsupported target")
        elif self.generated_question_version == "generated-question-v3":
            if (
                (self.question_type, self.cognitive_operation) != ("comprehension", "apply")
                or self.target_difficulty != 2
                or self.related_concepts
                or len(self.source_document_ids) != 1
            ):
                raise ValueError("generated-question-v3 supports only grounded comprehension/apply")
        else:
            raise ValueError("unsupported generated_question_version")
        return self


class GeneratedQuestionV4(StrictModel):
    """Display-grounded question with passage and question text in separate fields."""

    generated_question_id: str = Field(pattern=r"^gq_[0-9a-f]{32}$")
    generated_question_version: Literal["generated-question-v4"]
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: Literal["comprehension"]
    cognitive_operation: Literal["apply"]
    primary_concept: str
    related_concepts: list[str]
    prerequisite_concepts: list[str]
    target_difficulty: Literal[2]
    difficulty_rationale: str
    evidence_summary: str
    passage: str = Field(min_length=600, max_length=2000)
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
    source_document_ids: list[str] = Field(min_length=1, max_length=1)
    grounding_version: Literal["generation-grounding-v2"]
    source_passage_extraction_policy: Literal["first-concept-sentence-window-v1"]
    source_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    display_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    display_normalization_policy: Literal["pdf-display-normalization-v1"]
    input_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    usage: TokenUsage

    _validate_strings = field_validator(
        "topic_id",
        "primary_concept",
        "difficulty_rationale",
        "evidence_summary",
        "passage",
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
    def display_grounding_and_provenance_must_be_consistent(self) -> "GeneratedQuestionV4":
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
        if self.related_concepts:
            raise ValueError("generated-question-v4 supports one non-relational concept")
        actual_display_hash = "sha256:" + hashlib.sha256(self.passage.encode("utf-8")).hexdigest()
        if self.display_passage_hash != actual_display_hash:
            raise ValueError("display passage hash does not match passage")
        identity_and_output = self.model_dump(
            mode="json",
            exclude={"generated_question_id", "usage"},
        )
        canonical = json.dumps(
            identity_and_output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_id = "gq_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        if self.generated_question_id != expected_id:
            raise ValueError("generated question ID does not match v4 identity and output")
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
