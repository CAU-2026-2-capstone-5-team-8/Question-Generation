"""Grounded comprehension/apply vertical-slice tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from question_generation.errors import (
    GeneratedQuestionValidationError,
    InputContractError,
    UnsupportedQuestionSpecError,
)
from question_generation.generation import FakeQuestionGenerator, generate_grounded_question
from question_generation.input_adapter import load_generation_grounding
from question_generation.prompt import build_grounded_prompt
from question_generation.schemas import (
    AssessmentEvidenceRef,
    GenerationGrounding,
    ProviderQuestion,
    QuestionSpec,
)
from question_generation.validation import (
    validate_grounded_comprehension_spec,
    validate_grounding_for_spec,
    validate_provider_question,
)

HASH = "sha256:" + "1" * 64
GROUNDING_ARTIFACT_HASH = "sha256:" + "2" * 64


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _spec() -> QuestionSpec:
    return QuestionSpec(
        question_id="q_" + "a" * 20,
        topic_id="linear-algebra",
        question_type="comprehension",
        cognitive_operation="apply",
        concept_role="covered",
        primary_concept="matrix",
        related_concepts=[],
        prerequisite_concepts=[],
        target_difficulty=2,
        difficulty_version="question-difficulty-v1",
        difficulty_rationale="Apply the concept in one prose-grounded situation.",
        prerequisite_depth=None,
        primary_concept_count=1,
        related_concept_count=0,
        relationship_reasoning_required=False,
        multi_step_required=False,
        abstraction_level="concrete",
        source_text_complexity=0.4,
        assessment_priority=0.8,
        relation_candidate_source=None,
        evidence_summary="Grounded in one analyzed sample chapter.",
        supporting_book_ids=["book_" + "b" * 20],
        supporting_evidence=[
            AssessmentEvidenceRef(
                book_id="book_" + "b" * 20,
                concept_id="matrix",
                evidence_id="doc_grounding",
                evidence_type="sample_chapter",
                mention_count=2,
            )
        ],
        source_document_ids=["doc_grounding"],
        question_spec_version="question-spec-v1",
        config_version="assessment-config-v2-reviewed",
        config_hash=HASH,
    )


def _passage() -> str:
    return (
        "A matrix is a rectangular array with rows and columns. "
        + "The source explains how entries are identified by their row and column positions. " * 12
    ).strip()


def _grounding(spec: QuestionSpec | None = None) -> GenerationGrounding:
    target = spec or _spec()
    passage = _passage()
    return GenerationGrounding(
        schema_version=1,
        grounding_version="generation-grounding-v1",
        question_spec_id=target.question_id,
        question_spec_hash=_hash_json(target.model_dump(mode="json")),
        topic_id=target.topic_id,
        question_type="comprehension",
        cognitive_operation="apply",
        target_difficulty=2,
        primary_concept=target.primary_concept,
        related_concepts=[],
        source_document_id="doc_grounding",
        book_id="book_" + "b" * 20,
        document_type="sample_chapter",
        document_content_hash=HASH,
        passage_text=passage,
        passage_hash=_hash_text(passage),
        passage_extraction_policy="first-concept-sentence-window-v1",
        source_id="source_grounding",
        provider="open_textbook",
        source_type="open_textbook",
        source_url="https://example.test/open-textbook",
        source_retrieved_at="2026-09-01T00:00:00Z",
        license="Creative Commons Attribution 4.0 International License",
        rights_note="Explicitly licensed by the author.",
        source_content_hash=HASH,
        edition_relation="unspecified",
        canonical_file_hashes={
            name: HASH for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
        blueprint_hash=HASH,
    )


def _provider(spec: QuestionSpec | None = None) -> ProviderQuestion:
    target = spec or _spec()
    return ProviderQuestion(
        question_spec_id=target.question_id,
        topic_id=target.topic_id,
        question_type="comprehension",
        cognitive_operation="apply",
        primary_concept=target.primary_concept,
        related_concepts=[],
        target_difficulty=2,
        stem="3행 2열 배열에서 항목의 위치를 올바르게 나타낸 것은 무엇인가?",
        choices=[
            "첫 번째 첨자는 행, 두 번째 첨자는 열을 나타낸다.",
            "첫 번째 첨자는 열, 두 번째 첨자는 행을 나타낸다.",
            "두 첨자는 모두 행의 수만 나타낸다.",
            "두 첨자는 배열의 전체 항목 수만 나타낸다.",
        ],
        correct_choice_index=0,
        explanation="지문은 항목을 행과 열의 위치로 식별한다고 설명하므로 첫 선택지가 맞다.",
    )


def test_grounded_prompt_contains_only_exact_passage_and_bound_identity() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    validate_grounding_for_spec(grounding, spec, blueprint_hash=HASH)

    prompt = build_grounded_prompt(spec, grounding)
    rendered = prompt.render()
    assert prompt.prompt_version == "question-generation-grounded-prompt-v3"
    assert grounding.passage_text in rendered
    assert grounding.passage_hash in rendered
    assert grounding.source_document_id in rendered
    assert "provided passage" in prompt.system_instruction
    assert grounding.source_url not in rendered


def test_grounded_generation_embeds_exact_passage_and_is_deterministic() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    fake = FakeQuestionGenerator(_provider(spec))

    first = generate_grounded_question(
        spec,
        blueprint_hash=HASH,
        grounding=grounding,
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=fake,
    )
    second = generate_grounded_question(
        spec,
        blueprint_hash=HASH,
        grounding=grounding,
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider(spec)),
    )

    assert first == second
    assert first.generated_question_version == "generated-question-v3"
    assert first.prompt_version == "question-generation-grounded-prompt-v3"
    assert first.question_type == "comprehension"
    assert first.cognitive_operation == "apply"
    assert first.target_difficulty == 2
    assert first.input_artifact_hash == GROUNDING_ARTIFACT_HASH
    assert first.source_document_ids == [grounding.source_document_id]
    assert grounding.passage_text in first.stem
    assert first.stem.count(grounding.passage_text) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_spec_id", "q_" + "0" * 20),
        ("topic_id", "operating-systems"),
        ("primary_concept", "vector"),
        ("source_document_id", "doc_other"),
        ("book_id", "book_" + "c" * 20),
        ("blueprint_hash", "sha256:" + "9" * 64),
    ],
)
def test_grounding_identity_mismatch_fails_closed(field: str, value: str) -> None:
    grounding = _grounding().model_copy(update={field: value})
    with pytest.raises(UnsupportedQuestionSpecError):
        validate_grounding_for_spec(grounding, _spec(), blueprint_hash=HASH)


def test_grounding_loader_rejects_altered_passage_hash(tmp_path: Path) -> None:
    payload = _grounding().model_dump(mode="json")
    payload["passage_text"] += " altered"
    path = tmp_path / "grounding.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputContractError, match="passage hash"):
        load_generation_grounding(path)


def test_unsupported_comprehension_targets_remain_fail_closed() -> None:
    integrate = _spec().model_copy(
        update={"cognitive_operation": "integrate", "target_difficulty": 3}
    )
    with pytest.raises(UnsupportedQuestionSpecError, match="comprehension/apply/Level 2"):
        validate_grounded_comprehension_spec(integrate)

    relational = _spec().model_copy(
        update={
            "related_concepts": ["vector"],
            "related_concept_count": 1,
            "relationship_reasoning_required": True,
            "relation_candidate_source": "configured_pair",
        }
    )
    with pytest.raises(UnsupportedQuestionSpecError, match="one source"):
        validate_grounded_comprehension_spec(relational)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_type", "vocabulary"),
        ("cognitive_operation", "recall"),
        ("primary_concept", "vector"),
        ("target_difficulty", 1),
    ],
)
def test_provider_cannot_change_authoritative_grounded_target(field: str, value: object) -> None:
    output = _provider().model_copy(update={field: value})
    with pytest.raises(GeneratedQuestionValidationError, match="authoritative QuestionSpec"):
        validate_provider_question(output, _spec())


def test_generated_question_versions_enforce_their_own_semantics() -> None:
    question = generate_grounded_question(
        _spec(),
        blueprint_hash=HASH,
        grounding=_grounding(),
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider()),
    )
    payload = question.model_dump()
    payload["generated_question_version"] = "generated-question-v2"
    with pytest.raises(ValidationError, match="v2 contains an unsupported target"):
        type(question).model_validate(payload)
