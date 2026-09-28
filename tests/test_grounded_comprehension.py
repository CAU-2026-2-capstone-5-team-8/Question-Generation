"""Grounded comprehension/apply vertical-slice tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from question_generation.cli import _prepare_generation, app
from question_generation.errors import (
    GeneratedQuestionValidationError,
    InputContractError,
    UnsupportedQuestionSpecError,
)
from question_generation.generation import (
    FakeQuestionGenerator,
    generate_grounded_question,
    revise_grounded_question,
)
from question_generation.input_adapter import load_generation_grounding, load_question_spec
from question_generation.prompt import build_grounded_prompt, build_grounded_revision_prompt
from question_generation.schemas import (
    AssessmentEvidenceRef,
    GeneratedQuestion,
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
CANONICAL_HASHES = {
    name: HASH for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
}


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
        canonical_file_hashes=CANONICAL_HASHES,
        blueprint_hash=HASH,
    )


def _write_blueprint(path: Path, spec: QuestionSpec) -> None:
    path.write_text(
        json.dumps(
            {
                "topic_id": spec.topic_id,
                "question_specs": [spec.model_dump(mode="json")],
                "blueprint_version": "assessment-blueprint-v1",
                "config_version": spec.config_version,
                "config_hash": spec.config_hash,
                "canonical_file_hashes": CANONICAL_HASHES,
                "book_profiles_hash": HASH,
            }
        ),
        encoding="utf-8",
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


def _previous_question(
    spec: QuestionSpec | None = None,
    grounding: GenerationGrounding | None = None,
    *,
    blueprint_hash: str = HASH,
    grounding_artifact_hash: str = GROUNDING_ARTIFACT_HASH,
) -> GeneratedQuestion:
    target = spec or _spec()
    source = grounding or _grounding(target)
    return generate_grounded_question(
        target,
        blueprint_hash=blueprint_hash,
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=source,
        grounding_artifact_hash=grounding_artifact_hash,
        generator=FakeQuestionGenerator(_provider(target)),
    )


def test_grounded_prompt_contains_only_exact_passage_and_bound_identity() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    validate_grounding_for_spec(
        grounding,
        spec,
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
    )

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
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=grounding,
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=fake,
    )
    second = generate_grounded_question(
        spec,
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
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


def test_grounded_revision_prompt_preserves_passage_and_requires_application() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    previous = _previous_question(spec, grounding)
    feedback = "Require application to a new matrix instead of copying the passage example."

    prompt = build_grounded_revision_prompt(spec, grounding, previous, feedback)
    rendered = prompt.render()
    normalized = " ".join(rendered.split())

    assert prompt.prompt_version == "question-generation-grounded-revision-prompt-v1"
    assert rendered.count(grounding.passage_text) == 1
    assert feedback in rendered
    assert previous.generated_question_id in rendered
    assert "applying a rule from the passage to a new situation" in normalized
    assert "must not be answerable by copying an example" in normalized
    assert "Do not modify, rewrite, translate, summarize, or repeat the passage" in normalized


def test_grounded_revision_preserves_provenance_and_gets_new_id() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    previous = _previous_question(spec, grounding)
    revised_output = _provider(spec).model_copy(
        update={
            "stem": "새로운 4행 2열 배열의 원소 a₃,₂를 올바르게 찾는 방법은 무엇인가?",
            "choices": [
                "세 번째 행과 두 번째 열이 만나는 원소를 찾는다.",
                "두 번째 행과 세 번째 열이 만나는 원소를 찾는다.",
                "세 번째 행의 모든 원소를 더한다.",
                "두 번째 열의 원소 개수를 센다.",
            ],
            "explanation": "지문은 첫 첨자가 행, 둘째 첨자가 열의 위치를 나타낸다고 설명한다.",
        }
    )
    fake = FakeQuestionGenerator(revised_output)

    revised = revise_grounded_question(
        spec,
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=grounding,
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        previous=previous,
        feedback="Apply the row and column convention to a new matrix.",
        generator=fake,
    )

    assert revised.generated_question_version == "generated-question-v3"
    assert revised.prompt_version == "question-generation-grounded-revision-prompt-v1"
    assert revised.generated_question_id != previous.generated_question_id
    assert revised.question_spec_id == previous.question_spec_id
    assert revised.input_artifact_hash == previous.input_artifact_hash
    assert revised.source_document_ids == previous.source_document_ids
    assert revised.stem.startswith(f"지문:\n{grounding.passage_text}\n\n질문:\n")
    assert revised.stem.count(grounding.passage_text) == 1
    assert len(fake.prompts) == 1


def test_grounded_revision_rejects_changed_previous_passage_before_provider() -> None:
    spec = _spec()
    grounding = _grounding(spec)
    previous = _previous_question(spec, grounding).model_copy(
        update={"stem": "지문:\nchanged passage\n\n질문:\nquestion"}
    )
    fake = FakeQuestionGenerator(_provider(spec))

    with pytest.raises(InputContractError, match="exact passage"):
        revise_grounded_question(
            spec,
            blueprint_hash=HASH,
            canonical_file_hashes=CANONICAL_HASHES,
            grounding=grounding,
            grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
            previous=previous,
            feedback="Apply the rule to a new situation.",
            generator=fake,
        )

    assert fake.prompts == []


def test_cli_revises_grounded_question_without_overwriting_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    blueprint_path = tmp_path / "blueprint.json"
    grounding_path = tmp_path / "grounding.json"
    previous_path = tmp_path / "previous.json"
    feedback_path = tmp_path / "feedback.txt"
    output_path = tmp_path / "revised.json"
    _write_blueprint(blueprint_path, spec)
    loaded = load_question_spec(blueprint_path, question_id=spec.question_id)
    grounding = _grounding(spec).model_copy(update={"blueprint_hash": loaded.artifact_hash})
    grounding_path.write_text(grounding.model_dump_json(indent=2) + "\n", encoding="utf-8")
    loaded_grounding = load_generation_grounding(grounding_path)
    previous = _previous_question(
        spec,
        grounding,
        blueprint_hash=loaded.artifact_hash,
        grounding_artifact_hash=loaded_grounding.artifact_hash,
    )
    previous_path.write_text(previous.model_dump_json(indent=2) + "\n", encoding="utf-8")
    feedback_path.write_text("Apply the passage rule to a new matrix.\n", encoding="utf-8")
    previous_bytes = previous_path.read_bytes()
    grounding_bytes = grounding_path.read_bytes()
    revised_output = _provider(spec).model_copy(
        update={"stem": "새로운 4행 2열 배열에서 a₃,₂의 위치를 올바르게 찾은 것은?"}
    )
    fake = FakeQuestionGenerator(revised_output)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("question_generation.cli.GeminiQuestionGenerator", lambda settings: fake)

    result = CliRunner().invoke(
        app,
        [
            "revise",
            "--blueprint",
            str(blueprint_path),
            "--question-id",
            spec.question_id,
            "--grounding",
            str(grounding_path),
            "--previous",
            str(previous_path),
            "--feedback-file",
            str(feedback_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    revised = GeneratedQuestion.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert revised.generated_question_version == "generated-question-v3"
    assert revised.prompt_version == "question-generation-grounded-revision-prompt-v1"
    assert revised.generated_question_id != previous.generated_question_id
    assert revised.input_artifact_hash == loaded_grounding.artifact_hash
    assert grounding.passage_text in revised.stem
    assert previous_path.read_bytes() == previous_bytes
    assert grounding_path.read_bytes() == grounding_bytes
    assert fake.prompts[0].prompt_version == "question-generation-grounded-revision-prompt-v1"


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
        validate_grounding_for_spec(
            grounding,
            _spec(),
            blueprint_hash=HASH,
            canonical_file_hashes=CANONICAL_HASHES,
        )


def test_grounding_canonical_hash_mismatch_fails_closed() -> None:
    altered_hashes = {**CANONICAL_HASHES, "documents.jsonl": "sha256:" + "9" * 64}
    grounding = _grounding().model_copy(update={"canonical_file_hashes": altered_hashes})
    with pytest.raises(UnsupportedQuestionSpecError, match="canonical hashes"):
        validate_grounding_for_spec(
            grounding,
            _spec(),
            blueprint_hash=HASH,
            canonical_file_hashes=CANONICAL_HASHES,
        )


def test_grounded_comprehension_rejects_standalone_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec().model_dump(mode="json")), encoding="utf-8")
    loaded = load_question_spec(spec_path)

    assert loaded.blueprint_canonical_file_hashes is None
    with pytest.raises(InputContractError, match="requires --blueprint"):
        _prepare_generation(
            loaded,
            tmp_path / "unused-grounding.json",
            output_language="ko-KR",
        )


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
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=_grounding(),
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider()),
    )
    payload = question.model_dump()
    payload["generated_question_version"] = "generated-question-v2"
    with pytest.raises(ValidationError, match="v2 contains an unsupported target"):
        type(question).model_validate(payload)
