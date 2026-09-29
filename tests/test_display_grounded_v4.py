"""Source/display-separated grounding and generated-question-v4 tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from question_generation.cli import app
from question_generation.errors import GeneratedQuestionValidationError, InputContractError
from question_generation.generation import (
    FakeQuestionGenerator,
    generate_display_grounded_question,
)
from question_generation.input_adapter import load_generation_grounding, load_question_spec
from question_generation.prompt import build_display_grounded_prompt
from question_generation.schemas import (
    AssessmentEvidenceRef,
    GeneratedQuestionV4,
    GenerationGroundingV2,
    ProviderQuestion,
    QuestionSpec,
)

HASH = "sha256:" + "1" * 64
GROUNDING_ARTIFACT_HASH = "sha256:" + "2" * 64
DOCUMENT_ID = "doc_de934d33d551223812e8"
CANONICAL_HASHES = {
    name: HASH for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
}
SOURCE_PASSAGE = """2.6 DeﬁnitionAnm×n matrix is a rectangular array of numbers withm rows
andn columns. Each number in the matrix is anentry.
We usually denote a matrix with an upper case roman letter. For instance,
A =
(
1 2.2 5
3 4 −7
)
has2 rows and3 columns and so is a2×3 matrix. Read that aloud as “two-by-
three”; the number of rows is always stated ﬁrst. (The matrix has parentheses
around it so that when two matrices are adjacent we can tell where one ends and
the other begins.) We name matrix entries with the corresponding lower-case
letter, so that the entry in the second row and ﬁrst column of the above array
isa2,1 =3."""
DISPLAY_PASSAGE = (
    "2.6 Definition An m×n matrix is a rectangular array of numbers with m rows\n"
    "and n columns. Each number in the matrix is an entry.\n"
    "We usually denote a matrix with an upper case roman letter. For instance,\n"
    "A =\n"
    "(\n"
    "1 2.2 5\n"
    "3 4 −7\n"
    ")\n"
    "has 2 rows and 3 columns and so is a 2×3 matrix. Read that aloud as “"
    "two-by-three”; the number of rows is always stated first. "
    "(The matrix has parentheses\n"
    "around it so that when two matrices are adjacent we can tell where one ends and\n"
    "the other begins.) We name matrix entries with the corresponding lower-case\n"
    "letter, so that the entry in the second row and first column of the above array\n"
    "is a2,1 = 3."
)


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
                evidence_id=DOCUMENT_ID,
                evidence_type="sample_chapter",
                mention_count=2,
            )
        ],
        source_document_ids=[DOCUMENT_ID],
        question_spec_version="question-spec-v1",
        config_version="assessment-config-v2-reviewed",
        config_hash=HASH,
    )


def _grounding(
    spec: QuestionSpec | None = None,
    *,
    blueprint_hash: str = HASH,
) -> GenerationGroundingV2:
    target = spec or _spec()
    return GenerationGroundingV2(
        schema_version=2,
        grounding_version="generation-grounding-v2",
        question_spec_id=target.question_id,
        question_spec_hash=_hash_json(target.model_dump(mode="json")),
        topic_id=target.topic_id,
        question_type="comprehension",
        cognitive_operation="apply",
        target_difficulty=2,
        primary_concept=target.primary_concept,
        related_concepts=[],
        source_document_id=DOCUMENT_ID,
        book_id="book_" + "b" * 20,
        document_type="sample_chapter",
        document_content_hash=HASH,
        source_passage_text=SOURCE_PASSAGE,
        source_passage_hash=_hash_text(SOURCE_PASSAGE),
        source_passage_extraction_policy="first-concept-sentence-window-v1",
        display_passage_text=DISPLAY_PASSAGE,
        display_passage_hash=_hash_text(DISPLAY_PASSAGE),
        display_normalization_policy="pdf-display-normalization-v1",
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
        blueprint_hash=blueprint_hash,
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
        stem="새로운 3행 2열 행렬에서 원소 a₃,₂를 찾는 방법은 무엇인가?",
        choices=[
            "세 번째 행과 두 번째 열이 만나는 원소를 찾는다.",
            "두 번째 행과 세 번째 열이 만나는 원소를 찾는다.",
            "세 번째 행의 원소 개수만 센다.",
            "두 번째 열의 모든 원소를 더한다.",
        ],
        correct_choice_index=0,
        explanation="지문에서 행을 먼저, 열을 다음에 표시하므로 첫 선택지가 맞다.",
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


def test_v4_prompt_uses_display_text_and_retains_raw_provenance_hash() -> None:
    prompt = build_display_grounded_prompt(_spec(), _grounding())
    rendered = prompt.render()

    assert prompt.prompt_version == "question-generation-grounded-prompt-v4"
    assert json.dumps(DISPLAY_PASSAGE, ensure_ascii=False) in rendered
    assert json.dumps(SOURCE_PASSAGE, ensure_ascii=False) not in rendered
    assert _hash_text(SOURCE_PASSAGE) in rendered
    assert _hash_text(DISPLAY_PASSAGE) in rendered
    assert "pdf-display-normalization-v1" in rendered
    assert "grounding.display_passage_text" in rendered


def test_v4_generation_separates_passage_from_stem_and_is_deterministic() -> None:
    first = generate_display_grounded_question(
        _spec(),
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=_grounding(),
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider()),
    )
    second = generate_display_grounded_question(
        _spec(),
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=_grounding(),
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider()),
    )

    assert first == second
    assert first.generated_question_version == "generated-question-v4"
    assert first.prompt_version == "question-generation-grounded-prompt-v4"
    assert first.passage == DISPLAY_PASSAGE
    assert first.stem == _provider().stem
    assert DISPLAY_PASSAGE not in first.stem
    assert first.source_passage_hash == _hash_text(SOURCE_PASSAGE)
    assert first.display_passage_hash == _hash_text(DISPLAY_PASSAGE)
    assert first.display_normalization_policy == "pdf-display-normalization-v1"
    assert first.input_artifact_hash == GROUNDING_ARTIFACT_HASH


def test_v4_rejects_provider_stem_repeating_display_passage() -> None:
    output = _provider().model_copy(
        update={"stem": f"{DISPLAY_PASSAGE}\n\n이 지문을 적용한 질문은 무엇인가?"}
    )
    with pytest.raises(GeneratedQuestionValidationError, match="complete grounding passage"):
        generate_display_grounded_question(
            _spec(),
            blueprint_hash=HASH,
            canonical_file_hashes=CANONICAL_HASHES,
            grounding=_grounding(),
            grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
            generator=FakeQuestionGenerator(output),
        )


def test_v2_loader_rejects_source_display_or_policy_tampering(tmp_path: Path) -> None:
    payload = _grounding().model_dump(mode="json")
    cases = [
        ("source_passage_hash", HASH, "source passage hash"),
        ("display_passage_text", DISPLAY_PASSAGE + " altered", "display passage"),
        ("display_normalization_policy", "pdf-display-normalization-v2", "Input should be"),
    ]
    for field, value, message in cases:
        changed = {**payload, field: value}
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(InputContractError, match=message):
            load_generation_grounding(path)


def test_v4_contract_rejects_passage_hash_mismatch() -> None:
    question = generate_display_grounded_question(
        _spec(),
        blueprint_hash=HASH,
        canonical_file_hashes=CANONICAL_HASHES,
        grounding=_grounding(),
        grounding_artifact_hash=GROUNDING_ARTIFACT_HASH,
        generator=FakeQuestionGenerator(_provider()),
    )
    payload = question.model_dump(mode="json")
    payload["passage"] += " altered"
    with pytest.raises(ValidationError, match="display passage hash"):
        GeneratedQuestionV4.model_validate(payload)


def test_cli_dry_run_accepts_v2_without_key_or_provider_call(tmp_path: Path) -> None:
    spec = _spec()
    blueprint_path = tmp_path / "blueprint.json"
    grounding_path = tmp_path / "grounding.json"
    _write_blueprint(blueprint_path, spec)
    loaded = load_question_spec(blueprint_path, question_id=spec.question_id)
    grounding = _grounding(spec, blueprint_hash=loaded.artifact_hash)
    grounding_path.write_text(grounding.model_dump_json(indent=2) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--dry-run",
            "--blueprint",
            str(blueprint_path),
            "--question-id",
            spec.question_id,
            "--grounding",
            str(grounding_path),
        ],
        env={"GEMINI_API_KEY": ""},
    )

    assert result.exit_code == 0, result.output
    assert "question-generation-grounded-prompt-v4" in result.output
    assert json.dumps(DISPLAY_PASSAGE, ensure_ascii=False) in result.output
    assert json.dumps(SOURCE_PASSAGE, ensure_ascii=False) not in result.output
