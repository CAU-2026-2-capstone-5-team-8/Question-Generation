import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from question_generation.errors import (
    GeneratedQuestionValidationError,
    InputContractError,
    UnsupportedQuestionSpecError,
)
from question_generation.input_adapter import load_question_spec
from question_generation.schemas import ProviderQuestion, QuestionSpec
from question_generation.validation import validate_provider_question, validate_supported_spec
from tests.conftest import write_spec


def test_single_spec_input_parsing_preserves_provenance(
    tmp_path: Path, vocabulary_spec: QuestionSpec
) -> None:
    path = tmp_path / "spec.json"
    write_spec(path, vocabulary_spec)
    loaded = load_question_spec(path)
    assert loaded.spec == vocabulary_spec
    assert loaded.artifact_hash.startswith("sha256:")
    assert loaded.spec.config_hash == vocabulary_spec.config_hash
    assert loaded.spec.supporting_evidence == vocabulary_spec.supporting_evidence


def test_blueprint_input_selects_exact_question(
    tmp_path: Path, vocabulary_spec: QuestionSpec
) -> None:
    digest = "sha256:" + "a" * 64
    payload = {
        "topic_id": vocabulary_spec.topic_id,
        "question_specs": [vocabulary_spec.model_dump(mode="json")],
        "blueprint_version": "assessment-blueprint-v1",
        "config_version": vocabulary_spec.config_version,
        "config_hash": vocabulary_spec.config_hash,
        "canonical_file_hashes": {
            name: digest
            for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
        "book_profiles_hash": digest,
        "concept_pool": {"ignored_by_question_generation": True},
    }
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_question_spec(path, question_id=vocabulary_spec.question_id)
    assert loaded.spec == vocabulary_spec
    with pytest.raises(InputContractError, match="not found exactly once"):
        load_question_spec(path, question_id="q_00000000000000000000")


def test_input_rejects_unknown_question_spec_version(
    tmp_path: Path, vocabulary_spec: QuestionSpec
) -> None:
    path = tmp_path / "future.json"
    write_spec(
        path, vocabulary_spec.model_copy(update={"question_spec_version": "question-spec-v2"})
    )
    with pytest.raises(InputContractError, match="unsupported question_spec_version"):
        load_question_spec(path)


def test_v1_rejects_comprehension_and_relation_reasoning(vocabulary_spec: QuestionSpec) -> None:
    comprehension = vocabulary_spec.model_copy(
        update={
            "question_type": "comprehension",
            "cognitive_operation": "apply",
            "source_document_ids": ["doc_example"],
            "source_text_complexity": 0.5,
        }
    )
    with pytest.raises(UnsupportedQuestionSpecError, match="comprehension"):
        validate_supported_spec(comprehension)

    comparison = vocabulary_spec.model_copy(
        update={
            "cognitive_operation": "compare",
            "related_concepts": ["thread"],
            "related_concept_count": 1,
            "relationship_reasoning_required": True,
            "relation_candidate_source": "configured_pair",
            "target_difficulty": 2,
        }
    )
    with pytest.raises(UnsupportedQuestionSpecError, match="only vocabulary/recognize"):
        validate_supported_spec(comparison)


def test_provider_schema_enforces_four_choices_and_index(
    vocabulary_output: ProviderQuestion,
) -> None:
    payload = vocabulary_output.model_dump()
    with pytest.raises(ValidationError):
        ProviderQuestion.model_validate(payload | {"choices": payload["choices"][:3]})
    with pytest.raises(ValidationError):
        ProviderQuestion.model_validate(payload | {"correct_choice_index": 4})
    with pytest.raises(ValidationError):
        ProviderQuestion.model_validate(payload | {"stem": "   "})


def test_domain_validation_rejects_duplicate_choices(
    vocabulary_spec: QuestionSpec, vocabulary_output: ProviderQuestion
) -> None:
    duplicate = vocabulary_output.model_copy(
        update={"choices": ["Same answer", " same   answer ", "Third", "Fourth"]}
    )
    with pytest.raises(GeneratedQuestionValidationError, match="unique"):
        validate_provider_question(duplicate, vocabulary_spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_spec_id", "q_00000000000000000000"),
        ("topic_id", "linear-algebra"),
        ("primary_concept", "thread"),
        ("question_type", "background_knowledge"),
        ("cognitive_operation", "recall"),
        ("related_concepts", ["thread"]),
    ],
)
def test_domain_validation_rejects_spec_identity_changes(
    field: str,
    value: object,
    vocabulary_spec: QuestionSpec,
    vocabulary_output: ProviderQuestion,
) -> None:
    changed = vocabulary_output.model_copy(update={field: value})
    with pytest.raises(GeneratedQuestionValidationError, match="authoritative"):
        validate_provider_question(changed, vocabulary_spec)


def test_domain_validation_rejects_difficulty_change(
    vocabulary_spec: QuestionSpec, vocabulary_output: ProviderQuestion
) -> None:
    different_level_spec = vocabulary_spec.model_copy(update={"target_difficulty": 2})
    with pytest.raises(GeneratedQuestionValidationError, match="target_difficulty"):
        validate_provider_question(vocabulary_output, different_level_spec)


def test_domain_validation_rejects_answer_leak_placeholder_and_wrong_choice_reference(
    vocabulary_spec: QuestionSpec, vocabulary_output: ProviderQuestion
) -> None:
    answer = vocabulary_output.choices[vocabulary_output.correct_choice_index]
    leaked = vocabulary_output.model_copy(update={"stem": f"Why is {answer} correct?"})
    with pytest.raises(GeneratedQuestionValidationError, match="leaked"):
        validate_provider_question(leaked, vocabulary_spec)

    placeholder = vocabulary_output.model_copy(update={"explanation": "TODO: explain later"})
    with pytest.raises(GeneratedQuestionValidationError, match="placeholder"):
        validate_provider_question(placeholder, vocabulary_spec)

    wrong_reference = vocabulary_output.model_copy(
        update={"explanation": "Choice 2 is correct because it describes execution."}
    )
    with pytest.raises(GeneratedQuestionValidationError, match="choice number"):
        validate_provider_question(wrong_reference, vocabulary_spec)
