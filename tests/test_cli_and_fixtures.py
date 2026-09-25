import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from question_generation.cli import app
from question_generation.config import DEFAULT_MODEL, DEFAULT_OUTPUT_LANGUAGE, load_settings
from question_generation.errors import GenerationConfigurationError
from question_generation.generation import FakeQuestionGenerator, generate_question
from question_generation.schemas import (
    FixtureBundle,
    HumanQuestionReview,
    ProviderQuestion,
    QuestionSpec,
)
from tests.conftest import write_spec


def output_for(spec: QuestionSpec) -> ProviderQuestion:
    if spec.question_type == "vocabulary":
        stem = "Which description best defines an operating-system process?"
        choices = [
            "A program in execution with its own managed state",
            "A directory that contains only executable files",
            "A processor that executes exactly one instruction",
            "A translation from source code to machine code",
        ]
        explanation = "A process is a program in execution together with managed state."
    else:
        stem = "What is the primary purpose of a variable in programming?"
        choices = [
            "To give a name to a value that a program can use or change",
            "To guarantee that every program finishes without an error",
            "To convert every instruction directly into hardware",
            "To prevent a program from making any decisions",
        ]
        explanation = "A variable names stored data so a program can read or update it."
    return ProviderQuestion(
        question_spec_id=spec.question_id,
        topic_id=spec.topic_id,
        question_type=spec.question_type,
        cognitive_operation=spec.cognitive_operation,
        primary_concept=spec.primary_concept,
        related_concepts=spec.related_concepts,
        target_difficulty=1,
        stem=stem,
        choices=choices,
        correct_choice_index=0,
        explanation=explanation,
    )


def write_blueprint(path: Path, spec: QuestionSpec) -> None:
    digest = "sha256:" + "a" * 64
    path.write_text(
        json.dumps(
            {
                "topic_id": spec.topic_id,
                "question_specs": [spec.model_dump(mode="json")],
                "blueprint_version": "assessment-blueprint-v1",
                "config_version": spec.config_version,
                "config_hash": spec.config_hash,
                "canonical_file_hashes": {
                    name: digest
                    for name in (
                        "books.jsonl",
                        "documents.jsonl",
                        "toc.jsonl",
                        "sources.jsonl",
                    )
                },
                "book_profiles_hash": digest,
                "concept_pool": {"ignored_by_question_generation": True},
            }
        ),
        encoding="utf-8",
    )


def test_missing_key_has_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        load_settings(require_api_key=True)
    except GenerationConfigurationError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing key must fail")


def test_default_model_is_current_flash_lite(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("QUESTION_GENERATION_MODEL", raising=False)
    settings = load_settings(require_api_key=True)
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert settings.model == DEFAULT_MODEL


def test_default_and_overridden_output_language(monkeypatch) -> None:
    monkeypatch.delenv("QUESTION_GENERATION_LANGUAGE", raising=False)
    assert (
        load_settings(require_api_key=False).output_language == DEFAULT_OUTPUT_LANGUAGE == "ko-KR"
    )

    monkeypatch.setenv("QUESTION_GENERATION_LANGUAGE", "en-US")
    assert load_settings(require_api_key=False).output_language == "en-US"


def test_cli_dry_run_needs_no_key(
    tmp_path: Path, monkeypatch, vocabulary_spec: QuestionSpec
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    spec_path = tmp_path / "spec.json"
    write_spec(spec_path, vocabulary_spec)
    result = CliRunner().invoke(app, ["generate", "--spec", str(spec_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "question-generation-prompt-v2" in result.output
    assert "output language is ko-KR" in result.output
    assert vocabulary_spec.question_id in result.output


def test_cli_generate_without_key_fails_before_provider(
    tmp_path: Path, monkeypatch, vocabulary_spec: QuestionSpec
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    spec_path = tmp_path / "spec.json"
    write_spec(spec_path, vocabulary_spec)
    result = CliRunner().invoke(
        app,
        ["generate", "--spec", str(spec_path), "--output", str(tmp_path / "out.json")],
    )
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


@pytest.mark.parametrize("output_style", ["same", "relative", "symlink"])
def test_cli_rejects_spec_output_collision_before_provider(
    tmp_path: Path,
    monkeypatch,
    vocabulary_spec: QuestionSpec,
    output_style: str,
) -> None:
    spec_path = tmp_path / "input.json"
    write_spec(spec_path, vocabulary_spec)
    original = spec_path.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def unexpected_generator(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider must not be constructed for a path collision")

    monkeypatch.setattr("question_generation.cli.GeminiQuestionGenerator", unexpected_generator)
    if output_style == "same":
        output = str(spec_path)
    elif output_style == "relative":
        output = "./input.json"
    else:
        link = tmp_path / "input-link.json"
        link.symlink_to(spec_path)
        output = str(link)

    result = CliRunner().invoke(
        app,
        ["generate", "--spec", str(spec_path), "--output", output],
    )

    assert result.exit_code == 1
    assert "output path must differ from the input artifact path" in result.output
    assert spec_path.read_bytes() == original


def test_cli_rejects_blueprint_output_collision_before_provider(
    tmp_path: Path, monkeypatch, vocabulary_spec: QuestionSpec
) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    write_blueprint(blueprint_path, vocabulary_spec)
    original = blueprint_path.read_bytes()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def unexpected_generator(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider must not be constructed for a path collision")

    monkeypatch.setattr("question_generation.cli.GeminiQuestionGenerator", unexpected_generator)
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--blueprint",
            str(blueprint_path),
            "--question-id",
            vocabulary_spec.question_id,
            "--output",
            str(blueprint_path),
        ],
    )

    assert result.exit_code == 1
    assert "output path must differ from the input artifact path" in result.output
    assert blueprint_path.read_bytes() == original


def test_cli_writes_distinct_output_atomically(
    tmp_path: Path, monkeypatch, vocabulary_spec: QuestionSpec
) -> None:
    spec_path = tmp_path / "input.json"
    output_path = tmp_path / "generated" / "question.json"
    write_spec(spec_path, vocabulary_spec)
    fake = FakeQuestionGenerator(output_for(vocabulary_spec))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "question_generation.cli.GeminiQuestionGenerator",
        lambda settings: fake,
    )

    result = CliRunner().invoke(
        app,
        ["generate", "--spec", str(spec_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output_path.read_text(encoding="utf-8"))["question_spec_id"] == (
        vocabulary_spec.question_id
    )
    assert len(fake.prompts) == 1
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_cli_reports_atomic_output_write_failure(
    tmp_path: Path, monkeypatch, vocabulary_spec: QuestionSpec
) -> None:
    spec_path = tmp_path / "input.json"
    output_path = tmp_path / "generated.json"
    write_spec(spec_path, vocabulary_spec)
    fake = FakeQuestionGenerator(output_for(vocabulary_spec))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "question_generation.cli.GeminiQuestionGenerator",
        lambda settings: fake,
    )

    def failed_write(output: Path, content: str) -> None:
        raise OSError("sensitive filesystem detail")

    monkeypatch.setattr("question_generation.cli._write_json_atomic", failed_write)
    result = CliRunner().invoke(
        app,
        ["generate", "--spec", str(spec_path), "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert "could not write generated question output" in result.output
    assert "sensitive filesystem detail" not in result.output
    assert not output_path.exists()


def test_real_ml_fixture_generates_both_supported_types(fixture_bundle: FixtureBundle) -> None:
    assert {item.question_type for item in fixture_bundle.question_specs} == {
        "vocabulary",
        "background_knowledge",
    }
    generated = [
        generate_question(
            spec,
            artifact_hash=fixture_bundle.source_artifact_hash,
            generator=FakeQuestionGenerator(output_for(spec)),
        )
        for spec in fixture_bundle.question_specs
    ]
    assert len({item.generated_question_id for item in generated}) == 2
    assert all(len(item.choices) == 4 for item in generated)
    assert all(
        item.input_artifact_hash == fixture_bundle.source_artifact_hash for item in generated
    )
    assert all(item.output_language == "ko-KR" for item in generated)


def test_fixture_file_has_documented_source_hash() -> None:
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "ml_question_specs.json").read_text(encoding="utf-8")
    )
    assert payload["source_artifact_hash"] == (
        "sha256:3822916d24b155b50ee3efb2581b43aa5587926eb36b3f7e958bb1f740d47f9c"
    )


def test_human_review_template_matches_schema() -> None:
    path = Path(__file__).parents[1] / "examples" / "human_review.example.jsonl"
    review = HumanQuestionReview.model_validate_json(path.read_text(encoding="utf-8"))
    assert review.status == "approve"
    assert review.concept_alignment == 5
