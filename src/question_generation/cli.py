"""Command-line interface for one-spec-at-a-time generation."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import typer

from question_generation.config import load_settings
from question_generation.errors import InputContractError, QuestionGenerationError
from question_generation.gemini import GeminiQuestionGenerator
from question_generation.generation import (
    generate_grounded_question,
    generate_question,
    revise_question,
)
from question_generation.input_adapter import (
    LoadedGenerationGrounding,
    LoadedQuestionSpec,
    load_generation_grounding,
    load_question_spec,
)
from question_generation.prompt import PromptPayload, build_grounded_prompt, build_prompt
from question_generation.schemas import GeneratedQuestion
from question_generation.validation import validate_grounding_for_spec, validate_supported_spec

app = typer.Typer(
    help="Generate validated questions from authoritative ML QuestionSpec JSON artifacts.",
    no_args_is_help=True,
)


def _load_input(
    blueprint: Path | None,
    spec: Path | None,
    question_id: str | None,
) -> LoadedQuestionSpec:
    if (blueprint is None) == (spec is None):
        raise typer.BadParameter("provide exactly one of --blueprint or --spec")
    if blueprint is not None:
        if question_id is None:
            raise typer.BadParameter("--question-id is required with --blueprint")
        return load_question_spec(blueprint, question_id=question_id)
    if question_id is not None:
        raise typer.BadParameter("--question-id is only valid with --blueprint")
    assert spec is not None
    return load_question_spec(spec)


def _fail(exc: Exception) -> None:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _write_json_atomic(output: Path, content: str) -> None:
    """Write a complete JSON artifact before atomically replacing the destination."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_previous_question(path: Path) -> GeneratedQuestion:
    try:
        return GeneratedQuestion.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputContractError(f"invalid previous generated question: {path.name}") from exc


def _load_revision_feedback(path: Path) -> str:
    try:
        feedback = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputContractError(f"cannot read revision feedback: {path.name}") from exc
    feedback = feedback.strip()
    if not feedback:
        raise InputContractError("revision feedback must be nonblank")
    return feedback


def _prepare_generation(
    loaded: LoadedQuestionSpec,
    grounding_path: Path | None,
    *,
    output_language: str,
) -> tuple[PromptPayload, LoadedGenerationGrounding | None]:
    if loaded.spec.question_type == "comprehension":
        if loaded.blueprint_canonical_file_hashes is None:
            raise InputContractError(
                "grounded comprehension requires --blueprint; standalone --spec is unsupported"
            )
        if grounding_path is None:
            raise InputContractError("--grounding is required for comprehension generation")
        grounding = load_generation_grounding(grounding_path)
        validate_grounding_for_spec(
            grounding.grounding,
            loaded.spec,
            blueprint_hash=loaded.artifact_hash,
            canonical_file_hashes=loaded.blueprint_canonical_file_hashes,
        )
        return (
            build_grounded_prompt(
                loaded.spec,
                grounding.grounding,
                output_language=output_language,
            ),
            grounding,
        )
    if grounding_path is not None:
        raise InputContractError("--grounding is valid only for comprehension generation")
    validate_supported_spec(loaded.spec)
    return build_prompt(loaded.spec, output_language=output_language), None


@app.command("render-prompt")
def render_prompt(
    blueprint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    spec: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    question_id: Annotated[str | None, typer.Option("--question-id")] = None,
    grounding_path: Annotated[
        Path | None,
        typer.Option("--grounding", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Render the deterministic prompt without reading GEMINI_API_KEY or calling a provider."""

    try:
        loaded = _load_input(blueprint, spec, question_id)
        settings = load_settings(require_api_key=False)
        prompt, _ = _prepare_generation(
            loaded,
            grounding_path,
            output_language=settings.output_language,
        )
        typer.echo(prompt.render())
    except (QuestionGenerationError, ValueError) as exc:
        _fail(exc)


@app.command("generate")
def generate(
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    blueprint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    spec: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    question_id: Annotated[str | None, typer.Option("--question-id")] = None,
    grounding_path: Annotated[
        Path | None,
        typer.Option("--grounding", exists=True, dir_okay=False, readable=True),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render prompt only; never require a key or call Gemini."),
    ] = False,
) -> None:
    """Generate one question from a blueprint selection or standalone QuestionSpec."""

    try:
        loaded = _load_input(blueprint, spec, question_id)
        settings = load_settings(require_api_key=not dry_run)
        prompt, grounding = _prepare_generation(
            loaded,
            grounding_path,
            output_language=settings.output_language,
        )
        if dry_run:
            typer.echo(prompt.render())
            return
        if output is None:
            raise typer.BadParameter("--output is required unless --dry-run is used")
        resolved_output = output.resolve()
        protected_inputs = {loaded.artifact_path}
        if grounding is not None:
            protected_inputs.add(grounding.artifact_path)
        if resolved_output in protected_inputs:
            raise InputContractError("output path must differ from the input artifact path")
        generator = GeminiQuestionGenerator(settings)
        if grounding is None:
            question = generate_question(
                loaded.spec,
                artifact_hash=loaded.artifact_hash,
                generator=generator,
                output_language=settings.output_language,
            )
        else:
            assert loaded.blueprint_canonical_file_hashes is not None
            question = generate_grounded_question(
                loaded.spec,
                blueprint_hash=loaded.artifact_hash,
                canonical_file_hashes=loaded.blueprint_canonical_file_hashes,
                grounding=grounding.grounding,
                grounding_artifact_hash=grounding.artifact_hash,
                generator=generator,
                output_language=settings.output_language,
            )
        try:
            _write_json_atomic(
                resolved_output,
                json.dumps(question.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            raise InputContractError("could not write generated question output") from exc
        typer.secho("Generated and validated one question.", fg=typer.colors.GREEN)
        typer.echo(f"ID: {question.generated_question_id}")
        typer.echo(f"Target: {question.question_type} / {question.primary_concept}")
        typer.echo(f"Model: {question.generation_model}")
        typer.echo(f"Output: {resolved_output}")
        if question.usage.input_tokens is not None:
            typer.echo(
                "Tokens: "
                f"input={question.usage.input_tokens}, "
                f"output={question.usage.output_tokens}, total={question.usage.total_tokens}"
            )
    except (QuestionGenerationError, ValueError) as exc:
        _fail(exc)


@app.command("revise")
def revise(
    previous: Annotated[
        Path,
        typer.Option("--previous", exists=True, dir_okay=False, readable=True),
    ],
    feedback_file: Annotated[
        Path,
        typer.Option("--feedback-file", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    blueprint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    spec: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    question_id: Annotated[str | None, typer.Option("--question-id")] = None,
) -> None:
    """Revise one generated question from explicit feedback and revalidate it."""

    try:
        loaded = _load_input(blueprint, spec, question_id)
        validate_supported_spec(loaded.spec)
        previous_question = _load_previous_question(previous)
        feedback = _load_revision_feedback(feedback_file)
        resolved_output = output.resolve()
        protected_inputs = {
            loaded.artifact_path.resolve(),
            previous.resolve(),
            feedback_file.resolve(),
        }
        if resolved_output in protected_inputs:
            raise InputContractError("revision output path must differ from every input path")
        settings = load_settings(require_api_key=True)
        generator = GeminiQuestionGenerator(settings)
        question = revise_question(
            loaded.spec,
            artifact_hash=loaded.artifact_hash,
            previous=previous_question,
            feedback=feedback,
            generator=generator,
            output_language=settings.output_language,
        )
        try:
            _write_json_atomic(
                resolved_output,
                json.dumps(question.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            raise InputContractError("could not write revised question output") from exc
        typer.secho("Revised and validated one question.", fg=typer.colors.GREEN)
        typer.echo(f"ID: {question.generated_question_id}")
        typer.echo(f"Source: {previous_question.generated_question_id}")
        typer.echo(f"Target: {question.question_type} / {question.primary_concept}")
        typer.echo(f"Model: {question.generation_model}")
        typer.echo(f"Output: {resolved_output}")
    except (QuestionGenerationError, ValueError) as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
