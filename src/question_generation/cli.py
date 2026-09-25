"""Command-line interface for one-spec-at-a-time generation."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import typer

from question_generation.config import load_settings
from question_generation.errors import InputContractError, QuestionGenerationError
from question_generation.gemini import GeminiQuestionGenerator
from question_generation.generation import generate_question
from question_generation.input_adapter import LoadedQuestionSpec, load_question_spec
from question_generation.prompt import build_prompt
from question_generation.validation import validate_supported_spec

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


@app.command("render-prompt")
def render_prompt(
    blueprint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    spec: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    question_id: Annotated[str | None, typer.Option("--question-id")] = None,
) -> None:
    """Render the deterministic prompt without reading GEMINI_API_KEY or calling a provider."""

    try:
        loaded = _load_input(blueprint, spec, question_id)
        validate_supported_spec(loaded.spec)
        settings = load_settings(require_api_key=False)
        typer.echo(
            build_prompt(
                loaded.spec,
                output_language=settings.output_language,
            ).render()
        )
    except (QuestionGenerationError, ValueError) as exc:
        _fail(exc)


@app.command("generate")
def generate(
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    blueprint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    spec: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    question_id: Annotated[str | None, typer.Option("--question-id")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render prompt only; never require a key or call Gemini."),
    ] = False,
) -> None:
    """Generate one question from a blueprint selection or standalone QuestionSpec."""

    try:
        loaded = _load_input(blueprint, spec, question_id)
        validate_supported_spec(loaded.spec)
        if dry_run:
            settings = load_settings(require_api_key=False)
            typer.echo(
                build_prompt(
                    loaded.spec,
                    output_language=settings.output_language,
                ).render()
            )
            return
        if output is None:
            raise typer.BadParameter("--output is required unless --dry-run is used")
        resolved_output = output.resolve()
        if resolved_output == loaded.artifact_path:
            raise InputContractError("output path must differ from the input artifact path")
        settings = load_settings(require_api_key=True)
        generator = GeminiQuestionGenerator(settings)
        question = generate_question(
            loaded.spec,
            artifact_hash=loaded.artifact_hash,
            generator=generator,
            output_language=settings.output_language,
        )
        _write_json_atomic(
            resolved_output,
            json.dumps(question.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        )
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


if __name__ == "__main__":
    app()
