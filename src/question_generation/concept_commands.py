"""Additive CLI for concept generation; old generate/revise commands are unchanged."""

from pathlib import Path
from typing import Annotated

import typer

from question_generation.concept_contract import (
    ConceptBlueprint,
    ConceptGeneratedQuestion,
    content_hash,
)
from question_generation.concept_generation import (
    PROMPT_VERSION,
    file_hash,
    generate_concept_question,
    render_concept_prompt,
    write_review_packet,
)
from question_generation.concept_review import (
    ai_review_status,
    load_candidates,
    load_worksheet,
    prepare_review,
    review_status,
)
from question_generation.config import load_settings

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("local-drafts")
def local_drafts(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    drafts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
):
    """Package locally authored candidates without claiming remote generation or approval."""
    import json

    from question_generation.concept_contract import ConceptProviderQuestion
    from question_generation.concept_generation import assemble_question

    bank = ConceptBlueprint.model_validate_json(blueprint.read_text())
    rows = json.loads(drafts.read_text())
    by_cell = {(r["concept_id"], r["ability"]): r for r in rows}
    if len(by_cell) != len(rows) or set(by_cell) != {
        (s.primary_concept, s.ability) for s in bank.question_specs
    }:
        raise typer.BadParameter("draft cells must match the blueprint exactly")
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    for spec in bank.question_specs:
        row = dict(by_cell[(spec.primary_concept, spec.ability)])
        row.pop("concept_id")
        row.pop("ability")
        q = assemble_question(
            spec,
            ConceptProviderQuestion.model_validate(row),
            model="codex-local-draft",
            language="en-US",
            artifact_hash=file_hash(blueprint),
            usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )
        path = output_dir / f"{spec.question_id}.json"
        if path.exists() and path.read_text() != q.model_dump_json(indent=2) + "\n":
            raise typer.BadParameter("existing candidate differs; choose a new output directory")
        path.write_text(q.model_dump_json(indent=2) + "\n")
        questions.append(q)
    write_review_packet(bank, questions, output_dir / "review.md")
    typer.echo(f"{len(questions)} local drafts packaged; review pending")


@app.command("generate")
def generate(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    dry_run: bool = False,
):
    """Generate one candidate per concept/ability, resuming validated saved output."""
    bank = ConceptBlueprint.model_validate_json(blueprint.read_text())
    settings = load_settings(require_api_key=not dry_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hash = file_hash(blueprint)
    questions = []
    for spec in bank.question_specs:
        if dry_run:
            (output_dir / f"{spec.question_id}.prompt.json").write_text(render_concept_prompt(spec))
            continue
        path = output_dir / f"{spec.question_id}.json"
        if path.exists():
            question = ConceptGeneratedQuestion.model_validate_json(path.read_text())
            if (
                question.input_artifact_hash != artifact_hash
                or question.question_spec_id != spec.question_id
                or question.question_spec_hash != content_hash(spec.model_dump())
                or question.prompt_version != PROMPT_VERSION
            ):
                raise typer.BadParameter(
                    "saved question belongs to another blueprint or prompt; "
                    "choose a new output directory"
                )
        else:
            question = generate_concept_question(
                spec, settings, artifact_hash, raw_path=output_dir / "raw" / path.name
            )
            path.write_text(question.model_dump_json(indent=2) + "\n")
        questions.append(question)
        typer.echo(f"{spec.primary_concept} / {spec.ability}: candidate saved")
    if questions:
        write_review_packet(bank, questions, output_dir / "review.md")


@app.command("review")
def review(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
):
    """Revalidate all saved candidates and rebuild a reviewable packet."""
    try:
        bank, questions = load_candidates(blueprint, output_dir)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    write_review_packet(bank, questions, output_dir / "review.md")
    typer.echo(f"{len(questions)} candidates; review records are checked separately")


@app.command("prepare-review")
def prepare_review_command(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidates_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    review_dir: Annotated[Path, typer.Option()],
):
    """Create a new packet and blank human worksheet, preserving existing review work."""
    try:
        report = prepare_review(blueprint, candidates_dir, review_dir)
    except FileExistsError as exc:
        raise typer.BadParameter("review directory exists; choose a new directory") from exc
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"{report['candidate_count']} candidates; blank worksheet saved to {review_dir}")


@app.command("review-status")
def review_status_command(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidates_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    reviews: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
):
    """Read a version-bound worksheet and report coverage without changing the bank."""
    import json

    try:
        _, questions = load_candidates(blueprint, candidates_dir)
        report = review_status(questions, load_worksheet(reviews))
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("ai-review-status")
def ai_review_status_command(
    blueprint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidates_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    reviews: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
):
    """Check explicitly authored AI content reviews against the exact candidates."""
    import json

    try:
        report = ai_review_status(blueprint, candidates_dir, reviews)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
