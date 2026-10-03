"""Synthetic judgments test the review boundary, not the real question bank."""

import json

import pytest
from typer.testing import CliRunner

from question_generation.concept_commands import app
from question_generation.concept_contract import ConceptBlueprint, content_hash
from question_generation.concept_generation import assemble_question, file_hash
from question_generation.concept_review import (
    load_candidates,
    load_worksheet,
    prepare_review,
    review_status,
)
from question_generation.review import load_review_jsonl
from tests.test_concept_generation import output, spec


@pytest.fixture
def bundle(tmp_path):
    target = spec()
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text(
        ConceptBlueprint(
            blueprint_version="concept-assessment-blueprint-v2",
            topic_id="linear-algebra",
            question_specs=[target],
        ).model_dump_json()
    )
    question = assemble_question(
        target,
        output(),
        model="synthetic-test",
        language="en-US",
        artifact_hash=file_hash(blueprint),
        usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
    )
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / f"{target.question_id}.json").write_text(question.model_dump_json())
    return blueprint, candidates, tmp_path / "review"


def test_blank_reviews_are_not_importable_and_cannot_overwrite_human_work(bundle):
    blueprint, candidates, directory = bundle
    report = prepare_review(*bundle)
    assert report["candidate_count"] == report["cells_without_approval"] == 1
    assert report["approval_criteria_met_count"] == 0
    assert report["review_counts"] == {"pending": 1}
    path = directory / "reviews.jsonl"
    with pytest.raises(ValueError, match="invalid human review"):
        load_review_jsonl(path)
    path.write_text("human work in progress")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_review(blueprint, candidates, directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize(
    ("status", "correct", "expected"),
    [
        ("approve", True, 1),
        ("approve", False, 0),
        ("needs_revision", True, 0),
        ("reject", False, 0),
    ],
)
def test_only_complete_approved_correct_reviews_meet_existing_import_criteria(
    bundle,
    status,
    correct,
    expected,
):
    blueprint, candidates, directory = bundle
    prepare_review(*bundle)
    _, questions = load_candidates(blueprint, candidates)
    path = directory / "reviews.jsonl"
    row = json.loads(path.read_text())
    row.update(
        status=status,
        correct=correct,
        concept_alignment=3,
        difficulty_appropriate=False,
        distractor_quality=2,
        explanation_quality=4,
        notes="Synthetic test judgment only",
    )
    path.write_text(json.dumps(row) + "\n")
    # False is a completed judgment, not a missing value. The existing contract is unchanged.
    assert load_review_jsonl(path)[0].difficulty_appropriate is False
    report = review_status(questions, load_worksheet(path))
    assert report["approval_criteria_met_count"] == expected
    assert report["review_counts"] == {status: 1}
    assert report["cells"][0]["incomplete_fields"] == []
    row["explanation_quality"] = None
    path.write_text(json.dumps(row))
    report = review_status(questions, load_worksheet(path))
    assert report["approval_criteria_met_count"] == 0
    assert report["review_counts"] == {"pending": 1}
    assert report["cells"][0]["incomplete_fields"] == ["explanation_quality"]


@pytest.mark.parametrize("problem", ["duplicate", "missing", "stale"])
def test_rejects_review_sets_that_do_not_match_current_candidates(bundle, problem):
    blueprint, candidates, directory = bundle
    prepare_review(*bundle)
    _, questions = load_candidates(blueprint, candidates)
    rows = load_worksheet(directory / "reviews.jsonl")
    if problem == "duplicate":
        rows *= 2
    elif problem == "missing":
        rows = []
    else:
        rows[0].generated_question_id = "gq_" + "f" * 32
    with pytest.raises(ValueError, match="duplicate|must match"):
        review_status(questions, rows)


@pytest.mark.parametrize(
    "field",
    [
        "assessment_objective",
        "evidence_references",
        "target_difficulty",
        "input_artifact_hash",
        "question_spec_hash",
        "question_spec_id",
    ],
)
def test_rehashing_a_changed_candidate_cannot_bypass_blueprint_binding(bundle, field):
    blueprint, candidates, directory = bundle
    path = next(candidates.glob("q_*.json"))
    data = json.loads(path.read_text())
    data[field] = {
        "assessment_objective": "Different objective",
        "evidence_references": [{"book_id": "other", "toc_entry_id": "other"}],
        "target_difficulty": 3,
        "input_artifact_hash": "sha256:" + "f" * 64,
        "question_spec_hash": "sha256:" + "f" * 64,
        "question_spec_id": "q_" + "f" * 20,
    }[field]
    identity = {k: v for k, v in data.items() if k not in {"generated_question_id", "usage"}}
    data["generated_question_id"] = "gq_" + content_hash(identity).split(":")[1][:32]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        prepare_review(blueprint, candidates, directory)
    assert not directory.exists()


@pytest.mark.parametrize("extra", [True, False])
def test_missing_or_extra_candidates_fail_before_creating_a_worksheet(bundle, extra):
    blueprint, candidates, directory = bundle
    path = next(candidates.glob("q_*.json"))
    if extra:
        (candidates / "q_unexpected.json").write_bytes(path.read_bytes())
    else:
        path.unlink()
    with pytest.raises(ValueError, match="candidate files must match"):
        prepare_review(blueprint, candidates, directory)
    assert not directory.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"correct": "true"},
        {"concept_alignment": 6},
        {"status": "pending"},
        {"reviewer": "invented"},
    ],
)
def test_worksheet_rejects_coercion_bad_scores_and_unknown_fields(bundle, change):
    _, _, directory = bundle
    prepare_review(*bundle)
    path = directory / "reviews.jsonl"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, **change}))
    with pytest.raises(ValueError, match="line 1"):
        load_worksheet(path)


def test_cli_reports_pending_without_mutating_inputs_or_accepting_stale_reviews(bundle):
    blueprint, candidates, directory = bundle
    runner = CliRunner()
    args = ["--blueprint", str(blueprint), "--candidates-dir", str(candidates)]
    result = runner.invoke(app, ["prepare-review", *args, "--review-dir", str(directory)])
    assert result.exit_code == 0, result.output
    reviews = directory / "reviews.jsonl"
    before = reviews.read_bytes()
    result = runner.invoke(app, ["review-status", *args, "--reviews", str(reviews)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["approval_criteria_met_count"] == 0
    assert reviews.read_bytes() == before
    reviews.write_text("")
    result = runner.invoke(app, ["review-status", *args, "--reviews", str(reviews)])
    assert result.exit_code != 0
