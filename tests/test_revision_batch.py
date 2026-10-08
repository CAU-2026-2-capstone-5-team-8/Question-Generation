import json
from types import SimpleNamespace

import pytest

from question_generation import revision_batch
from question_generation.concept_batch import generate_concept_batch, save_json
from question_generation.concept_generation import assemble_question, file_hash
from tests.test_concept_batch import bundle as bundle
from tests.test_concept_generation import output


def reviewed(bundle, rejected=1, source_metadata=True):
    blueprint, directory, settings, _ = bundle
    report = generate_concept_batch(blueprint, directory, settings, max_new=3, client=object())
    if source_metadata:
        report.update(sourceSnapshotId="synthetic-source", contentReportHash="sha256:" + "e" * 64)
    generation = directory / "generation-report.json"
    save_json(generation, report)
    review = directory / "review"
    review.mkdir()
    rows = []
    for i, path in enumerate(sorted(directory.glob("q_*.json"))):
        q = json.loads(path.read_text())
        rows.append(
            {
                "review_version": "ai-question-review-v1",
                "reviewer_type": "ai",
                "reviewer_name": settings.model,
                "validation_scope": "content-only",
                "review": {
                    "generated_question_id": q["generated_question_id"],
                    "status": "reject" if i < rejected else "approve",
                    "correct": i >= rejected,
                    "concept_alignment": 4,
                    "difficulty_appropriate": True,
                    "distractor_quality": 4,
                    "explanation_quality": 4,
                    "notes": "The supplied answer index must agree with the explanation.",
                },
            }
        )
    (review / "reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    review_report = review / "review-report.json"
    save_json(review_report, {"status": "REVIEW_BLOCKED", "graphApproved": True})
    return blueprint, directory, generation, review_report, settings


@pytest.mark.parametrize("source_metadata", [True, False])
def test_repair_changes_only_rejected_content_requires_new_review_and_resumes(
    bundle, monkeypatch, source_metadata
):
    args = reviewed(bundle, source_metadata=source_metadata)
    original = {p.name: p.read_bytes() for p in args[1].glob("q_*.json")}
    calls = []

    def author(spec, settings, artifact_hash, **kwargs):
        calls.append(kwargs["revision_feedback"])
        provider = output().model_copy(
            update={"stem": r"Find entry \((1,1)\) of the identity matrix."}
        )
        return assemble_question(
            spec,
            provider,
            model=settings.model,
            language="en-US",
            artifact_hash=artifact_hash,
            usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )

    monkeypatch.setattr(revision_batch, "generate_concept_question", author)
    path = revision_batch.revise_batch(*args, client=SimpleNamespace())
    report = json.loads(path.read_text())
    assert len(calls) == 1 and report["revisionRound"] == 1
    for key in ("sourceSnapshotId", "contentReportHash"):
        assert (key in report) == source_metadata
        if source_metadata:
            assert report[key] == json.loads(args[2].read_text())[key]
    assert report["contentReview"] == "pending" and report["diagnosisReady"] is False
    assert report["previousGenerationReportHash"] == file_hash(args[2])
    assert {p.name: p.read_bytes() for p in args[1].glob("q_*.json")} == original
    changed = [name for name, data in original.items() if (path.parent / name).read_bytes() != data]
    assert len(changed) == 1
    assert revision_batch.revise_batch(*args, client=SimpleNamespace()) == path
    assert len(calls) == 1
    with pytest.raises(ValueError, match="not eligible"):
        revision_batch.revise_batch(
            args[0], path.parent, path, args[3], args[4], client=SimpleNamespace()
        )


def test_many_defects_or_rejected_graph_do_not_start_automatic_repair(bundle):
    args = reviewed(bundle, rejected=3)
    with pytest.raises(ValueError, match="limit exceeded"):
        revision_batch.revise_batch(*args, client=SimpleNamespace())
    save_json(args[3], {"status": "REVIEW_BLOCKED", "graphApproved": False})
    with pytest.raises(ValueError, match="not eligible"):
        revision_batch.revise_batch(*args, client=SimpleNamespace())
