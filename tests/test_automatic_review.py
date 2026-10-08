import json
from types import SimpleNamespace

import pytest

from question_generation.automatic_review import review_batch
from question_generation.concept_batch import generate_concept_batch
from tests.test_concept_batch import bundle as bundle


class Reviewer:
    def __init__(self, fail=False, wrong=False, reject=False):
        self.models = self
        self.calls = []
        self.fail = fail
        self.wrong = wrong
        self.reject = reject

    def generate_content(self, **kw):
        value = json.loads(kw["contents"])
        self.calls.append(value)
        if "edges" in value:
            response = {
                "appropriate": True,
                "reason": "The supplied concepts fit this academic field.",
                "edges": [],
            }
        else:
            if self.fail and len(self.calls) > 2:
                raise RuntimeError("network lost")
            response = {
                "generated_question_id": ("gq_" + "f" * 32)
                if self.wrong
                else value["generated_question_id"],
                "status": "needs_revision" if self.reject else "approve",
                "correct": not self.reject,
                "concept_alignment": 4,
                "difficulty_appropriate": True,
                "distractor_quality": 4,
                "explanation_quality": 4,
                "notes": "Independent solution verifies the unique answer and each distractor.",
            }
        return SimpleNamespace(text=json.dumps(response))


def setup(bundle):
    blueprint, directory, settings, _ = bundle
    generate_concept_batch(blueprint, directory, settings, max_new=3, client=object())
    outline = blueprint.parent / "outline.json"
    outline.write_text(json.dumps({"topic_id": "linear-algebra", "concepts": [], "edges": []}))
    return blueprint, directory, outline, directory / "review", settings


def test_bounded_review_resumes_exact_versions_without_inventing_human_approval(bundle):
    args = setup(bundle)
    client = Reviewer()
    first = review_batch(*args, client=client)
    assert first["status"] == "REVIEW_PENDING" and first["reviewedCount"] == 2
    final = review_batch(*args, client=client)
    assert (
        final["status"] == "APPROVED"
        and final["approvedCount"] == 3
        and not final["diagnosisReady"]
    )
    assert len(client.calls) == 4
    assert review_batch(*args, client=client) == final and len(client.calls) == 4
    reviews = (args[3] / "reviews.jsonl").read_text()
    assert '"reviewer_type":"ai"' in reviews and "humanReview" not in reviews
    assert all("evidence_references" not in call for call in client.calls)


def test_network_failure_retains_completed_reviews(bundle):
    args = setup(bundle)
    client = Reviewer(fail=True)
    with pytest.raises(RuntimeError):
        review_batch(*args, client=client)
    assert len(list(args[3].glob("gq_*.json"))) == 1
    report = review_batch(*args, client=Reviewer())
    assert report["status"] == "APPROVED" and report["reviewedCount"] == 3


def test_rejected_review_blocks_activation(bundle):
    args = setup(bundle)
    client = Reviewer(reject=True)
    review_batch(*args, client=client)
    final = review_batch(*args, client=client)
    assert final["status"] == "REVIEW_BLOCKED" and final["approvedCount"] == 0


def test_stale_id_never_becomes_review_approval(bundle):
    with pytest.raises(ValueError, match="ID differs"):
        review_batch(*setup(bundle), client=Reviewer(wrong=True))


def test_changed_candidate_or_model_rejects_cached_review(bundle):
    args = setup(bundle)
    review_batch(*args, client=Reviewer())
    from dataclasses import replace

    with pytest.raises(ValueError, match="inputs changed"):
        review_batch(*args[:-1], replace(args[-1], model="other-model"), client=Reviewer())
