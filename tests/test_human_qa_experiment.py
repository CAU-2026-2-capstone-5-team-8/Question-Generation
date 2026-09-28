from collections import Counter
from pathlib import Path
from statistics import mean

import pytest
from pydantic import ValidationError

from question_generation.review import load_review_jsonl
from question_generation.schemas import HumanQuestionReview

ROOT = Path(__file__).parents[1]
REVIEW_FILE = ROOT / "reviews" / "os-reviewed-question-generation-v1.jsonl"
LINEAR_ALGEBRA_REVIEW_FILE = (
    ROOT / "reviews" / "linear-algebra-reviewed-question-generation-v1.jsonl"
)


def test_os_reviewed_question_generation_human_qa_is_complete() -> None:
    reviews = load_review_jsonl(REVIEW_FILE)

    assert len(reviews) == 5
    assert len({item.generated_question_id for item in reviews}) == 5
    assert Counter(item.status for item in reviews) == {
        "approve": 4,
        "needs_revision": 1,
    }
    assert all(item.correct for item in reviews)
    assert all(item.difficulty_appropriate for item in reviews)
    assert all(item.notes.strip() for item in reviews if item.status == "needs_revision")


def test_os_reviewed_question_generation_human_qa_summary() -> None:
    reviews = load_review_jsonl(REVIEW_FILE)

    assert mean(item.concept_alignment for item in reviews) == 5.0
    assert mean(item.distractor_quality for item in reviews) == 4.2
    assert mean(item.explanation_quality for item in reviews) == 4.2


def test_linear_algebra_reviewed_question_generation_human_qa_is_complete() -> None:
    reviews = load_review_jsonl(LINEAR_ALGEBRA_REVIEW_FILE)

    assert len(reviews) == 5
    assert {item.generated_question_id for item in reviews} == {
        "gq_ebf68bf0595a5fdb96e8757ca92382d9",
        "gq_770923a236b225553467b5bddcb89601",
        "gq_142923d9ad544ed9f827253ba57df545",
        "gq_2b08c81f68f1ca6973de873b6e9c30c0",
        "gq_12d0e100aa9a7715e308bea46394b77f",
    }
    assert Counter(item.status for item in reviews) == {"approve": 5}
    assert all(item.correct for item in reviews)
    assert all(item.difficulty_appropriate for item in reviews)
    assert all(item.notes.strip() for item in reviews)


def test_linear_algebra_reviewed_question_generation_human_qa_summary() -> None:
    reviews = load_review_jsonl(LINEAR_ALGEBRA_REVIEW_FILE)

    assert mean(item.concept_alignment for item in reviews) == 4.8
    assert mean(item.distractor_quality for item in reviews) == 4.6
    assert mean(item.explanation_quality for item in reviews) == 4.6


def test_human_review_rejects_out_of_range_score() -> None:
    review = load_review_jsonl(REVIEW_FILE)[0]

    with pytest.raises(ValidationError):
        HumanQuestionReview.model_validate(review.model_dump() | {"concept_alignment": 6})
