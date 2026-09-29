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
GROUNDED_COMPREHENSION_REVIEW_FILE = (
    ROOT / "reviews" / "linear-algebra-grounded-comprehension-v1.jsonl"
)
DISPLAY_GROUNDED_COMPREHENSION_REVIEW_FILE = (
    ROOT / "reviews" / "linear-algebra-display-grounded-comprehension-v1.jsonl"
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


def test_grounded_comprehension_human_review_is_recorded_exactly() -> None:
    reviews = load_review_jsonl(GROUNDED_COMPREHENSION_REVIEW_FILE)

    assert len(reviews) == 1
    review = reviews[0]
    assert review.generated_question_id == "gq_4783115ebe51684dd059a1724b42df8b"
    assert review.correct is True
    assert review.concept_alignment == 5
    assert review.difficulty_appropriate is False
    assert review.distractor_quality == 4
    assert review.explanation_quality == 5
    assert review.status == "needs_revision"
    assert review.notes == (
        "정답과 해설은 passage에 정확히 근거하며 matrix concept alignment도 적절하다. 그러나 "
        "정답 선택지가 passage에 제시된 행렬 예시를 사실상 그대로 재현하므로, 사용자가 matrix의 "
        "행/열 구조를 새로운 상황에 적용하지 않고 단순 시각적 대조로 답할 수 있다. 따라서 "
        "comprehension/apply/Level 2의 진단 목적에는 부족하다. 동일 규칙을 새로운 배열이나 entry "
        "위치에 적용해야 풀 수 있도록 수정이 필요하다. 또한 grounding passage에 PDF text "
        "extraction으로 인한 공백 및 문자 결합 오류가 다수 존재하여 사용자 표시 품질 개선을 후속 "
        "검토해야 한다."
    )


def test_display_grounded_comprehension_human_review_is_recorded_exactly() -> None:
    reviews = load_review_jsonl(DISPLAY_GROUNDED_COMPREHENSION_REVIEW_FILE)

    assert len(reviews) == 2
    assert len({item.generated_question_id for item in reviews}) == 2
    first_pass, revised = reviews
    assert first_pass.generated_question_id == "gq_c0e6f6c774e467bcb8ab2c10a4cd9379"
    assert first_pass.correct is True
    assert first_pass.concept_alignment == 5
    assert first_pass.difficulty_appropriate is True
    assert first_pass.distractor_quality == 3
    assert first_pass.explanation_quality == 5
    assert first_pass.status == "needs_revision"
    assert first_pass.notes == (
        "display passage는 raw source의 의미와 행렬 구조를 보존하면서 PDF 추출 오류를 충분히 "
        "개선했다. 문항은 passage의 2×3 예시를 그대로 대조하는 대신 4행 5열이라는 새로운 상황에 "
        "행 우선 표기 규칙을 적용해야 하므로 comprehension/apply/Level 2 목적에 적절하다. 정답과 "
        "해설도 passage에 근거해 정확하다. 다만 선택지 A의 '오브젝트-포(five-by-four)' 표현은 "
        "자연스럽지 않고 오답 단서를 과도하게 제공하여 distractor 품질을 저하시킨다. 선택지 간 "
        "오개념을 더 자연스럽고 서로 다른 형태로 구성한 뒤 재검토하는 것이 적절하다."
    )
    assert revised.generated_question_id == "gq_66f3360464d1336ec1612715ebbdd3ed"
    assert revised.correct is True
    assert revised.concept_alignment == 5
    assert revised.difficulty_appropriate is True
    assert revised.distractor_quality == 4
    assert revised.explanation_quality == 4
    assert revised.status == "approve"
    assert revised.notes == (
        "passage의 2×3 예시를 그대로 대조하는 기존 문제와 달리, 4행 3열이라는 새로운 상황에 행 "
        "우선 m×n 표기 규칙을 적용해야 하므로 comprehension/apply/Level 2 목적에 적절하다. 정답은 "
        "passage만으로 결정할 수 있고 외부 지식이 필요하지 않으며, 네 선택지는 행/열 순서와 표기 "
        "규칙의 혼동을 활용해 이전보다 자연스럽게 구성됐다. 다만 일부 distractor는 표기와 설명을 "
        "조합한 단순한 오개념 형태이고, 해설의 'stated된다' 표현은 다소 부자연스럽다. 이러한 "
        "표현상 한계는 정답성이나 concept alignment를 훼손하지 않으므로 현재 문항은 승인한다."
    )


def test_human_review_rejects_out_of_range_score() -> None:
    review = load_review_jsonl(REVIEW_FILE)[0]

    with pytest.raises(ValidationError):
        HumanQuestionReview.model_validate(review.model_dump() | {"concept_alignment": 6})
