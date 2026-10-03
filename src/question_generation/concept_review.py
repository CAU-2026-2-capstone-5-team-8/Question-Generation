"""Version-bound worksheets for human review; never create approval judgments."""

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from question_generation.concept_contract import (
    ConceptBlueprint,
    ConceptGeneratedQuestion,
    content_hash,
)
from question_generation.concept_generation import file_hash, write_review_packet
from question_generation.schemas import AiQuestionReview, QuestionReviewJudgment, ReviewStatus


class ReviewWorksheetRow(BaseModel):
    """Same fields as HumanQuestionReview, with explicit nulls for unfinished judgments."""

    model_config = ConfigDict(extra="forbid", strict=True)

    generated_question_id: str = Field(pattern=r"^gq_[0-9a-f]{32}$")
    status: ReviewStatus | None
    correct: bool | None
    concept_alignment: int | None = Field(ge=1, le=5)
    difficulty_appropriate: bool | None
    distractor_quality: int | None = Field(ge=1, le=5)
    explanation_quality: int | None = Field(ge=1, le=5)
    notes: str = ""

    def completed_review(self) -> QuestionReviewJudgment | None:
        data = self.model_dump()
        if any(value is None for value in data.values()):
            return None
        return QuestionReviewJudgment.model_validate(data)


def load_candidates(
    blueprint_path: Path, candidates_dir: Path
) -> tuple[ConceptBlueprint, list[ConceptGeneratedQuestion]]:
    bank = ConceptBlueprint.model_validate_json(blueprint_path.read_text(encoding="utf-8"))
    expected = {f"{spec.question_id}.json" for spec in bank.question_specs}
    if {path.name for path in candidates_dir.glob("q_*.json")} != expected:
        raise ValueError(
            "candidate files must match blueprint exactly (missing or extra candidate)"
        )
    artifact_hash = file_hash(blueprint_path)
    questions = []
    for spec in bank.question_specs:
        question = ConceptGeneratedQuestion.model_validate_json(
            (candidates_dir / f"{spec.question_id}.json").read_text(encoding="utf-8")
        )
        expected_fields = {
            "question_spec_id": spec.question_id,
            "question_spec_hash": content_hash(spec.model_dump()),
            "input_artifact_hash": artifact_hash,
            **{
                field: spec.model_dump()[field]
                for field in (
                    "topic_id",
                    "primary_concept",
                    "ability",
                    "cognitive_operation",
                    "measurement_context",
                    "target_difficulty",
                    "assessment_objective",
                    "evidence_references",
                )
            },
        }
        actual = question.model_dump()
        for field, value in expected_fields.items():
            if actual[field] != value:
                raise ValueError(f"{spec.question_id}: candidate disagrees with blueprint: {field}")
        questions.append(question)
    return bank, questions


def load_worksheet(path: Path) -> list[ReviewWorksheetRow]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(ReviewWorksheetRow.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid review worksheet at line {number}: {exc}") from exc
    return rows


def ai_review_status(blueprint: Path, candidates_dir: Path, reviews: Path) -> dict:
    """Validate authored AI judgments; never infer approval from structural checks."""
    _, questions = load_candidates(blueprint, candidates_dir)
    parsed = []
    for number, line in enumerate(reviews.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed.append(AiQuestionReview.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid AI review at line {number}: {exc}") from exc
    report = review_status(
        questions,
        [ReviewWorksheetRow.model_validate(item.review.model_dump()) for item in parsed],
    )
    report["reviewer_type"] = "ai"
    report["reviewer_names"] = sorted({item.reviewer_name for item in parsed})
    report["validation_scope"] = "content-only"
    return report


def review_status(
    questions: list[ConceptGeneratedQuestion], rows: list[ReviewWorksheetRow]
) -> dict:
    ids = [row.generated_question_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate review ID")
    expected = {q.generated_question_id for q in questions}
    if set(ids) != expected:
        raise ValueError("review IDs must match candidates exactly (missing, unknown or stale ID)")
    by_id = {row.generated_question_id: row for row in rows}
    cells = []
    for question in questions:
        row = by_id[question.generated_question_id]
        review = row.completed_review()
        eligible = review is not None and review.status == "approve" and review.correct
        cells.append(
            {
                "concept": question.primary_concept,
                "ability": question.ability,
                "generated_question_id": question.generated_question_id,
                "candidate_count": 1,
                "review_status": review.status if review else "pending",
                "approval_criteria_met": eligible,
                "incomplete_fields": [
                    key for key, value in row.model_dump().items() if value is None
                ],
            }
        )
    approved = sum(cell["approval_criteria_met"] for cell in cells)
    return {
        "report_version": "concept-review-status-v1",
        "scope": "supplied candidates and review file only; not the active assessment bank",
        "review_authorship": "not authenticated by this tool",
        "candidate_count": len(questions),
        "review_counts": dict(Counter(cell["review_status"] for cell in cells)),
        "approval_criteria_met_count": approved,
        "cells_without_approval": len(cells) - approved,
        "cells": cells,
    }


def prepare_review(blueprint: Path, candidates_dir: Path, review_dir: Path) -> dict:
    bank, questions = load_candidates(blueprint, candidates_dir)
    rows = [
        ReviewWorksheetRow(
            generated_question_id=q.generated_question_id,
            status=None,
            correct=None,
            concept_alignment=None,
            difficulty_appropriate=None,
            distractor_quality=None,
            explanation_quality=None,
        )
        for q in questions
    ]
    report = review_status(questions, rows)
    # An existing directory may contain a person's work, even when it looks incomplete.
    review_dir.mkdir(parents=True, exist_ok=False)
    (review_dir / "reviews.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (review_dir / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    packet = review_dir / "review.md"
    write_review_packet(bank, questions, packet)
    instructions = (
        "# 사람 검토 작성 안내\n\n"
        "아래 문항을 읽고 같은 생성 ID의 `reviews.jsonl` 행을 작성하세요. "
        "판정은 모두 비워 두었습니다. AI 사전 점검은 승인으로 옮기지 않습니다.\n\n"
        "- `status`: approve / needs_revision / reject.\n"
        "- `correct`: 정답과 해설이 맞으며 정답이 하나인지 true / false.\n"
        "- `concept_alignment`: 제시한 개념과 수행 목표를 측정하는 정도, 1~5.\n"
        "- `difficulty_appropriate`: 설계 난도에 적절한지 true / false.\n"
        "- `distractor_quality`: 오개념 반영과 오답의 타당성, 1~5.\n"
        "- `explanation_quality`: 해설의 정확성·명료성, 1~5.\n"
        "- 점수 기준: 1 매우 부족 / 2 부족 / 3 보통 / 4 좋음 / 5 매우 좋음.\n"
        "- `notes`: 수정할 보기, 영어 읽기 부담, 길이·표현으로 드러나는 정답 단서 등.\n\n"
        "미검토 항목은 null로 남기세요. 일부만 작성한 행은 pending입니다. "
        "완료 행은 기존 HumanQuestionReview 계약과 같습니다. 모든 필드가 채워지고 "
        "approve 및 correct=true인 문항만 승인 조건을 충족합니다. "
        "도구는 작성자의 신원이나 판단의 진실성을 확인하지 않습니다.\n\n"
        "`coverage.json`은 생성 시점의 후보 검토 현황입니다. 편집 후에는 "
        "`concept review-status`로 다시 확인하세요. 실제 bank 등록은 별도 단계입니다. "
        "각 개념·수행 목표에 후보가 하나씩 있어 대체 문항 여유는 없습니다.\n\n"
        f"설계서 해시: `{file_hash(blueprint)}`\n\n---\n\n"
    )
    packet.write_text(instructions + packet.read_text(encoding="utf-8"), encoding="utf-8")
    return report
