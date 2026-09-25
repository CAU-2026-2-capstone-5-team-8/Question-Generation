"""Human-review artifact helpers."""

import json
from pathlib import Path

from question_generation.schemas import HumanQuestionReview


def load_review_jsonl(path: Path) -> list[HumanQuestionReview]:
    reviews: list[HumanQuestionReview] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            reviews.append(HumanQuestionReview.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid human review at line {line_number}: {exc}") from exc
    return reviews


def write_review_jsonl(reviews: list[HumanQuestionReview], path: Path) -> None:
    content = "\n".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for item in reviews
    )
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
