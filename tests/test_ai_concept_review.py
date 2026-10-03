import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from question_generation.concept_contract import ConceptProviderQuestion
from question_generation.concept_generation import quality_flags
from question_generation.concept_review import ai_review_status, load_candidates
from question_generation.schemas import AiQuestionReview, HumanQuestionReview
from tests.test_concept_review import bundle  # noqa: F401


def ai_record(question_id):
    return {
        "review_version": "ai-question-review-v1",
        "reviewer_type": "ai",
        "reviewer_name": "Synthetic test reviewer",
        "validation_scope": "content-only",
        "review": {
            "generated_question_id": question_id,
            "status": "approve",
            "correct": True,
            "concept_alignment": 4,
            "difficulty_appropriate": True,
            "distractor_quality": 3,
            "explanation_quality": 4,
            "notes": "Synthetic contract judgment; not an actual candidate approval.",
        },
    }


def test_ai_review_is_version_bound_and_never_parsed_as_a_human_review(bundle):  # noqa: F811
    blueprint, candidates, directory = bundle
    _, questions = load_candidates(blueprint, candidates)
    data = ai_record(questions[0].generated_question_id)
    path = directory.with_suffix(".jsonl")
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        HumanQuestionReview.model_validate(data)
    report = ai_review_status(blueprint, candidates, path)
    assert report["reviewer_type"] == "ai"
    assert report["validation_scope"] == "content-only"
    assert report["approval_criteria_met_count"] == 1
    data["review"]["generated_question_id"] = "gq_" + "f" * 32
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stale ID"):
        ai_review_status(blueprint, candidates, path)


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer_type": "human"},
        {"reviewer_name": " "},
        {"validation_scope": "empirically-validated"},
        {"review_version": "ai-question-review-v2"},
    ],
)
def test_ai_contract_rejects_ambiguous_identity_and_unsupported_claims(change):
    with pytest.raises(ValidationError):
        AiQuestionReview.model_validate({**ai_record("gq_" + "a" * 32), **change})


def test_all_revised_drafts_pass_content_contract_and_have_unique_cells():
    rows = json.loads(
        (Path(__file__).parents[1] / "examples/concept-local-drafts-v4.json").read_text()
    )
    assert len({(row["concept_id"], row["ability"]) for row in rows}) == len(rows) == 18
    for row in rows:
        output = ConceptProviderQuestion.model_validate(
            {k: v for k, v in row.items() if k not in {"concept_id", "ability"}}
        )
        assert quality_flags(output) == []


def test_revised_numeric_distractors_represent_actual_errors():
    rows = json.loads(
        (Path(__file__).parents[1] / "examples/concept-local-drafts-v4.json").read_text()
    )
    cells = {(row["concept_id"], row["ability"]): row for row in rows}
    matrix = cells[("matrix", "application")]
    a, b = [[1, 2], [0, 1]], [[2, 1], [3, 1]]
    correct = sum(a[0][k] * b[k][0] for k in range(2))
    row_row = sum(x * y for x, y in zip(a[0], b[0], strict=True))
    elementwise = a[0][0] * b[0][0]
    assert matrix["choices"][matrix["correct_choice_index"]] == rf"\({correct}\)"
    assert matrix["choices"].count(rf"\({correct}\)") == 1
    assert row_row != elementwise != correct and row_row != correct
    assert rf"\({row_row}\)" in matrix["choices"]
    assert rf"\({elementwise}\)" in matrix["choices"]
    vector = cells[("vector", "application")]
    u, v = (1, 2), (-1, 1)
    correct_vector = tuple(2 * x - 3 * y for x, y in zip(u, v, strict=True))
    add_scalars = tuple((x + 2) - (y + 3) for x, y in zip(u, v, strict=True))
    assert (
        vector["choices"][vector["correct_choice_index"]]
        == rf"\({str(correct_vector).replace(' ', '')}\)"
    )
    assert rf"\({str(add_scalars).replace(' ', '')}\)" in vector["choices"]
    # First two coordinates fix the coefficients; the revised third coordinate prevents membership.
    w = (2, 3, 4)
    assert w[0] + w[1] != w[2]
    assert (1, 1, 2)[0] + (1, 1, 2)[1] == (1, 1, 2)[2]
