"""Deterministic support and semantic checks after structured parsing."""

import re
import unicodedata

from question_generation.errors import (
    GeneratedQuestionValidationError,
    UnsupportedQuestionSpecError,
)
from question_generation.schemas import ProviderQuestion, QuestionSpec

_PLACEHOLDER = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bFIXME\b|\[insert\b|\{\{.+?\}\}|<placeholder>)",
    re.IGNORECASE,
)
_NUMBERED_CHOICE = re.compile(r"\b(?:choice|option)\s*([0-9]+)\b", re.IGNORECASE)
_LETTERED_CHOICE = re.compile(r"\b(?:choice|option)\s*([A-Z])\b", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_supported_spec(spec: QuestionSpec) -> None:
    """Enforce the intentionally narrow, source-text-free v1 boundary."""

    if spec.question_type == "comprehension":
        raise UnsupportedQuestionSpecError(
            "comprehension and prose-grounded generation are not supported in v1"
        )
    expected_operation = {
        "vocabulary": "recognize",
        "background_knowledge": "recall",
    }.get(spec.question_type)
    if expected_operation is None:
        raise UnsupportedQuestionSpecError(f"unsupported question_type: {spec.question_type}")
    if spec.cognitive_operation != expected_operation:
        raise UnsupportedQuestionSpecError(
            f"v1 supports only vocabulary/recognize and background_knowledge/recall; "
            f"received {spec.question_type}/{spec.cognitive_operation}"
        )
    if spec.target_difficulty != 1:
        raise UnsupportedQuestionSpecError("v1 supports only Level 1 targets")
    if (
        spec.related_concepts
        or spec.relationship_reasoning_required
        or spec.multi_step_required
        or spec.cognitive_operation in {"integrate", "infer", "compare", "relate", "apply"}
    ):
        raise UnsupportedQuestionSpecError(
            "relation reasoning, application, integration, inference, and multi-step targets "
            "are not supported in v1"
        )
    if spec.source_document_ids:
        raise UnsupportedQuestionSpecError(
            "source-text or prose-grounded targets are not supported"
        )


def _validate_choice_references(explanation: str, correct_index: int) -> None:
    expected_number = correct_index + 1
    for match in _NUMBERED_CHOICE.finditer(explanation):
        number = int(match.group(1))
        if number != expected_number:
            raise GeneratedQuestionValidationError(
                "explanation refers to a choice number that is not the correct choice"
            )
    expected_letter = "ABCD"[correct_index]
    for match in _LETTERED_CHOICE.finditer(explanation):
        if match.group(1).upper() != expected_letter:
            raise GeneratedQuestionValidationError(
                "explanation refers to a choice letter that is not the correct choice"
            )


def validate_provider_question(output: ProviderQuestion, spec: QuestionSpec) -> None:
    """Validate structured syntax plus meaning that JSON Schema cannot guarantee."""

    expected = {
        "question_spec_id": spec.question_id,
        "topic_id": spec.topic_id,
        "question_type": spec.question_type,
        "cognitive_operation": spec.cognitive_operation,
        "primary_concept": spec.primary_concept,
        "related_concepts": spec.related_concepts,
        "target_difficulty": spec.target_difficulty,
    }
    actual = {
        "question_spec_id": output.question_spec_id,
        "topic_id": output.topic_id,
        "question_type": output.question_type,
        "cognitive_operation": output.cognitive_operation,
        "primary_concept": output.primary_concept,
        "related_concepts": output.related_concepts,
        "target_difficulty": output.target_difficulty,
    }
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if mismatches:
        raise GeneratedQuestionValidationError(
            "provider output changed authoritative QuestionSpec fields: " + ", ".join(mismatches)
        )

    normalized_choices = [normalize_text(choice) for choice in output.choices]
    if len(normalized_choices) != 4:
        raise GeneratedQuestionValidationError("question must contain exactly four choices")
    if any(not choice for choice in normalized_choices):
        raise GeneratedQuestionValidationError("choices must be nonblank")
    if len(set(normalized_choices)) != 4:
        raise GeneratedQuestionValidationError("choices must be unique after normalization")

    fields = [output.stem, output.explanation, *output.choices]
    if any(_PLACEHOLDER.search(value) for value in fields):
        raise GeneratedQuestionValidationError("question contains a TODO or placeholder")

    correct = normalized_choices[output.correct_choice_index]
    if len(correct) >= 4 and correct in normalize_text(output.stem):
        raise GeneratedQuestionValidationError("the full correct answer is leaked verbatim in stem")
    _validate_choice_references(output.explanation, output.correct_choice_index)
