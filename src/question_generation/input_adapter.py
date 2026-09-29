"""Read versioned JSON artifacts without importing code from the ML repository."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from question_generation.config import SUPPORTED_QUESTION_SPEC_VERSION
from question_generation.errors import InputContractError
from question_generation.schemas import (
    AssessmentBlueprintEnvelope,
    GenerationGrounding,
    GenerationGroundingV2,
    QuestionSpec,
)


@dataclass(frozen=True)
class LoadedQuestionSpec:
    spec: QuestionSpec
    artifact_hash: str
    artifact_path: Path
    blueprint_canonical_file_hashes: dict[str, str] | None


@dataclass(frozen=True)
class LoadedGenerationGrounding:
    grounding: GenerationGrounding | GenerationGroundingV2
    artifact_hash: str
    artifact_path: Path


def _artifact_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise InputContractError(f"cannot read input artifact {path}: {exc}") from exc
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputContractError(f"input artifact is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise InputContractError("input artifact root must be a JSON object")
    return payload, content


def load_question_spec(path: Path, *, question_id: str | None = None) -> LoadedQuestionSpec:
    """Load either one QuestionSpec object or select one from an AssessmentBlueprint."""

    payload, content = _read_json(path)
    blueprint_canonical_file_hashes: dict[str, str] | None = None
    try:
        if "question_specs" in payload:
            if question_id is None:
                raise InputContractError("--question-id is required for a blueprint artifact")
            blueprint = AssessmentBlueprintEnvelope.model_validate(payload)
            matches = [item for item in blueprint.question_specs if item.question_id == question_id]
            if len(matches) != 1:
                raise InputContractError(
                    f"question_id {question_id!r} was not found exactly once in the blueprint"
                )
            spec = matches[0]
            blueprint_canonical_file_hashes = blueprint.canonical_file_hashes
        else:
            spec = QuestionSpec.model_validate(payload)
            if question_id is not None and spec.question_id != question_id:
                raise InputContractError("single QuestionSpec does not match --question-id")
    except ValidationError as exc:
        raise InputContractError(f"input contract validation failed: {exc}") from exc
    if spec.question_spec_version != SUPPORTED_QUESTION_SPEC_VERSION:
        raise InputContractError(
            "unsupported question_spec_version: "
            f"{spec.question_spec_version!r}; expected {SUPPORTED_QUESTION_SPEC_VERSION!r}"
        )
    return LoadedQuestionSpec(
        spec=spec,
        artifact_hash=_artifact_hash(content),
        artifact_path=path.resolve(),
        blueprint_canonical_file_hashes=blueprint_canonical_file_hashes,
    )


def load_generation_grounding(path: Path) -> LoadedGenerationGrounding:
    """Load one supported grounding artifact and hash its exact bytes."""

    payload, content = _read_json(path)
    try:
        grounding_version = payload.get("grounding_version")
        if grounding_version == "generation-grounding-v1":
            grounding = GenerationGrounding.model_validate(payload)
        elif grounding_version == "generation-grounding-v2":
            grounding = GenerationGroundingV2.model_validate(payload)
        else:
            raise InputContractError(f"unsupported grounding_version: {grounding_version!r}")
    except ValidationError as exc:
        raise InputContractError(f"grounding contract validation failed: {exc}") from exc
    return LoadedGenerationGrounding(
        grounding=grounding,
        artifact_hash=_artifact_hash(content),
        artifact_path=path.resolve(),
    )
