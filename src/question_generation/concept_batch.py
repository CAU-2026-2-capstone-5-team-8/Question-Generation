"""Bounded, resumable generation. Candidates and review approval remain separate."""

import json
from importlib.metadata import version
from pathlib import Path

from google import genai
from google.genai import types

from question_generation.concept_contract import (
    ConceptBlueprint,
    ConceptGeneratedQuestion,
    content_hash,
)
from question_generation.concept_generation import (
    PROMPT_VERSION,
    SYSTEM,
    file_hash,
    generate_concept_question,
    quality_flags,
)
from question_generation.concept_review import load_candidates, prepare_review
from question_generation.config import GenerationSettings

BATCH_VERSION = "concept-generation-batch-v1"


def generation_identity(blueprint: Path, settings: GenerationSettings) -> dict:
    package = Path(__file__).parent
    return {
        "batchVersion": BATCH_VERSION,
        "blueprintHash": file_hash(blueprint),
        "model": settings.model,
        "language": settings.output_language,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "maxOutputTokens": 4096,
        "promptVersion": PROMPT_VERSION,
        "systemHash": content_hash(SYSTEM),
        "implementationHashes": {p.name: file_hash(p) for p in sorted(package.glob("*.py"))},
        "sdkVersion": version("google-genai"),
    }


def save_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def generate_concept_batch(
    blueprint: Path,
    output_dir: Path,
    settings: GenerationSettings,
    *,
    max_new: int = 2,
    client=None,
) -> dict:
    """Persist each valid question atomically; a failed later call preserves earlier ones.

    An external single-writer lease owns the directory. Never overwrite incompatible
    candidates, and never infer content approval from format/schema checks.
    """
    if not 1 <= max_new <= 3:
        raise ValueError("batch size must be 1..3")
    bank = ConceptBlueprint.model_validate_json(blueprint.read_text())
    identity = generation_identity(blueprint, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "generation-input.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("generation settings or inputs changed; use a new directory")
    elif list(output_dir.glob("q_*.json")):
        raise ValueError("unbound candidates exist; use a new directory")
    else:
        save_json(marker, identity)
    expected = {f"{s.question_id}.json" for s in bank.question_specs}
    if any(p.name not in expected for p in output_dir.glob("q_*.json")):
        raise ValueError("unexpected candidate outside blueprint")
    questions = {}
    for spec in bank.question_specs:
        path = output_dir / f"{spec.question_id}.json"
        if not path.exists():
            continue
        question = ConceptGeneratedQuestion.model_validate_json(path.read_text())
        if (
            question.question_spec_id != spec.question_id
            or question.question_spec_hash != content_hash(spec.model_dump())
            or question.input_artifact_hash != identity["blueprintHash"]
            or question.prompt_version != PROMPT_VERSION
            or question.generation_model != settings.model
            or question.output_language != settings.output_language
            or quality_flags(question)
        ):
            raise ValueError("saved candidate differs from the generation inputs")
        questions[spec.question_id] = question
    pending = [s for s in bank.question_specs if s.question_id not in questions][:max_new]
    own_client = client is None and bool(pending)
    if own_client:
        client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(
                timeout=35000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
    try:
        for spec in pending:
            path = output_dir / f"{spec.question_id}.json"
            question = generate_concept_question(
                spec,
                settings,
                identity["blueprintHash"],
                client=client,
                raw_path=output_dir / "raw" / path.name,
            )
            # Provider is not allowed to change the task or metadata.
            if question.question_spec_hash != content_hash(spec.model_dump()):
                raise ValueError("provider changed question specification")
            save_json(path, question.model_dump())
            questions[spec.question_id] = question
    finally:
        if own_client:
            client.close()
    complete = len(questions) == len(bank.question_specs)
    if complete:
        load_candidates(blueprint, output_dir)
        review_dir = output_dir / "human-review"
        if not review_dir.exists():
            prepare_review(blueprint, output_dir, review_dir)
    result = {
        "contractVersion": BATCH_VERSION,
        "topicId": bank.topic_id,
        "blueprintHash": identity["blueprintHash"],
        "generationIdentityHash": content_hash(identity),
        "status": "CANDIDATES_READY" if complete else "GENERATING",
        "generatedCount": len(questions),
        "plannedCount": len(bank.question_specs),
        "contentReview": "pending",
        "diagnosisReady": False,
        "candidateHashes": {
            name: file_hash(output_dir / name)
            for name in sorted(expected)
            if (output_dir / name).exists()
        },
    }
    save_json(output_dir / "generation-report.json", result)
    return result
