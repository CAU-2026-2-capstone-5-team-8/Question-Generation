import json
from dataclasses import replace

import pytest

from question_generation import concept_batch
from question_generation.concept_contract import ConceptBlueprint
from question_generation.concept_generation import assemble_question
from question_generation.config import GenerationSettings
from tests.test_concept_generation import output, spec


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text(
        ConceptBlueprint(
            blueprint_version="concept-assessment-blueprint-v2",
            topic_id="linear-algebra",
            question_specs=[spec(ability) for ability in ("meaning", "application", "reasoning")],
        ).model_dump_json()
    )
    calls = []

    def generate(target, settings, artifact_hash, **kwargs):
        calls.append(target.question_id)
        return assemble_question(
            target,
            output(),
            model=settings.model,
            language=settings.output_language,
            artifact_hash=artifact_hash,
            usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
        )

    monkeypatch.setattr(concept_batch, "generate_concept_question", generate)
    return blueprint, tmp_path / "questions", GenerationSettings(api_key="private-test-key"), calls


def test_batches_resume_without_duplicates_and_review_remains_blank(bundle):
    blueprint, directory, settings, calls = bundle
    first = concept_batch.generate_concept_batch(
        blueprint, directory, settings, max_new=2, client=object()
    )
    assert first["generatedCount"] == 2 and first["status"] == "GENERATING"
    second = concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
    assert second["generatedCount"] == 3 and second["status"] == "CANDIDATES_READY"
    assert len(calls) == len(set(calls)) == 3
    assert (
        concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
        == second
    )
    assert len(calls) == 3
    assert second["contentReview"] == "pending" and second["diagnosisReady"] is False
    rows = [
        json.loads(line)
        for line in (directory / "human-review/reviews.jsonl").read_text().splitlines()
    ]
    assert all(row["status"] is None and row["correct"] is None for row in rows)
    assert "private-test-key" not in (directory / "generation-input.json").read_text()
    human = directory / "human-review/instructions.md"
    if human.exists():
        original = human.read_bytes()
        concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
        assert human.read_bytes() == original


def test_failed_later_call_preserves_the_first_candidate(bundle, monkeypatch):
    blueprint, directory, settings, calls = bundle
    generate = concept_batch.generate_concept_question

    def fail_later(*args, **kwargs):
        if len(calls) == 1:
            raise RuntimeError("simulated disconnected provider")
        return generate(*args, **kwargs)

    monkeypatch.setattr(concept_batch, "generate_concept_question", fail_later)
    with pytest.raises(RuntimeError):
        concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
    assert len(list(directory.glob("q_*.json"))) == 1
    monkeypatch.setattr(concept_batch, "generate_concept_question", generate)
    report = concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
    assert report["generatedCount"] == 3 and len(calls) == 3


@pytest.mark.parametrize("change", ["model", "temperature", "tampered", "extra"])
def test_changed_inputs_and_tampered_candidates_fail_without_regeneration(bundle, change):
    blueprint, directory, settings, calls = bundle
    concept_batch.generate_concept_batch(blueprint, directory, settings, max_new=1, client=object())
    if change == "model":
        settings = replace(settings, model="other-model")
    elif change == "temperature":
        settings = replace(settings, temperature=0.4)
    elif change == "tampered":
        path = next(directory.glob("q_*.json"))
        data = json.loads(path.read_text())
        data["stem"] = "Tampered question text"
        path.write_text(json.dumps(data))
    else:
        (directory / ("q_" + "f" * 20 + ".json")).write_text("{}")
    with pytest.raises(ValueError):
        concept_batch.generate_concept_batch(blueprint, directory, settings, client=object())
    assert len(calls) == 1
