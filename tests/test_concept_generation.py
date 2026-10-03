from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from question_generation.concept_contract import (
    ConceptBlueprint,
    ConceptGeneratedQuestion,
    ConceptProviderQuestion,
    ConceptQuestionSpec,
    content_hash,
)
from question_generation.concept_generation import (
    SYSTEM,
    assemble_question,
    generate_concept_question,
    render_concept_prompt,
)
from question_generation.config import GenerationSettings


def spec(ability="application"):
    operations = {"meaning": "recognize", "application": "apply", "reasoning": "infer"}
    data = dict(
        question_spec_version="concept-question-spec-v2",
        topic_id="linear-algebra",
        primary_concept="matrix",
        ability=ability,
        cognitive_operation=operations[ability],
        measurement_context="prior-knowledge",
        target_difficulty=2,
        assessment_objective="Find a product entry.",
        misconception_targets=["elementwise multiplication", "wrong row and column"],
        evidence_references=[{"book_id": "synthetic", "toc_entry_id": "synthetic-toc"}],
        config_version="synthetic-v1",
        config_hash="sha256:" + "a" * 64,
        canonical_file_hashes={
            k: "sha256:" + "a" * 64
            for k in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
    )
    data["question_id"] = "q_" + content_hash(data).split(":")[1][:20]
    return ConceptQuestionSpec.model_validate(data)


def output():
    return ConceptProviderQuestion(
        stem=r"Find entry \((1,1)\) of \(\begin{bmatrix}1&2\\0&1\end{bmatrix}\) squared.",
        choices=["1", "2", "3", "4"],
        correct_choice_index=0,
        explanation=r"The entry is \(1\cdot 1+2\cdot 0=1\).",
    )


def test_reasoning_target_is_supported_and_provider_cannot_change_metadata():
    target = spec("reasoning")
    question = assemble_question(
        target,
        output(),
        model="synthetic-provider",
        language="en-US",
        artifact_hash="sha256:" + "b" * 64,
        usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
    )
    assert question.ability == "reasoning"
    assert question.cognitive_operation == "infer"
    assert question.prompt_version == "concept-question-generation-prompt-v2"
    assert question.question_spec_hash == content_hash(target.model_dump())
    assert "misconception_targets" in render_concept_prompt(target)
    assert "counterexample" in SYSTEM
    data = question.model_dump()
    data["choices"][0] = "99"
    with pytest.raises(ValidationError, match="ID does not match"):
        ConceptGeneratedQuestion.model_validate(data)
    with pytest.raises(ValidationError):
        ConceptProviderQuestion.model_validate({**output().model_dump(), "ability": "meaning"})


def test_generic_provider_schema_keeps_old_provider_contract_unchanged():
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed=output().model_dump(), usage_metadata=None, text="{}")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    q = generate_concept_question(
        spec(), GenerationSettings(api_key=None), "sha256:" + "b" * 64, client=client
    )
    assert q.generated_question_version == "generated-question-v5"
    assert set(calls[0]["config"].response_json_schema["properties"]) == {
        "stem",
        "choices",
        "correct_choice_index",
        "explanation",
    }


@pytest.mark.parametrize("choices", [["A", "a", "B", "C"], ["A", "B", "C", " A "]])
def test_duplicate_or_untrimmed_choices_fail(choices):
    with pytest.raises(ValidationError):
        ConceptProviderQuestion.model_validate({**output().model_dump(), "choices": choices})


def test_generation_does_not_relabel_saved_v1_output_as_current_prompt(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from question_generation import concept_commands
    from question_generation.concept_generation import file_hash

    target = spec()
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text(
        ConceptBlueprint(
            blueprint_version="concept-assessment-blueprint-v2",
            topic_id="linear-algebra",
            question_specs=[target],
        ).model_dump_json()
    )
    question = assemble_question(
        target,
        output(),
        model="synthetic",
        language="en-US",
        artifact_hash=file_hash(blueprint),
        usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
    )
    data = question.model_dump()
    data["prompt_version"] = "concept-question-generation-prompt-v1"
    identity = {k: v for k, v in data.items() if k not in {"generated_question_id", "usage"}}
    data["generated_question_id"] = "gq_" + content_hash(identity).split(":")[1][:32]
    old = ConceptGeneratedQuestion.model_validate(data)
    directory = tmp_path / "saved"
    directory.mkdir()
    saved = directory / f"{target.question_id}.json"
    saved.write_text(old.model_dump_json())
    original = saved.read_bytes()
    monkeypatch.setattr(
        concept_commands, "load_settings", lambda **_: GenerationSettings(api_key=None)
    )

    def never_send(*args, **kwargs):
        raise AssertionError("No provider call is authorized in this test")

    monkeypatch.setattr(concept_commands, "generate_concept_question", never_send)
    result = CliRunner().invoke(
        concept_commands.app,
        ["generate", "--blueprint", str(blueprint), "--output-dir", str(directory)],
        terminal_width=160,
    )
    assert result.exit_code != 0
    import re

    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    message = " ".join(plain.replace("│", " ").split())
    assert "choose a new output directory" in message, result.output
    assert saved.read_bytes() == original
