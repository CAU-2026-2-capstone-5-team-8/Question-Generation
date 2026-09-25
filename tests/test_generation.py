from types import SimpleNamespace

import pytest

from question_generation.config import GenerationSettings
from question_generation.errors import (
    GenerationConfigurationError,
    MalformedProviderOutputError,
    ProviderError,
    ProviderUnavailableError,
)
from question_generation.gemini import GeminiQuestionGenerator
from question_generation.generation import FakeQuestionGenerator, generate_question
from question_generation.prompt import PromptPayload, build_prompt
from question_generation.schemas import ProviderQuestion, QuestionSpec, TokenUsage


def test_prompt_is_versioned_minimal_and_preserves_target(vocabulary_spec: QuestionSpec) -> None:
    prompt = build_prompt(vocabulary_spec)
    rendered = prompt.render()
    assert prompt.prompt_version == "question-generation-prompt-v1"
    assert vocabulary_spec.question_id in rendered
    assert vocabulary_spec.primary_concept in rendered
    assert vocabulary_spec.difficulty_rationale in rendered
    assert vocabulary_spec.config_hash in rendered
    assert "audit context only" in rendered
    assert "exactly four" in rendered
    assert "response_schema" not in rendered


def test_provider_response_schema_has_no_numeric_literal_const() -> None:
    schema = ProviderQuestion.model_json_schema()
    target_schema = schema["properties"]["target_difficulty"]
    assert target_schema["type"] == "integer"
    assert target_schema["minimum"] == 1
    assert target_schema["maximum"] == 3
    assert "const" not in target_schema

    def numeric_consts(value: object) -> list[int | float]:
        if isinstance(value, dict):
            found = []
            if isinstance(value.get("const"), int | float):
                found.append(value["const"])
            return found + [item for child in value.values() for item in numeric_consts(child)]
        if isinstance(value, list):
            return [item for child in value for item in numeric_consts(child)]
        return []

    assert numeric_consts(schema) == []


def test_fake_generation_preserves_provenance_and_is_deterministic(
    vocabulary_spec: QuestionSpec, vocabulary_output: ProviderQuestion
) -> None:
    usage = TokenUsage(input_tokens=101, output_tokens=52, total_tokens=153)
    fake = FakeQuestionGenerator(vocabulary_output, usage=usage)
    first = generate_question(
        vocabulary_spec,
        artifact_hash="sha256:" + "a" * 64,
        generator=fake,
    )
    second = generate_question(
        vocabulary_spec,
        artifact_hash="sha256:" + "a" * 64,
        generator=FakeQuestionGenerator(vocabulary_output, usage=usage),
    )
    assert first == second
    assert first.generated_question_id.startswith("gq_")
    assert first.question_spec_id == vocabulary_spec.question_id
    assert first.question_spec_config_hash == vocabulary_spec.config_hash
    assert first.supporting_book_ids == vocabulary_spec.supporting_book_ids
    assert first.supporting_evidence_ids == [
        item.evidence_id for item in vocabulary_spec.supporting_evidence
    ]
    assert first.usage == usage
    assert len(fake.prompts) == 1


def test_output_change_changes_deterministic_id(
    vocabulary_spec: QuestionSpec, vocabulary_output: ProviderQuestion
) -> None:
    first = generate_question(
        vocabulary_spec,
        artifact_hash="sha256:" + "a" * 64,
        generator=FakeQuestionGenerator(vocabulary_output),
    )
    changed_output = vocabulary_output.model_copy(
        update={"explanation": vocabulary_output.explanation + " It is independently scheduled."}
    )
    second = generate_question(
        vocabulary_spec,
        artifact_hash="sha256:" + "a" * 64,
        generator=FakeQuestionGenerator(changed_output),
    )
    assert first.generated_question_id != second.generated_question_id


class FakeModels:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def test_gemini_uses_pydantic_structured_output_and_sdk_usage(
    vocabulary_output: ProviderQuestion,
) -> None:
    response = SimpleNamespace(
        parsed=vocabulary_output,
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=45,
            total_token_count=165,
        ),
    )
    models = FakeModels(response)
    client = SimpleNamespace(models=models)
    generator = GeminiQuestionGenerator(
        GenerationSettings(api_key="test-key", model="gemini-test"), client=client
    )
    result = generator.generate(PromptPayload(system_instruction="system", contents="contents"))
    assert result.output == vocabulary_output
    assert result.usage == TokenUsage(input_tokens=120, output_tokens=45, total_tokens=165)
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ProviderQuestion
    assert models.calls[0]["model"] == "gemini-test"


def test_gemini_rejects_malformed_parsed_output() -> None:
    response = SimpleNamespace(parsed={"stem": "incomplete"}, usage_metadata=None)
    client = SimpleNamespace(models=FakeModels(response))
    generator = GeminiQuestionGenerator(
        GenerationSettings(api_key="test-key", model="gemini-test"), client=client
    )
    with pytest.raises(MalformedProviderOutputError):
        generator.generate(PromptPayload(system_instruction="system", contents="contents"))


class RaisingModels:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate_content(self, **kwargs: object) -> object:
        raise self.error


def test_gemini_error_categories_are_distinct() -> None:
    with pytest.raises(GenerationConfigurationError):
        GeminiQuestionGenerator(GenerationSettings(api_key=None))

    unavailable = GeminiQuestionGenerator(
        GenerationSettings(api_key="test-key"),
        client=SimpleNamespace(models=RaisingModels(TimeoutError())),
    )
    with pytest.raises(ProviderUnavailableError):
        unavailable.generate(PromptPayload(system_instruction="system", contents="contents"))

    failed = GeminiQuestionGenerator(
        GenerationSettings(api_key="test-key"),
        client=SimpleNamespace(models=RaisingModels(RuntimeError("provider failed"))),
    )
    with pytest.raises(ProviderError):
        failed.generate(PromptPayload(system_instruction="system", contents="contents"))
