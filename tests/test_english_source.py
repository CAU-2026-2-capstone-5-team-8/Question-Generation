from types import SimpleNamespace

import pytest

from question_generation.config import DEFAULT_OUTPUT_LANGUAGE, GenerationSettings, load_settings
from question_generation.errors import GenerationConfigurationError, MalformedProviderOutputError
from question_generation.gemini import GeminiQuestionGenerator
from question_generation.prompt import DISPLAY_GROUNDED_REVISION_INSTRUCTION, build_prompt


def test_source_generation_and_revision_default_to_english(vocabulary_spec):
    assert DEFAULT_OUTPUT_LANGUAGE == "en-US"
    prompt = build_prompt(vocabulary_spec)
    assert prompt.output_language == "en-US"
    assert "Korean" not in prompt.system_instruction
    assert "Korean" not in DISPLAY_GROUNDED_REVISION_INSTRUCTION
    assert "authoritative output language" in DISPLAY_GROUNDED_REVISION_INSTRUCTION


def test_english_provider_output_rejects_untranslated_korean(vocabulary_spec, vocabulary_output):
    response = SimpleNamespace(
        parsed=vocabulary_output.model_copy(update={"stem": "프로세스의 정의는 무엇인가?"})
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    generator = GeminiQuestionGenerator(GenerationSettings(api_key="fixture"), client=client)
    with pytest.raises(MalformedProviderOutputError, match="untranslated"):
        generator.generate(build_prompt(vocabulary_spec))


def test_source_cli_rejects_legacy_korean_environment(monkeypatch):
    monkeypatch.setenv("QUESTION_GENERATION_LANGUAGE", "ko-KR")
    with pytest.raises(GenerationConfigurationError, match="must be English"):
        load_settings(require_api_key=False)
    monkeypatch.setenv("QUESTION_GENERATION_LANGUAGE", "en-GB")
    assert load_settings(require_api_key=False).output_language == "en-GB"
