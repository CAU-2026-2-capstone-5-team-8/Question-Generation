"""Centralized, versioned generation settings."""

import os
from dataclasses import dataclass

from question_generation.errors import GenerationConfigurationError

DEFAULT_MODEL = "gemini-2.5-flash-lite"
PROMPT_VERSION = "question-generation-prompt-v1"
GENERATION_CONFIG_VERSION = "gemini-generation-config-v1"
GENERATED_QUESTION_VERSION = "generated-question-v1"
SUPPORTED_QUESTION_SPEC_VERSION = "question-spec-v1"


@dataclass(frozen=True)
class GenerationSettings:
    api_key: str | None
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    seed: int = 0
    max_output_tokens: int = 1024


def load_settings(*, require_api_key: bool) -> GenerationSettings:
    """Read process environment without loading or exposing a secret file."""

    api_key = os.getenv("GEMINI_API_KEY") or None
    model = os.getenv("QUESTION_GENERATION_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise GenerationConfigurationError("QUESTION_GENERATION_MODEL must not be blank")
    if require_api_key and api_key is None:
        raise GenerationConfigurationError(
            "GEMINI_API_KEY is required for generation; render-prompt and --dry-run need no key"
        )
    return GenerationSettings(api_key=api_key, model=model)
