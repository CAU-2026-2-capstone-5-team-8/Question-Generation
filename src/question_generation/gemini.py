"""Official google-genai structured-output adapter."""

from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from question_generation.config import GenerationSettings
from question_generation.errors import (
    GenerationConfigurationError,
    MalformedProviderOutputError,
    ProviderError,
    ProviderUnavailableError,
)
from question_generation.generation import ProviderGeneration
from question_generation.prompt import PromptPayload
from question_generation.schemas import ProviderQuestion, TokenUsage


def _token_count(metadata: Any, name: str) -> int | None:
    value = getattr(metadata, name, None) if metadata is not None else None
    return value if isinstance(value, int) and value >= 0 else None


class GeminiQuestionGenerator:
    """One Gemini request per QuestionSpec, parsed through a Pydantic response schema."""

    def __init__(self, settings: GenerationSettings, *, client: Any | None = None) -> None:
        if settings.api_key is None:
            raise GenerationConfigurationError("GeminiQuestionGenerator requires an API key")
        self.settings = settings
        self.client = client or genai.Client(api_key=settings.api_key)

    def generate(self, prompt: PromptPayload) -> ProviderGeneration:
        if prompt.output_language != self.settings.output_language:
            raise GenerationConfigurationError(
                "prompt output_language must exactly match GenerationSettings"
            )
        try:
            response = self.client.models.generate_content(
                model=self.settings.model,
                contents=prompt.contents,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.system_instruction,
                    temperature=self.settings.temperature,
                    seed=self.settings.seed,
                    max_output_tokens=self.settings.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=ProviderQuestion.model_json_schema(),
                ),
            )
        except ValidationError as exc:
            raise MalformedProviderOutputError(
                "Gemini structured output failed during SDK parsing"
            ) from exc
        except errors.APIError as exc:
            status = getattr(exc, "code", None)
            if status == 429 or (isinstance(status, int) and status >= 500):
                raise ProviderUnavailableError(
                    f"Gemini is temporarily unavailable ({status})"
                ) from exc
            raise ProviderError(f"Gemini request failed ({status or 'unknown status'})") from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise ProviderUnavailableError("Gemini could not be reached") from exc
        except Exception as exc:  # SDK transport errors are not a stable public hierarchy.
            raise ProviderError(f"unexpected Gemini SDK failure: {type(exc).__name__}") from exc

        parsed = getattr(response, "parsed", None)
        try:
            if isinstance(parsed, ProviderQuestion):
                output = parsed
            elif isinstance(parsed, dict):
                output = ProviderQuestion.model_validate(parsed)
            else:
                raise MalformedProviderOutputError(
                    "Gemini returned no parsed ProviderQuestion; raw JSON fallback is disabled"
                )
        except ValidationError as exc:
            raise MalformedProviderOutputError(
                "Gemini structured output failed ProviderQuestion validation"
            ) from exc

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=_token_count(usage_metadata, "prompt_token_count"),
            output_tokens=_token_count(usage_metadata, "candidates_token_count"),
            total_tokens=_token_count(usage_metadata, "total_token_count"),
        )
        return ProviderGeneration(output=output, model=self.settings.model, usage=usage)
