"""Public error categories for the generation boundary."""


class QuestionGenerationError(Exception):
    """Base class for expected generation failures."""


class GenerationConfigurationError(QuestionGenerationError):
    """Local configuration is missing or invalid."""


class InputContractError(QuestionGenerationError):
    """The supplied blueprint or QuestionSpec violates the input contract."""


class UnsupportedQuestionSpecError(InputContractError):
    """The QuestionSpec is valid but outside the deliberately narrow v1 scope."""


class ProviderUnavailableError(QuestionGenerationError):
    """The provider could not serve the request because of a transient condition."""


class ProviderError(QuestionGenerationError):
    """The provider rejected or failed the request for a non-transient reason."""


class MalformedProviderOutputError(QuestionGenerationError):
    """The provider response did not match the requested structured schema."""


class GeneratedQuestionValidationError(QuestionGenerationError):
    """Structured output failed deterministic domain validation."""
