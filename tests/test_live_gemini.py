import os

import pytest

from question_generation.config import load_settings
from question_generation.gemini import GeminiQuestionGenerator
from question_generation.generation import generate_question
from question_generation.schemas import FixtureBundle


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("GEMINI_LIVE_TEST") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="set GEMINI_LIVE_TEST=1 and GEMINI_API_KEY to run live Gemini smoke tests",
)
@pytest.mark.parametrize("spec_index", [0, 1], ids=["vocabulary", "background-knowledge"])
def test_live_gemini_supported_fixture(fixture_bundle: FixtureBundle, spec_index: int) -> None:
    question = generate_question(
        fixture_bundle.question_specs[spec_index],
        artifact_hash=fixture_bundle.source_artifact_hash,
        generator=GeminiQuestionGenerator(load_settings(require_api_key=True)),
    )
    assert len(question.choices) == 4
    assert question.question_spec_id == fixture_bundle.question_specs[spec_index].question_id
