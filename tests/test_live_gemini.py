import os
import re

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
    settings = load_settings(require_api_key=True)
    question = generate_question(
        fixture_bundle.question_specs[spec_index],
        artifact_hash=fixture_bundle.source_artifact_hash,
        generator=GeminiQuestionGenerator(settings),
        output_language=settings.output_language,
    )
    assert len(question.choices) == 4
    assert question.question_spec_id == fixture_bundle.question_specs[spec_index].question_id
    assert question.primary_concept == fixture_bundle.question_specs[spec_index].primary_concept
    assert question.target_difficulty == fixture_bundle.question_specs[spec_index].target_difficulty
    assert question.output_language == settings.output_language
    if settings.output_language == "ko-KR":
        assert re.search(r"[가-힣]", question.stem)
        assert all(re.search(r"[가-힣]", choice) for choice in question.choices)
        assert re.search(r"[가-힣]", question.explanation)
