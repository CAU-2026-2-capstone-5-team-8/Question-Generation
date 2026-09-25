import json
from pathlib import Path

import pytest

from question_generation.schemas import FixtureBundle, ProviderQuestion, QuestionSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "ml_question_specs.json"


@pytest.fixture(scope="session")
def fixture_bundle() -> FixtureBundle:
    return FixtureBundle.model_validate_json(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def vocabulary_spec(fixture_bundle: FixtureBundle) -> QuestionSpec:
    return fixture_bundle.question_specs[0]


@pytest.fixture
def background_spec(fixture_bundle: FixtureBundle) -> QuestionSpec:
    return fixture_bundle.question_specs[1]


@pytest.fixture
def vocabulary_output(vocabulary_spec: QuestionSpec) -> ProviderQuestion:
    return ProviderQuestion(
        question_spec_id=vocabulary_spec.question_id,
        topic_id=vocabulary_spec.topic_id,
        question_type="vocabulary",
        cognitive_operation="recognize",
        primary_concept=vocabulary_spec.primary_concept,
        related_concepts=[],
        target_difficulty=1,
        stem="Which description best defines an operating-system process?",
        choices=[
            "A program in execution with its own managed state",
            "A permanent collection of files on secondary storage",
            "A physical processor core reserved for one application",
            "A rule that translates virtual addresses into source code",
        ],
        correct_choice_index=0,
        explanation="A program in execution with managed state is a process.",
    )


def write_spec(path: Path, spec: QuestionSpec) -> None:
    path.write_text(
        json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
