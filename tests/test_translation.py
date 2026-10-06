import json
from types import SimpleNamespace

import pytest

from question_generation.config import GenerationSettings
from question_generation.translation import numbers_preserved, protect, restore, translate_question


def test_preserves_math_numbers_and_detects_loss_or_duplicates():
    source = r"GDP is 500 plus \(X-M\), not 40."
    masked, values = protect(source)
    assert restore(masked.replace("GDP is", "국내총생산은"), values).endswith(r"40.")
    assert restore("__BM_KEEP_001__ __BM_KEEP_000__ __BM_KEEP_002__", values) == r"\(X-M\) 500 40"
    with pytest.raises(ValueError, match="duplicated or missing"):
        restore("__BM_KEEP_001__ __BM_KEEP_000__ __BM_KEEP_000__", values)
    assert restore("1인당 산출량은 __BM_KEEP_000__입니다.", ["500"]) == "인당 산출량은 500입니다."
    assert restore("__BM_KEEP_000__인당", ["1"]) == "1인당"


def test_spelled_number_equivalence_does_not_allow_changed_or_extra_values():
    source = r"Two nonzero vectors in \(R^3\) have zero dot product."
    assert numbers_preserved(source, r"\(R^3\)의 0이 아닌 벡터 2개의 내적은 0입니다.")
    assert not numbers_preserved(source, r"\(R^3\)의 0이 아닌 벡터 3개의 내적은 0입니다.")
    assert not numbers_preserved(source, r"\(R^3\)의 0이 아닌 벡터 2개의 내적은 0이고 0입니다.")
    assert not numbers_preserved(source, r"\(R^4\)의 0이 아닌 벡터 2개의 내적은 0입니다.")
    assert not numbers_preserved("2 and zero", "3 및 0")


class Client:
    def __init__(self, payload, approve=True):
        self.calls = 0
        self.models = self
        self.payload = payload
        self.approve = approve

    def generate_content(self, **kwargs):
        self.calls += 1
        result = (
            self.payload
            if kwargs["config"].response_json_schema["title"] == "DisplayText"
            else {
                "equivalent": self.approve,
                "choices_order_preserved": True,
                "no_added_hints": True,
                "reason": "내용과 조건이 동일합니다.",
            }
        )
        return SimpleNamespace(text=json.dumps(result, ensure_ascii=False))


def test_translates_once_and_reuses_exact_source_cache(tmp_path):
    source = {
        "passage": None,
        "prompt": "GDP equals 500 plus 120. What is the result?",
        "choices": ["620", "600", "500", "120"],
    }
    client = Client(
        {
            "passage": None,
            "prompt": (
                "국내총생산은 __BM_KEEP_000__ 더하기 __BM_KEEP_001__입니다. 결과는 얼마인가요?"
            ),
            "choices": ["__BM_KEEP_000__"] * 4,
        }
    )
    settings = GenerationSettings(api_key="test", model="test-model")
    translated = translate_question(source, tmp_path, settings, client=client)
    assert translated["text"]["choices"] == source["choices"]
    assert translate_question(source, tmp_path, settings, client=client) == translated
    assert client.calls == 2


def test_semantic_rejection_is_not_published(tmp_path):
    source = {"passage": None, "prompt": "What is 500?", "choices": ["500", "120", "40", "80"]}
    client = Client(
        {
            "passage": None,
            "prompt": "__BM_KEEP_000__은 무엇인가요?",
            "choices": ["__BM_KEEP_000__"] * 4,
        },
        approve=False,
    )
    with pytest.raises(ValueError, match="correction"):
        translate_question(source, tmp_path, GenerationSettings(api_key="test"), client=client)
    assert not list(tmp_path.rglob("translation.json"))
    assert client.calls == 4
    assert len(list(tmp_path.rglob("provider-review*.json"))) == 2
    assert len(list(tmp_path.rglob("failure*.json"))) == 2
    with pytest.raises(ValueError, match="correction"):
        translate_question(source, tmp_path, GenerationSettings(api_key="test"), client=client)
    assert client.calls == 4


def test_rejected_translation_is_corrected_then_reviewed_again(tmp_path):
    source = {"passage": None, "prompt": "What is 500?", "choices": ["500", "120", "40", "80"]}
    payload = {
        "passage": None,
        "prompt": "__BM_KEEP_000__은 무엇인가요?",
        "choices": ["__BM_KEEP_000__"] * 4,
    }

    class CorrectingClient(Client):
        def generate_content(self, **kwargs):
            self.approve = self.calls >= 2
            if self.calls == 2:
                assert "failedChecks" in kwargs["contents"]
                assert "rejectedDraft" in kwargs["contents"]
            return super().generate_content(**kwargs)

    client = CorrectingClient(payload, approve=False)
    settings = GenerationSettings(api_key="test")
    result = translate_question(source, tmp_path, settings, client=client)
    assert result["status"] == "READY" and client.calls == 4
    assert translate_question(source, tmp_path, settings, client=client) == result
    assert client.calls == 4


def test_invalid_numeric_draft_is_corrected_without_approving_bad_draft(tmp_path):
    source = {"passage": None, "prompt": "What is 500?", "choices": ["500", "120", "40", "80"]}

    class CorrectingClient(Client):
        def generate_content(self, **kwargs):
            if self.calls == 1:
                assert "numeric or mathematical content changed" in kwargs["contents"]
                self.calls += 1
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "passage": None,
                            "prompt": "__BM_KEEP_000__은 무엇인가요?",
                            "choices": ["__BM_KEEP_000__"] * 4,
                        }
                    )
                )
            return super().generate_content(**kwargs)

    client = CorrectingClient(
        {
            "passage": None,
            "prompt": "1번 문제: __BM_KEEP_000__은 무엇인가요?",
            "choices": ["__BM_KEEP_000__"] * 4,
        }
    )
    result = translate_question(source, tmp_path, GenerationSettings(api_key="test"), client=client)
    assert result["status"] == "READY" and client.calls == 3
    assert result["text"]["prompt"] == "500은 무엇인가요?"
