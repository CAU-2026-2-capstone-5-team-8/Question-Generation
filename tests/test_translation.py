import json
from types import SimpleNamespace

import pytest

from question_generation.config import GenerationSettings
from question_generation.translation import protect, restore, translate_question


def test_preserves_math_numbers_and_detects_reordering():
    source = r"GDP is 500 plus \(X-M\), not 40."
    masked, values = protect(source)
    assert restore(masked.replace("GDP is", "국내총생산은"), values).endswith(r"40.")
    with pytest.raises(ValueError, match="reordered"):
        restore("__BM_KEEP_001__ __BM_KEEP_000__ __BM_KEEP_002__", values)
    assert restore("1인당 산출량은 __BM_KEEP_000__입니다.", ["500"]) == "인당 산출량은 500입니다."
    assert restore("__BM_KEEP_000__인당", ["1"]) == "1인당"


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
            if self.calls == 1
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
