import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from question_generation.concept_generation import render_concept_prompt
from question_generation.content_format import canonical_generated_math, content_format_flags


def test_generated_symbolic_math_keeps_expressions_and_rejects_currency_ambiguity():
    raw = r"Bundle $A$ costs $p^A = (2, 1)$; compare $p^A \cdot x^B = 9$."
    normalized = canonical_generated_math(raw)
    assert normalized == r"Bundle \(A\) costs \(p^A = (2, 1)\); compare \(p^A \cdot x^B = 9\)."
    assert content_format_flags(normalized) == []
    assert canonical_generated_math(normalized) == normalized
    assert (
        canonical_generated_math("Subtract $x-y$; negate $-x$.")
        == r"Subtract \(x-y\); negate \(-x\)."
    )
    for ambiguous in ["Costs $5 and $10", "Costs $5", "$x$$", "$x\n+2$", "Pay $5-$10"]:
        assert canonical_generated_math(ambiguous) == ambiguous


@pytest.mark.parametrize(
    "text",
    [
        r"In \(\mathbb{R}^2\), take \(b_1=(1,1)\).",
        "A matrix:\n" + r"\[\begin{bmatrix}1&2\\0&1\end{bmatrix}\]",
        r"Use \(\frac{1}{2}\) and \(\{v_1,v_2\}\).",
        "**Explanation**\n\n- First reason\n- Second reason",
    ],
)
def test_supported_content(text):
    assert content_format_flags(text) == []


@pytest.mark.parametrize(
    "text,flag",
    [
        (r"In \(\mathbb{R}^2", "unclosed math delimiter"),
        (r"\(\frac{1}{2\)", "unbalanced math braces"),
        (r"\(\begin{bmatrix}1&2\end{pmatrix}\)", "unmatched math environment"),
        ("An array [[1,2],[0,1]] in R^2", "unformatted mathematical notation"),
        ("$x^2$", "use canonical math delimiters"),
        ("A \bmatrix", "invalid JSON escape in content"),
        ("A \theta", "invalid JSON escape in content"),
        ("\\(x\n+1\\)", "line break inside inline math"),
        ('<img src="https://example.invalid">', "unsupported markup"),
    ],
)
def test_malformed_content_has_actionable_flags(text, flag):
    assert flag in content_format_flags(text)


def test_prompt_names_content_format_and_all_local_drafts_pass():
    spec = SimpleNamespace(
        primary_concept="matrix",
        ability="application",
        assessment_objective="Find an entry",
        misconception_targets=["wrong row"],
        target_difficulty=2,
        measurement_context="prior-knowledge",
    )
    assert json.loads(render_concept_prompt(spec))["content_format"] == "markdown-latex-v1"
    rows = json.loads(
        (Path(__file__).parents[1] / "examples/concept-local-drafts.json").read_text()
    )
    for row in rows:
        for field in (row["stem"], *row["choices"], row["explanation"]):
            assert content_format_flags(field) == [], (row["concept_id"], row["ability"], field)
