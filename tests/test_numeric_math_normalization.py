import pytest

from question_generation.content_format import (
    canonical_math_layout,
    canonical_numeric_math,
    content_format_flags,
)


def test_numeric_math_characters_are_preserved_and_canonical_delimiters_pass():
    original = "Path 1 costs $2$, Path 2 costs $5$, and $2 < 5$."
    converted = canonical_numeric_math(original)
    assert converted == r"Path 1 costs \(2\), Path 2 costs \(5\), and \(2 < 5\)."
    assert content_format_flags(converted) == []
    assert canonical_numeric_math(converted) == converted
    assert canonical_numeric_math("Value $$2^{32}$$.") == "Value \n\\[\n2^{32}\n\\]\n."


@pytest.mark.parametrize(
    "text",
    ["Costs $5", "Costs $5 and $10", "$x + 2$", "$2$$", "$2\n+3$", "$\\frac{2}{3}$", r"Costs \$5"],
)
def test_ambiguous_or_unsupported_notation_is_not_guessed(text):
    assert canonical_numeric_math(text) == text


@pytest.mark.parametrize("text", [r"Value \\(2\\)", r"Value \\[2\\]", r"Paragraph\nnext"])
def test_overescaped_provider_strings_cannot_be_accepted_as_renderable_content(text):
    assert "overescaped content" in content_format_flags(text)


def test_embedded_display_formula_preserves_characters_and_standalone_blocks():
    text = r"Demand is \[P = 120 - 3Q\] and cost is \[MC = 30\]."
    assert canonical_math_layout(text) == r"Demand is \(P = 120 - 3Q\) and cost is \(MC = 30\)."
    display = "A matrix:\n" + r"\[\begin{bmatrix}1&2\\0&1\end{bmatrix}\]" + "\nNext."
    assert canonical_math_layout(display) == display
    assert canonical_math_layout(r"Broken \[x+2") == r"Broken \[x+2"
