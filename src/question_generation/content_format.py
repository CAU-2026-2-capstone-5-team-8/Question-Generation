"""Lightweight authoring checks; frontend KaTeX validation checks actual rendering."""

import re

CONTENT_FORMAT = "markdown-latex-v1"
FORMAT_INSTRUCTION = r"""Keep the JSON schema. In stem, choices, and explanation, use
lightweight Markdown for paragraphs, **emphasis**, and lists. Write all mathematics with LaTeX:
inline \(...\), display \[...\] on separate lines. Do not use dollar delimiters.
Use \mathbb{R}^n, b_1, \frac{a}{b}, \cdot, and bmatrix/pmatrix for matrices.
Do not emit ASCII matrix arrays, plain R^2, b1, or multiplication stars in math.
Avoid raw HTML, images, links, code fences, and a whole Markdown question document.
Escape backslashes correctly in JSON strings: the decoded text must contain LaTeX commands,
not backspace/tab/newline characters produced by incorrect JSON escaping.
Close every math delimiter, brace, and environment. Put long formulas on display lines.
Formatting must not change mathematical values, distractors, or the correct option."""


def canonical_numeric_math(text: str) -> str:
    """Normalize balanced numeric dollar math only, preserving every math character.

    Currency, alphabetic expressions, unmatched or mixed delimiters are left alone
    and still fail the existing structural checks. Raw provider output is retained.
    """
    tokens = list(re.finditer(r"(?<!\\)\${1,2}", text))
    if len(tokens) % 2:
        return text
    parts = []
    position = 0
    for opening, closing in zip(tokens[::2], tokens[1::2], strict=True):
        math = text[opening.end() : closing.start()]
        if opening.group() != closing.group() or not re.fullmatch(r"[0-9\s.,+*/<>=^(){}\-]+", math):
            return text
        if not re.search(r"[0-9]", math) or (opening.group() == "$" and "\n" in math):
            return text
        parts.append(text[position : opening.start()])
        parts.append(
            (r"\(" + math + r"\)")
            if opening.group() == "$"
            else ("\n" + r"\[" + "\n" + math + "\n" + r"\]" + "\n")
        )
        position = closing.end()
    parts.append(text[position:])
    return "".join(parts)


def content_format_flags(text: str) -> list[str]:
    flags = []
    if re.search(r"\\{2,}[()[\]]", text):
        flags.append("overescaped content")
    if any(ord(c) < 32 and c not in "\n\r" for c in text):
        flags.append("invalid JSON escape in content")
    if re.search(r"</?[A-Za-z][^>]*>|!\[[^]]*\]\(|\[[^]]*\]\(https?://|```", text):
        flags.append("unsupported markup")
    plain = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", "", text, flags=re.S)
    if r"\n" in plain or r"\r" in plain:
        flags.append("overescaped content")
    if re.search(r"\[\[|\bR\^[0-9a-z]|\b[bcv][0-9]+\b", plain):
        flags.append("unformatted mathematical notation")
    if re.search(r"(?<!\\)\$", text):
        flags.append("use canonical math delimiters")
    opening = None
    start = 0
    for match in re.finditer(r"\\[()[\]]", text):
        token = match.group()
        if token in (r"\(", r"\["):
            if opening:
                flags.append("nested math delimiter")
            opening, start = token, match.end()
        else:
            expected = {r"\(": r"\)", r"\[": r"\]"}.get(opening)
            if token != expected:
                flags.append("unmatched math delimiter")
            else:
                math = text[start : match.start()]
                if opening == r"\(" and ("\n" in math or "\r" in math):
                    flags.append("line break inside inline math")
                depth = 0
                for brace in re.finditer(r"(?<!\\)[{}]", math):
                    depth += 1 if brace.group() == "{" else -1
                    if depth < 0:
                        break
                if depth != 0:
                    flags.append("unbalanced math braces")
                environments = []
                for env in re.finditer(r"\\(begin|end)\{([^}]+)\}", math):
                    if env[1] == "begin":
                        environments.append(env[2])
                    elif not environments or environments.pop() != env[2]:
                        flags.append("unmatched math environment")
                if environments:
                    flags.append("unmatched math environment")
            opening = None
    if opening:
        flags.append("unclosed math delimiter")
    return sorted(set(flags))


def canonical_math_layout(text: str) -> str:
    """Use inline delimiters for a one-line expression embedded in prose.

    Preserve every formula character. Standalone display blocks remain display
    blocks; incomplete or multiline expressions are left for validation.
    """

    def inline_if_embedded(match):
        start, end = match.span()
        before = text[text.rfind("\n", 0, start) + 1 : start]
        line_end = text.find("\n", end)
        after = text[end : line_end if line_end >= 0 else len(text)]
        if before.strip() or after.strip():
            return r"\(" + match.group(1) + r"\)"
        return match.group(0)

    return re.sub(r"(?<!\\)\\\[([^\n\r]*?)\\\]", inline_if_embedded, text)
