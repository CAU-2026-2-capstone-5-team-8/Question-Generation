"""Concept diagnostics with explicit objectives; never create human approvals."""

import hashlib
import json
import re
from pathlib import Path

from google import genai
from google.genai import types

from question_generation.concept_contract import (
    ConceptGeneratedQuestion,
    ConceptProviderQuestion,
    ConceptQuestionSpec,
    content_hash,
)
from question_generation.config import GenerationSettings
from question_generation.content_format import (
    CONTENT_FORMAT,
    FORMAT_INSTRUCTION,
    content_format_flags,
)

PROMPT_VERSION = "concept-question-generation-prompt-v2"

SYSTEM = (
    """Author one English four-choice diagnostic question from the selected objective.
Assess existing concept knowledge. Do not provide a teaching passage, definition, worked example,
or rule that gives away the target knowledge. Short situations and numeric data are allowed.
Meaning: distinguish definitions, conditions, or properties, rather than merely translate a term.
Application: apply knowledge to a concrete changed situation or compute a result.
Reasoning: choose a valid explanation, necessary condition, or counterexample. Do not disguise a
definition-recall task as reasoning. Multiple-choice reasoning measures recognition of an argument,
not the ability to write a proof. Stay within the supplied objective and primary concept.
Use small exact integers or fractions and independently verify all mathematics. Exactly one option
must be correct. Use plausible misconception distractors. Do not use typos, silly answers,
answer length, grammar, meta-choices, or other surface clues. Avoid negative question stems.
Do not mention named books or pretend to quote a source. TOC references select assessment topics;
they do not supply factual text. Include a concise explanation of why each option is right or wrong.
Return only stem, choices, correct_choice_index, explanation in the requested schema."""
    + "\n"
    + FORMAT_INSTRUCTION
)


def render_concept_prompt(spec: ConceptQuestionSpec) -> str:
    return json.dumps(
        {
            "primary_concept": spec.primary_concept,
            "ability": spec.ability,
            "assessment_objective": spec.assessment_objective,
            "misconception_targets": spec.misconception_targets,
            "target_difficulty": spec.target_difficulty,
            "measurement_context": spec.measurement_context,
            "content_format": CONTENT_FORMAT,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def quality_flags(output: ConceptProviderQuestion) -> list[str]:
    fields = [output.stem, *output.choices, output.explanation]
    flags = []
    for text in fields:
        flags.extend(content_format_flags(text))
    if any(re.search(r"\b(?:TODO|TBD|FIXME)\b|\[insert", s, re.I) for s in fields):
        flags.append("placeholder")
    if any(
        re.search(r"[\u0400-\u052f\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", s)
        for s in fields
    ):
        flags.append("non-English source text")
    answer = " ".join(output.choices[output.correct_choice_index].casefold().split())
    if len(answer) >= 8 and answer in " ".join(output.stem.casefold().split()):
        flags.append("verbatim answer in stem")
    if any(re.search(r"\b(?:all|none) of the above\b", s, re.I) for s in output.choices):
        flags.append("meta-choice")
    return flags


def assemble_question(spec, output, *, model, language, artifact_hash, usage):
    flags = quality_flags(output)
    if flags:
        raise ValueError("question failed structural checks: " + ", ".join(flags))
    data = dict(
        **output.model_dump(),
        generated_question_version="generated-question-v5",
        question_spec_id=spec.question_id,
        topic_id=spec.topic_id,
        primary_concept=spec.primary_concept,
        ability=spec.ability,
        cognitive_operation=spec.cognitive_operation,
        measurement_context=spec.measurement_context,
        target_difficulty=spec.target_difficulty,
        assessment_objective=spec.assessment_objective,
        evidence_references=[r.model_dump() for r in spec.evidence_references],
        question_spec_hash=content_hash(spec.model_dump()),
        input_artifact_hash=artifact_hash,
        generation_model=model,
        output_language=language,
        prompt_version=PROMPT_VERSION,
        generation_config_version="concept-question-generation-config-v1",
    )
    data["generated_question_id"] = "gq_" + content_hash(data).split(":")[1][:32]
    data["usage"] = usage
    return ConceptGeneratedQuestion.model_validate(data)


def generate_concept_question(
    spec: ConceptQuestionSpec,
    settings: GenerationSettings,
    artifact_hash: str,
    *,
    client=None,
    raw_path: Path | None = None,
) -> ConceptGeneratedQuestion:
    own_client = client is None
    client = client or genai.Client(api_key=settings.api_key)
    try:
        response = client.models.generate_content(
            model=settings.model,
            contents=render_concept_prompt(spec),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=settings.temperature,
                seed=settings.seed,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_json_schema=ConceptProviderQuestion.model_json_schema(),
            ),
        )
        if raw_path is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(response.text or "", encoding="utf-8")
        parsed = response.parsed
        if isinstance(parsed, ConceptProviderQuestion):
            output = parsed
        elif isinstance(parsed, dict):
            output = ConceptProviderQuestion.model_validate(parsed)
        else:
            raise ValueError("provider returned no structured concept question")
        metadata = response.usage_metadata
        usage = {
            key: getattr(metadata, name, None)
            for key, name in (
                ("input_tokens", "prompt_token_count"),
                ("output_tokens", "candidates_token_count"),
                ("total_tokens", "total_token_count"),
            )
        }
        return assemble_question(
            spec,
            output,
            model=settings.model,
            language=settings.output_language,
            artifact_hash=artifact_hash,
            usage=usage,
        )
    finally:
        if own_client:
            client.close()


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_review_packet(blueprint, questions, output: Path) -> None:
    by_spec = {s.question_id: s for s in blueprint.question_specs}
    lines = [
        "# 선형대수 새 진단 문항 검토",
        "",
        "생성 후보입니다. 자동 구조 검사를 통과했으며 사람의 승인 기록은 없습니다.",
        "정답·보기·목표·표현을 검토한 뒤 HumanQuestionReview JSONL로 승인합니다.",
        "",
    ]
    for index, q in enumerate(questions, 1):
        spec = by_spec[q.question_spec_id]
        if q.question_spec_hash != content_hash(spec.model_dump()):
            raise ValueError("question and review blueprint disagree")
        lines += [
            f"## {index}. {q.primary_concept} / {q.ability}",
            "",
            f"목표: {q.assessment_objective}",
            "",
            f"관찰 조건: {q.measurement_context} · 설계 난도 {q.target_difficulty}",
            "",
            q.stem,
            "",
        ]
        lines += [f"{i + 1}. {text}" for i, text in enumerate(q.choices)]
        lines += [
            "",
            f"정답: {q.correct_choice_index + 1}",
            "",
            q.explanation,
            "",
            "예상 오개념: " + "; ".join(spec.misconception_targets),
            "",
            f"생성 ID: `{q.generated_question_id}`",
            "",
        ]
    output.write_text("\n".join(lines), encoding="utf-8")
