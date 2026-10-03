# BookMatch Question Generation

This repository turns an already-selected, versioned ML `QuestionSpec` into one validated
multiple-choice question. It does **not** select concepts, infer difficulty, rank books, or decide
what should be assessed.

The production boundary is intentionally narrow:

- supported v2: Level 1 `vocabulary/recognize` and `background_knowledge/recall`
- supported grounded v3: single-source Level 2 `comprehension/apply` with a validated
  `generation-grounding-v1` artifact
- supported display-grounded v4: the same narrow target with a validated
  `generation-grounding-v2` artifact and separate `passage` / `stem` fields
- rejected: ungrounded comprehension, `compare`, `relate`, `integrate`, `infer`, multi-step
  reasoning, arbitrary relation reasoning, and all other operation/difficulty combinations
- output: exactly four choices, one correct answer, an explanation, generation provenance, and
  exact input provenance

## Component boundary

```text
Data-Pipeline  collects canonical evidence
      ↓
ML             chooses the concept, operation, difficulty, and evidence-backed QuestionSpec
      ↓
Question-Generation  renders one prompt, calls Gemini, parses structured output, validates it
      ↓
Backend        stores questions, assessment responses, and lifecycle state
      ↓
Frontend       presents questions and responses
```

No runtime import crosses repository boundaries. The input is a versioned JSON artifact written by
ML. The generator treats its target fields as authoritative.

## Requirements and setup

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- the official [`google-genai`](https://googleapis.github.io/python-genai/) Python SDK

```bash
uv sync
cp .env.example .env
```

The CLI reads process environment variables; it does not load `.env` itself. Export the values or
source the file in your shell.

```text
GEMINI_API_KEY=                 # required only for a live generate call
QUESTION_GENERATION_MODEL=gemini-3.5-flash-lite
QUESTION_GENERATION_LANGUAGE=en-US
```

The default model and output language are centralized in `question_generation.config`. The output
language defaults to `en-US`; CLI generation and revision require an English locale (for example
`en-US` or `en-GB`). An existing `.env` with `ko-KR` must be updated before using the CLI.
`gemini-generation-config-v3` records this policy. Historical Korean artifacts and their review
identities are preserved. Optional Korean display translation belongs to the frontend presentation
layer and does not change the English source question, choice IDs, answer key, or concept IDs.

## Input contract

The authoritative contract is ML `question-spec-v1`. The adapter accepts either:

1. a complete ML `AssessmentBlueprint` JSON plus `--question-id`, or
2. a JSON file containing one `QuestionSpec`.

The local Pydantic contract preserves and validates:

- identity and target: `question_id`, `topic_id`, `question_type`, `cognitive_operation`,
  `primary_concept`, `related_concepts`, `prerequisite_concepts`, `target_difficulty`
- target rationale: `difficulty_rationale`, `evidence_summary`
- audit provenance: `supporting_book_ids`, `supporting_evidence`, `source_document_ids`
- versions: `question_spec_version`, `config_version`, `config_hash`

The adapter also hashes the exact input artifact bytes. It never imports ML Python modules and never
chooses a replacement spec.

Grounded comprehension requires an ML `generation-grounding-v1` or `generation-grounding-v2` JSON
artifact. V1 binds one exact 600–1,800 character canonical English analysis substring (`en_text`
when present, otherwise original English `text`), while the document content hash still identifies
the collected original and the canonical file hash binds both fields. V2 preserves that role as
`source_passage_text` and adds separately hashed `display_passage_text` under the reviewed,
source-hash-bound `pdf-display-normalization-v1` policy. Both strict local contracts bind the exact
QuestionSpec and blueprint, document/book/source identities, document and source hashes,
license/rights provenance, and all four canonical dataset hashes. The adapter rejects any mismatch;
it does not read Data-Pipeline files or resolve source IDs itself.

The committed fixture at `tests/fixtures/ml_question_specs.json` contains one real vocabulary spec
and one real background-knowledge spec minimized from:

```text
ML commit: e3adc460121b1e04d98728fc56eb657cf56a7be2
artifact: data/output/operating_systems_assessment_blueprint.json
SHA-256: 3822916d24b155b50ee3efb2581b43aa5587926eb36b3f7e958bb1f740d47f9c
```

It includes evidence identifiers and summaries, but no raw book prose.

## Structured Gemini generation

The production adapter uses `google.genai.Client.models.generate_content` with:

- `ProviderQuestion.model_json_schema()` as `response_json_schema`
- `application/json` response MIME type
- `temperature=0` and `seed=0`
- one request per `QuestionSpec`
- `response.parsed` only; there is no regex or raw-JSON fallback

This follows Google's official [JSON Schema structured-output
pattern](https://googleapis.github.io/python-genai/#json-response-schema). The prompt does not repeat
the JSON schema. The pipeline is:

```text
QuestionSpec → versioned prompt → Gemini structured response → Pydantic → domain validation
             → GeneratedQuestion
```

There is no retry loop. Provider failures are categorized as configuration, temporarily
unavailable, provider error, malformed structured output, or deterministic validation failure.

## Prompt and evidence boundary

`question-generation-prompt-v2` sends only the selected target plus compact audit metadata:
concept IDs, difficulty rationale, evidence summary, evidence types/IDs, and supporting book IDs.
It does not send a whole topic list, taxonomy, or raw book prose.

The configured English locale is authoritative. The stem, choices, explanation, and revision
instructions use English while canonical concept/topic identifiers remain unchanged. The Gemini
adapter rejects untranslated Korean/CJK text in English question bodies before finalization.

Evidence metadata proves why ML selected the target; it is not source text. The prompt prohibits
quotations and book/page/chapter claims because those claims cannot be grounded from identifiers
alone.

For the grounded slice, `question-generation-grounded-prompt-v3` sends exactly the passage selected
by ML together with its document and passage hashes. The provider must use only that passage, must
not require external knowledge, and returns only the question sentence. Finalization prepends the
unchanged source passage to the stem so the assessment user sees `passage + question + choices`.
No whole chapter, fallback document, summary, source URL, or synthetic prose is sent.

With `generation-grounding-v2`, `question-generation-grounded-prompt-v4` sends only the reviewed
display passage as factual source text. It includes both source/display hashes and the normalization
policy for audit, but not the raw passage text or source URL. Final `generated-question-v4` keeps the
display `passage` separate from the provider's question-only `stem`; v2 and v3 serialization and IDs
are unchanged.

Inspect the exact prompt without an API key or network call:

```bash
uv run bookmatch-question-generation render-prompt \
  --blueprint ../ML/data/output/operating_systems_assessment_blueprint.json \
  --question-id q_6b2414f8ad5225c1aef7
```

The `generate` command offers the same safe dry run:

```bash
uv run bookmatch-question-generation generate \
  --blueprint ../ML/data/output/operating_systems_assessment_blueprint.json \
  --question-id q_6b2414f8ad5225c1aef7 \
  --dry-run
```

Inspect the exact grounded prompt without a key or network call:

```bash
uv run bookmatch-question-generation render-prompt \
  --blueprint ../ML/data/output/linear_algebra_assessment_blueprint_reviewed.json \
  --question-id q_375e5b6bef551015f67c \
  --grounding ../ML/data/output/linear_algebra_matrix_grounding.json
```

Use the same command with the v2 grounding artifact to render the separated v4 prompt:

```bash
uv run bookmatch-question-generation generate \
  --dry-run \
  --blueprint ../ML/data/output/linear_algebra_assessment_blueprint_reviewed.json \
  --question-id q_375e5b6bef551015f67c \
  --grounding ../ML/data/output/linear_algebra_matrix_grounding_v2.json
```

## Generate one question

From a blueprint:

```bash
uv run bookmatch-question-generation generate \
  --blueprint ../ML/data/output/operating_systems_assessment_blueprint.json \
  --question-id q_6b2414f8ad5225c1aef7 \
  --output data/generated/process-question.json
```

From a standalone spec:

```bash
uv run bookmatch-question-generation generate \
  --spec /path/to/question-spec.json \
  --output data/generated/question.json
```

From a grounded comprehension target:

```bash
uv run bookmatch-question-generation generate \
  --blueprint ../ML/data/output/linear_algebra_assessment_blueprint_reviewed.json \
  --question-id q_375e5b6bef551015f67c \
  --grounding ../ML/data/output/linear_algebra_matrix_grounding.json \
  --output data/generated/la-matrix-comprehension.json
```

Passing `linear_algebra_matrix_grounding_v2.json` instead produces `generated-question-v4` with a
separate display `passage`. Live generation is not required to validate or dry-run this contract.

Grounded comprehension requires `--blueprint`; standalone `--spec` input is rejected because it
cannot prove the authoritative blueprint hash and canonical dataset hashes.

`data/generated/` is ignored. The CLI reports the generated ID, target, model, output path, and
provider-reported token counts when present. It never estimates missing token usage.

## Revise one generated question

Use the revision command when human feedback identifies a concrete problem in an already generated
question. The original artifact stays unchanged; write the replacement to a new versioned path.

```bash
uv run bookmatch-question-generation revise \
  --blueprint /path/to/assessment-blueprint.json \
  --question-id q_example \
  --previous data/generated/example.json \
  --feedback-file data/generated/revision-feedback/example.txt \
  --output data/generated/example-rev1.json
```

The command verifies that the previous question belongs to the exact same `QuestionSpec`, input
artifact, and output language before calling Gemini. It uses
`question-generation-revision-prompt-v1`, returns the unchanged `GeneratedQuestion` schema, and
runs the same deterministic validation as first-pass generation. Revision feedback is generation
input, not a `HumanQuestionReview`; the command does not assign an approval status or write the
canonical review JSONL.

`GeneratedQuestion` does not currently contain a `revised_from` field. A revision is traceable by
its revision prompt version and new deterministic ID together with the preserved original artifact
and the documented review workflow. Canonical lineage is deferred until Backend lifecycle needs
justify a separate schema/version change.

For ungrounded targets, the revision command remains restricted to the existing v2 Level 1
targets. Grounded v3 revision uses the same command with an authoritative blueprint and the exact
grounding artifact:

```bash
uv run bookmatch-question-generation revise \
  --blueprint /path/to/assessment-blueprint.json \
  --question-id q_example \
  --grounding /path/to/generation-grounding.json \
  --previous data/generated/example.json \
  --feedback-file data/generated/revision-feedback/example.txt \
  --output data/generated/example-rev1.json
```

The grounded path uses `question-generation-grounded-revision-prompt-v1`, requires
`generated-question-v3`, revalidates the blueprint and canonical hashes, and reconstructs the final
stem from the unchanged grounding passage. Standalone `--spec` grounded revision and any mismatch
in passage, source document, grounding hash, target identity, or output language fail closed. The
existing ungrounded v2 revision path and its prompt remain unchanged.

Display-grounded v4 revision uses the same command with `generation-grounding-v2` and a preserved
`generated-question-v4` input. It uses
`question-generation-display-grounded-revision-prompt-v2`, sends the display passage only as the
factual basis, and accepts provider output only for `stem`, `choices`, `correct_choice_index`, and
`explanation`. Finalization copies the validated previous `passage` byte-for-byte and preserves all
raw/display hashes, normalization policy, source document identity, and grounding artifact hash.
The CLI preflights those deterministic inputs before requiring an API key or constructing the
provider, while the generation function repeats the same validation defensively. Any mismatch
fails with `InputContractError`. The original artifact remains unchanged and the replacement
receives a new deterministic ID.

## Output and deterministic ID

`generated-question-v2` contains:

- immutable target identity, cognitive operation, prerequisites, difficulty rationale, and evidence
  summary copied from the input spec
- `stem`, exactly four `choices`, zero-based `correct_choice_index`, and `explanation`
- `generation_model`, `output_language`, `prompt_version`, and `generation_config_version`
- input spec version/config version/config hash, supporting book/evidence/document IDs, and the exact
  input artifact hash
- provider-reported input/output/total token counts, or `null` when unavailable

`generated_question_id` is a deterministic SHA-256-derived ID over the identity, provenance, and
question output, including `output_language`. Gemini does not propose or control it. Provider token
usage is excluded because it can vary without changing the question.

`generated-question-v3` uses the same field names for the single grounded
`comprehension/apply/Level 2` slice. Its `stem` contains the exact passage followed by the provider's
question, `source_document_ids` contains exactly one ML-selected document, and
`input_artifact_hash` identifies the exact grounding artifact (which in turn binds the blueprint and
canonical dataset). Existing v2 generation, serialization, and deterministic IDs are unchanged.

`generated-question-v4` is a separate schema. `passage` contains the deterministic display text and
`stem` contains only the question. It retains the grounding version, source extraction policy,
source passage hash, display passage hash, display normalization policy, single source document,
and exact grounding artifact hash. Its deterministic ID covers those fields. See the
[display-grounded v4 report](docs/experiments/display-grounded-question-v4.md) for the provider
passage decision, dry-run evidence, and Backend boundary.

## Deterministic validation

Generation fails if any of these checks fail:

- not exactly four choices, invalid correct index, blank stem/choice/explanation
- duplicate choices after Unicode, case, and whitespace normalization
- changed spec ID, topic, type, operation, primary concept, related concepts, or difficulty
- answer text copied verbatim into the stem in an obvious answer leak
- TODOs or placeholders
- an explanation that names a different choice number or letter as correct
- a spec outside the supported production boundaries
- a missing, altered, unlicensed, wrong-document, wrong-book, wrong-topic, or wrong-QuestionSpec
  grounding artifact
- grounded targets other than single-source `comprehension/apply/Level 2`

Structured output constrains syntax; these checks enforce cross-field semantics.
Language quality is intentionally left to human review instead of using brittle character-ratio
heuristics that could reject valid technical terms.

## Human review

Generated questions still require human review. `HumanQuestionReview` and
`examples/human_review.example.jsonl` use:

- `status`: `approve`, `reject`, or `needs_revision`
- `correct`
- `concept_alignment` (1–5)
- `difficulty_appropriate`
- `distractor_quality` (1–5)
- `explanation_quality` (1–5)
- `notes`

JSONL is the canonical simple review format; it can be converted to CSV without changing field
meaning. Human approval, not successful parsing, is the quality decision.

## Tests and live smoke checks

Default checks are network-free and need no API key:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

They cover input parsing, unsupported targets, structured parsing, all domain validators,
deterministic IDs, prompt rendering, missing-key behavior, fake generation, CLI dry run, and both
real ML fixture types.

Live Gemini tests are opt-in and never run in CI:

```bash
GEMINI_LIVE_TEST=1 uv run pytest -q -m live
```

These are smoke checks for provider compatibility and contract validity, not claims that a question
is educationally correct. Save any inspected live result only under ignored `live-output/` or
`data/generated/`.

## Non-goals and current limitations

This milestone does not modify Backend, Frontend, Data-Pipeline, ML ranking, concept graphs,
matchers, vector databases, web grounding, async queues, cloud deployment, or authentication. It
contains no topic-specific generation conditionals.

The existing v2 targets still carry evidence metadata rather than source text and cannot justify
source-specific claims. Grounded v3/v4 are limited to one apply target, one source document, Level
2, and a fixed bounded passage policy. V4 currently accepts only reviewed source/hash normalization
rules; an unseen passage fails closed until reviewed. Integrate/Level 3, multiple-source synthesis,
and v4 revision remain unsupported. All question types use four choices, so
`background_knowledge / recall` measures cued recognition more closely than pure free recall.

Backend currently imports only `generated-question-v2`, Level 1 vocabulary/background targets, and
empty source-document IDs. It therefore rejects grounded v3 and v4 by design. Supporting the final
grounded contract later requires an explicit Backend update; this repository does not weaken the
current importer.


## 개념별 진단 v2

[개편 계획](docs/concept-assessment-plan.md)에 따라 기존 명령과 분리된 `concept` 명령을 제공합니다.
ML의 `build-concept-assessment`가 실제 목차 근거를 연결한 6개 개념 × 3개 수행 목표를 생성합니다.
`concept-question-spec-v2`는 종전 용어·배경지식·독해 할당을 사용하지 않습니다.
`generated-question-v5`는 사전 지식 진단을 위한 정의·적용·추론 선택형 문항입니다.
추론 선택형 문항의 정답을 자유 서술 능력으로 해석하지 않습니다. 설계 난도는 검증된 심리측정 난도가 아닙니다.

```sh
# ML 저장소에서 먼저 실행
uv run bookmatch-ml build-concept-assessment --data-dir ../Data-Pipeline/data/processed --output data/output/concept-assessment-v2/blueprint.json

# 이 저장소에서: API를 호출하지 않는 프롬프트 확인
uv run bookmatch-question-generation concept generate --blueprint ../ML/data/output/concept-assessment-v2/blueprint.json --output-dir data/generated/concept-assessment-v2 --dry-run

# 외부 전송에 승인된 경우에만 실제 Gemini 생성. 기존 영어 원문 정책을 유지한다.
uv run bookmatch-question-generation concept generate --blueprint ../ML/data/output/concept-assessment-v2/blueprint.json --output-dir data/generated/concept-assessment-v2

# 외부 전송 없는 원본 로컬 초안. Gemini 생성이나 사람 승인으로 표시하지 않는다.
uv run bookmatch-question-generation concept local-drafts --blueprint ../ML/data/output/concept-assessment-v2/blueprint.json --drafts examples/concept-local-drafts.json --output-dir data/generated/concept-assessment-local-v2
```

출력에는 질문·보기·정답·해설과 문항 목표를 비교할 `review.md`가 포함됩니다. 저장된 후보는 원본 설계서·목표 해시를 검증하고 재사용하며, 변경된 로컬 초안은 새 출력 폴더에 보존합니다. JSON 구조 검사는 수학적 정확성이나 교육적 타당성에 대한 사람 검토를 대신하지 않습니다. 사람이 검토한 기존 `HumanQuestionReview` JSONL에서 approve/correct=true를 확인해야 Backend에 등록합니다. 실제 후보에 AI가 사람 승인 기록을 만들어 넣지 않습니다.

2026-10-03: 외부 생성 호출은 자동 승인 검토에서 보류됐습니다. 로컬 초안 18개를 별도 폴더에 만들었으며 사람 검토 전입니다. 외부 생성 결과와 구분하여 `generation_model=codex-local-draft`를 기록합니다.
## 문항의 Markdown·수식 표시

개념 문항의 텍스트 필드는 가벼운 Markdown과 LaTeX를 사용합니다. JSON 구조와 정답·개념 계약은 유지합니다.
생성 프롬프트는 `concept-question-generation-prompt-v2`이며 v1 산출물도 읽을 수 있습니다.
표기 규칙과 생성 후 실제 렌더링 검사 명령은 [문제 본문 형식](docs/question-content-format.md)을 참고하세요.
