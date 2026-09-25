# BookMatch Question Generation

This repository turns an already-selected, versioned ML `QuestionSpec` into one validated
multiple-choice question. It does **not** select concepts, infer difficulty, rank books, or decide
what should be assessed.

The v1 boundary is intentionally narrow:

- supported: Level 1 `vocabulary/recognize` and `background_knowledge/recall`
- rejected: `comprehension`, prose-grounded questions, `compare`, `relate`, `apply`, `integrate`,
  `infer`, multi-step reasoning, and arbitrary relation reasoning
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
```

The default model is centralized in `question_generation.config`. A model override changes only
the provider model; it does not change the versioned prompt or validation contract.

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

`question-generation-prompt-v1` sends only the selected target plus compact audit metadata:
concept IDs, difficulty rationale, evidence summary, evidence types/IDs, and supporting book IDs.
It does not send a whole topic list, taxonomy, or raw book prose.

Evidence metadata proves why ML selected the target; it is not source text. The prompt prohibits
quotations and book/page/chapter claims because those claims cannot be grounded from identifiers
alone.

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

`data/generated/` is ignored. The CLI reports the generated ID, target, model, output path, and
provider-reported token counts when present. It never estimates missing token usage.

## Output and deterministic ID

`generated-question-v1` contains:

- immutable target identity, cognitive operation, prerequisites, difficulty rationale, and evidence
  summary copied from the input spec
- `stem`, exactly four `choices`, zero-based `correct_choice_index`, and `explanation`
- `generation_model`, `prompt_version`, and `generation_config_version`
- input spec version/config version/config hash, supporting book/evidence/document IDs, and the exact
  input artifact hash
- provider-reported input/output/total token counts, or `null` when unavailable

`generated_question_id` is a deterministic SHA-256-derived ID over the identity, provenance, and
question output. Gemini does not propose or control it. Provider token usage is excluded because it
can vary without changing the question.

## Deterministic validation

Generation fails if any of these checks fail:

- not exactly four choices, invalid correct index, blank stem/choice/explanation
- duplicate choices after Unicode, case, and whitespace normalization
- changed spec ID, topic, type, operation, primary concept, related concepts, or difficulty
- answer text copied verbatim into the stem in an obvious answer leak
- TODOs or placeholders
- an explanation that names a different choice number or letter as correct
- a spec outside the supported v1 boundary

Structured output constrains syntax; these checks enforce cross-field semantics.

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

This milestone does not modify or implement Backend, Frontend, Data-Pipeline, ML ranking, concept
graphs, matchers, adapters, comprehension/RAG, vector databases, web grounding, async queues,
cloud deployment, or authentication. It contains no topic-specific generation conditionals.

The current generator cannot justify source-specific claims because the `QuestionSpec` carries
evidence metadata rather than source text. It supports only two Level 1 target shapes. The next
milestone should gather human review results and define the Backend handoff before considering any
broader question types.
