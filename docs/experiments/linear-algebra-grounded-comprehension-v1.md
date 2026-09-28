# Linear Algebra grounded comprehension v1

## Input provenance

- QuestionSpec: `q_375e5b6bef551015f67c`
- target: matrix, comprehension/apply, Level 2
- source document: `doc_de934d33d551223812e8`, Hefferon sample chapter
- document hash: `sha256:82e9c6f9fadf91f241cbe56fe5e3aa694cd74c94fe6eba9beade2b78be032225`
- license: GNU Free Documentation License or CC BY-SA 3.0 US
- passage length: 617 characters
- passage hash: `sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0`
- grounding artifact hash: `sha256:d621639d33fe6fd0615e259439e53371303363ba11bd5c9767ed6b2a6a1b841c`
- prompt: `question-generation-grounded-prompt-v3`

The passage is an exact contiguous source substring defining matrix dimensions and entry positions.
It is displayed to the user unchanged; the full 140,910-character source document is not sent to
the provider. The local generated artifact remains ignored under `data/generated/`.

## Live generation

One Gemini call produced `gq_4783115ebe51684dd059a1724b42df8b` with
`generated-question-v3`. Structured parsing, authoritative-target validation, passage embedding,
and final Pydantic validation passed. The provider reported 1,647 input, 357 output, and 2,004 total
tokens.

The question asks the reader to identify the array that preserves the passage's 2-row, 3-column
matrix and entry ordering. It has four choices, correct index 2 (zero-based), and a Korean
passage-grounded explanation. Passing validation is not human approval.

## Human QA result

The first question received `needs_revision` in
`reviews/linear-algebra-grounded-comprehension-v1.jsonl`:

- correct: `true`
- concept alignment: `5/5`
- difficulty appropriate: `false`
- distractor quality: `4/5`
- explanation quality: `5/5`

The correct answer and explanation are passage-grounded, but the answer reproduces the passage's
example matrix. A reader can solve it by visual comparison instead of applying the row/column
convention to a new situation. The original generated artifact remains preserved.

## Grounded revision

The smallest follow-up reuses the existing revision command, provider adapter, structured provider
schema, semantic output validation, deterministic ID, and atomic output writer. The grounded path
adds `question-generation-grounded-revision-prompt-v1` and requires the authoritative blueprint,
the same `generation-grounding-v1` artifact, and the preserved original v3 question. It revalidates
the QuestionSpec, canonical hashes, grounding hash, source document, and exact passage before any
provider call. The replacement is written to a new ignored output file.

The revision prompt requires applying a passage rule to a new situation and explicitly rejects
copying an example or matching an identical string. It does not change the v2 revision prompt or
relax the existing ungrounded Level 1 boundary.

## Passage presentation limitation

The exact passage contains PDF extraction artifacts such as missing spaces and character joins.
This revision deliberately preserves the source substring and passage hash. A future milestone may
evaluate a provenance-preserving display-normalization layer, separate raw and display passages, a
cleaner HTML canonical source, or deterministic text normalization. None is implemented here.

## Remaining boundary

`integrate`/Level 3, multi-source grounding, and Backend import are unsupported.
Backend's current strict v2 handoff rejects this v3 artifact until a separate reviewed contract
milestone explicitly adds comprehension support.
