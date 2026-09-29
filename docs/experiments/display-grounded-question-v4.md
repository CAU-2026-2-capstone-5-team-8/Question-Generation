# Display-grounded question v4

## Dependency and scope

This contract consumes ML `generation-grounding-v2` from ML PR #29. It is additive: existing
`generated-question-v2`, `generated-question-v3`, and grounded v3 revision behavior are unchanged.
No live provider call was made for this milestone.

The supported target remains deliberately narrow:

```text
single source / comprehension / apply / Level 2
```

`integrate`, Level 3, multi-source grounding, v4 revision, and Backend import remain unsupported.

## Input validation

Question-Generation keeps a strict local copy of the ML v2 contract. It validates:

- exact QuestionSpec and blueprint binding;
- all four canonical dataset hashes;
- source document/book/evidence identity;
- `source_passage_hash` against `source_passage_text`;
- the reviewed, source-hash-bound `pdf-display-normalization-v1` transformation;
- `display_passage_hash` against `display_passage_text`.

The local policy copy is intentionally fail closed. It does not accept an unknown document/hash,
an unsupported policy, a changed source fragment, or a display passage that differs from the
deterministic ML result.

## Provider passage choice

`question-generation-grounded-prompt-v4` sends `display_passage_text`, not the raw extraction, as
the provider's only factual passage. This is safe for the current slice because every change is an
exact reviewed spacing, ligature, or PDF line-wrap repair; both forms and both hashes remain bound
in the input artifact. The prompt includes the source passage hash, display passage hash, policy,
document hash, and extraction policy, but omits raw source text and source URL.

Using the display passage for both provider grounding and user presentation avoids asking the model
to reason over the same broken word joins that the user no longer sees. It also leaves one visible,
hash-checked factual representation instead of maintaining two divergent prompt/display paths.

## `generated-question-v4`

The v3 format combines the passage and provider question inside one `stem` string. V4 separates
them:

```text
passage  = deterministic display passage
stem     = provider's question sentence only
choices  = exactly four choices
```

The output also retains:

- `grounding_version`;
- `source_passage_extraction_policy`;
- `source_passage_hash`;
- `display_passage_hash`;
- `display_normalization_policy`;
- one `source_document_id` and the exact grounding artifact hash.

The deterministic question ID covers the separate passage, question output, target identity, input
artifact hash, and display provenance. Provider token usage remains excluded, matching the existing
ID policy. The output schema re-hashes `passage`, recomputes the v4 deterministic ID, and rejects
either mismatch. A future Backend reader must still compare the provenance fields with the exact
grounding artifact rather than treating the output schema alone as proof of external source
lineage.

## Actual offline dry run

The production CLI loaded the real reviewed blueprint and local ML v2 artifact, validated the full
binding, and rendered the v4 prompt with `--dry-run`. No API key was required and no Gemini request
was made.

- QuestionSpec: `q_375e5b6bef551015f67c`
- grounding artifact hash: `sha256:a9d7f5f1b1487040c6aff681fbfe8591800a274aa0b5e2231cf784ea8c1fe2a6`
- source passage hash: `sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0`
- display passage hash: `sha256:c5126e3650a7c2329f2a4281efc774465bb9954e19091808d7162938926850a4`
- policy: `pdf-display-normalization-v1`
- prompt: `question-generation-grounded-prompt-v4`

Repeated fake-provider finalization with identical structured output produced the same complete v4
object and deterministic ID. Existing v2/v3 generation and v3 revision tests continue to exercise
their historical paths.

## Preserved experiment history and Backend boundary

The original `gq_4783115ebe51684dd059a1724b42df8b` artifact and its `needs_revision` human review
remain unchanged. V4 does not reinterpret or replace that result, and this milestone does not retry
the failed live revision.

Backend still accepts only `generated-question-v2`; it must continue rejecting v3 and v4 until a
separate reviewed integration defines storage and presentation for the explicit `passage` field.
