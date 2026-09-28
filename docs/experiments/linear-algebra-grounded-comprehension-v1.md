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

## Human QA worksheet

- [ ] passage만으로 정답 결정 가능
- [ ] 외부 지식이 필수적이지 않음
- [ ] target concept을 실제로 평가함
- [ ] 단순 문자열 검색만으로 풀리는 문제가 아님
- [ ] 정답이 passage 내용과 일치
- [ ] distractor도 passage 맥락에서 그럴듯함
- [ ] passage와 문제 사이에 의미적 연결이 있음
- [ ] difficulty가 QuestionSpec과 맞음
- [ ] 한국어가 자연스러움

No `approve`, `reject`, or `needs_revision` decision has been made, and no
`HumanQuestionReview` JSONL row has been written.

## Remaining boundary

`integrate`/Level 3, multi-source grounding, grounded revision, and Backend import are unsupported.
Backend's current strict v2 handoff rejects this v3 artifact until a separate reviewed contract
milestone explicitly adds comprehension support.
