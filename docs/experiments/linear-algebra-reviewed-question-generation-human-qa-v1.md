# Linear Algebra reviewed QuestionSpec generation human QA v1

## Context

This experiment records final human QA for five live Gemini questions generated from the reviewed
Linear Algebra assessment blueprint. Four questions were revised from preserved first-pass
artifacts through the versioned revision workflow; the vector question retained its first-pass
artifact. Generated provider outputs remain ignored local artifacts. The source-controlled record
contains only `HumanQuestionReview` decisions linked by deterministic `generated_question_id`.

- QuestionSpec source: ML reviewed Linear Algebra AssessmentBlueprint from ML PR #27
- assessment config: `assessment-config-v2-reviewed`
- QuestionSpec contract: `question-spec-v1`
- generation model: `gemini-3.5-flash-lite`
- output language: `ko-KR`
- first-pass prompt: `question-generation-prompt-v2`
- revision prompt: `question-generation-revision-prompt-v1`
- generation config: `gemini-generation-config-v2`
- canonical review record: `reviews/linear-algebra-reviewed-question-generation-v1.jsonl`

## Sample

- total: `5`
- vocabulary / recognize: `4` — matrix, vector, linear system, orthogonality
- background knowledge / recall: `1` — systems of equations

This is a five-item diagnostic sample and must not be reported as a production-accuracy estimate.

## Human QA result

- approve: `5`
- needs_revision: `0`
- reject: `0`
- correct: `5/5`
- difficulty appropriate: `5/5`
- mean concept alignment: `4.8/5`
- mean distractor quality: `4.6/5`
- mean explanation quality: `4.6/5`

| Concept | Generated question ID | Status | Alignment | Distractors | Explanation |
| --- | --- | --- | ---: | ---: | ---: |
| matrix | `gq_ebf68bf0595a5fdb96e8757ca92382d9` | approve | 5 | 5 | 4 |
| vector | `gq_770923a236b225553467b5bddcb89601` | approve | 4 | 5 | 4 |
| linear system | `gq_142923d9ad544ed9f827253ba57df545` | approve | 5 | 4 | 5 |
| orthogonality | `gq_2b08c81f68f1ca6973de873b6e9c30c0` | approve | 5 | 5 | 5 |
| systems of equations | `gq_12d0e100aa9a7715e308bea46394b77f` | approve | 5 | 4 | 5 |

## Revision outcome

Human QA found expression-level problems in four first-pass questions. Repeating ordinary generation
for the same deterministic prompt reproduced the same outputs, so an explicit revision prompt was
needed to incorporate reviewer feedback without changing the authoritative QuestionSpec. The
matrix terminology, linear-system set definition, mathematical definition of orthogonality, and
systems-of-equations prerequisite wording were corrected. The vector question remained unchanged.

The current `GeneratedQuestion` schema intentionally has no `revised_from` field. Revision remains
traceable through the revision prompt version, the new deterministic generated ID, preservation of
the original ignored artifact, and the documented workflow. A canonical lineage field is deferred
until Backend lifecycle requirements justify a separate schema/version change.

## Known rendering limitation

Question-Generation v1 renders every supported target as a four-choice multiple-choice question.
Consequently, `background_knowledge / recall` measures cued recognition more closely than pure free
recall. The systems-of-equations question is valid within the current contract, but its result must
be interpreted with this v1 limitation in mind.
