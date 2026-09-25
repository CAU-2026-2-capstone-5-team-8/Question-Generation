# OS reviewed QuestionSpec generation human QA v1

## Context

This experiment records human QA for five live Gemini questions generated from the reviewed
Operating Systems assessment blueprint. Generated provider outputs remain ignored local artifacts;
the source-controlled record contains only `HumanQuestionReview` decisions linked by deterministic
`generated_question_id`.

- QuestionSpec source: ML reviewed Operating Systems AssessmentBlueprint from ML PR #26
- assessment config: `assessment-config-v2-reviewed`
- QuestionSpec contract: `question-spec-v1`
- generation model: `gemini-3.5-flash-lite`
- output language: `ko-KR`
- prompt: `question-generation-prompt-v2`
- generation config: `gemini-generation-config-v2`
- canonical review record: `reviews/os-reviewed-question-generation-v1.jsonl`

## Sample

- total: `5`
- vocabulary / recognize: `3` — process, thread, scheduling
- background knowledge / recall: `2` — computer architecture, assembly language

This is a deliberately small diagnostic sample. **4/5 approval is a five-item diagnostic sample
and must not be reported as an estimated 80% production accuracy.**

## Human QA result

- approve: `4`
- needs_revision: `1`
- reject: `0`
- correct: `5/5`
- difficulty appropriate: `5/5`
- mean concept alignment: `5.0/5`
- mean distractor quality: `4.2/5`
- mean explanation quality: `4.2/5`

| Concept | Generated question ID | Status | Alignment | Distractors | Explanation |
| --- | --- | --- | ---: | ---: | ---: |
| process | `gq_99d2b731a84d35c35ca56650b372d5f2` | approve | 5 | 4 | 4 |
| thread | `gq_054a4f26f82ec2a0f4842e646ddde0ba` | approve | 5 | 4 | 4 |
| scheduling | `gq_6ee48aae6763b9ef5cb50be016429ea7` | approve | 5 | 5 | 5 |
| computer architecture | `gq_8c8e8256de654c116f90860f7898ce97` | approve | 5 | 4 | 5 |
| assembly language | `gq_7a99e636b775f2f876e9da894dddbef5` | needs_revision | 5 | 4 | 3 |

## Failure attribution

### Assessment-target selection improvement

The earlier legacy OS blueprint selected `programming` as a background-knowledge target. Human
review found that target too general to distinguish OS readiness. The ML assessment review gate
excluded `programming` and selected `computer architecture` and `assembly language` instead. Both
generated background questions are more relevant to OS prerequisite knowledge than the previous
general-programming target.

This is an assessment-target selection improvement. It does not remove `programming` from book
evidence, the recommendation knowledge graph, matching, or ranking.

### Generation-level issue

The `assembly language` QuestionSpec target is appropriate and received concept alignment 5/5.
The generated wording is the problem: it describes assembly language as having an invariably
strict one-to-one correspondence with machine-code instructions and overstates direct processor
control. Pseudo-instructions and architecture-specific mappings make the absolute wording unsafe.
This is attributed to generation, not to ML QuestionSpec selection.

### Approved without blocking failures

The process, thread, scheduling, and computer-architecture questions were approved. The thread
wording uses the somewhat unnatural phrase `CPU utilization`; that is a possible later wording
improvement, not a blocking correctness failure in this review.

## Decision on generator changes

No prompt, generation configuration, model, or validator rule changed in response to this sample.
One expression-level failure in five questions is not enough evidence for a general prompt policy,
and an assembly-specific blacklist would overfit the generator to one observed output. A larger
15–20 QuestionSpec human-QA sample should establish whether the pattern repeats before tuning.
