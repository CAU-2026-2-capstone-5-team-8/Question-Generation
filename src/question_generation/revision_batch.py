"""One bounded semantic repair round; unchanged accepted versions remain immutable."""

import json
import shutil

from google import genai
from google.genai import types

from question_generation.automatic_review import eligible
from question_generation.concept_batch import BATCH_VERSION, generation_identity, save_json
from question_generation.concept_contract import content_hash
from question_generation.concept_generation import file_hash, generate_concept_question
from question_generation.concept_review import load_candidates, prepare_review
from question_generation.schemas import AiQuestionReview


def revise_batch(blueprint, candidates, generation_report, review_report, settings, *, client=None):
    original = json.loads(generation_report.read_text())
    review = json.loads(review_report.read_text())
    if (
        original.get("revisionRound", 0) != 0
        or review["status"] != "REVIEW_BLOCKED"
        or not review["graphApproved"]
    ):
        raise ValueError("automatic revision is not eligible")
    bank, questions = load_candidates(blueprint, candidates)
    judgments = {}
    for line in (review_report.parent / "reviews.jsonl").read_text().splitlines():
        row = AiQuestionReview.model_validate_json(line)
        if row.review.generated_question_id in judgments:
            raise ValueError("duplicate revision review")
        judgments[row.review.generated_question_id] = row.review
    if set(judgments) != {q.generated_question_id for q in questions}:
        raise ValueError("revision requires exact complete review")
    rejected = [q for q in questions if not eligible(judgments[q.generated_question_id])]
    if not 1 <= len(rejected) <= 2:
        raise ValueError("automatic revision limit exceeded")
    identity = {
        "generation": file_hash(generation_report),
        "review": file_hash(review_report),
        "authoring": generation_identity(blueprint, settings),
        "revisionRound": 1,
    }
    output = candidates / "revisions" / content_hash(identity)[7:31]
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "revision-input.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("revision inputs changed")
    save_json(marker, identity)
    bad_ids = {q.generated_question_id for q in rejected}
    specs = {s.question_id: s for s in bank.question_specs}
    own = client is None
    if own:
        client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(
                timeout=35000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
    try:
        for question in questions:
            name = question.question_spec_id + ".json"
            path = output / name
            if question.generated_question_id not in bad_ids:
                if path.exists() and path.read_bytes() != (candidates / name).read_bytes():
                    raise ValueError("accepted revision candidate changed")
                if not path.exists():
                    shutil.copyfile(candidates / name, path)
            elif not path.exists():
                repaired = generate_concept_question(
                    specs[question.question_spec_id],
                    settings,
                    original["blueprintHash"],
                    client=client,
                    raw_path=output / "raw" / name,
                    revision_feedback={
                        "previous_question": question.model_dump(
                            include={"stem", "choices", "correct_choice_index", "explanation"}
                        ),
                        "review": judgments[question.generated_question_id].model_dump(),
                    },
                )
                if repaired.generated_question_id == question.generated_question_id:
                    raise ValueError("revision did not change rejected content")
                save_json(path, repaired.model_dump())
    finally:
        if own:
            client.close()
    load_candidates(blueprint, output)
    if not (output / "human-review").exists():
        prepare_review(blueprint, output, output / "human-review")
    result = {
        "contractVersion": BATCH_VERSION,
        "topicId": bank.topic_id,
        "blueprintHash": original["blueprintHash"],
        "generationIdentityHash": content_hash(identity),
        "status": "CANDIDATES_READY",
        "generatedCount": len(questions),
        "plannedCount": len(questions),
        "contentReview": "pending",
        "diagnosisReady": False,
        "revisionRound": 1,
        "previousGenerationReportHash": identity["generation"],
        "previousReviewReportHash": identity["review"],
        "candidateHashes": {p.name: file_hash(p) for p in sorted(output.glob("q_*.json"))},
        "revisedQuestionSpecIds": sorted(q.question_spec_id for q in rejected),
    }
    for key in ("sourceSnapshotId", "contentReportHash"):
        if key in original:
            result[key] = original[key]
    save_json(output / "generation-report.json", result)
    return output / "generation-report.json"
