"""
STEP 3 of the pipeline: evaluate a model in any environment by running it
against the "validation" documents it has never seen, and recording what it
extracted - field values and confidence scores - for a human to review.

Validation documents live in the same Dev container as training data, under
a separate prefix - never copied into this repo, never used for training:

    <container>/validation/   <- held-out documents

This never trains anything, and it doesn't score "correct vs incorrect"
itself - there's no expected-values file to compare against. It just runs
Analyze Document over every validation document and writes a report of the
extracted fields and their confidence. A developer or team reads that
report (and/or opens the documents next to it) to judge whether a model
version is accurate enough to promote (see copy_model.py) - that judgment
call is intentionally left out of this script.

Each doc type's state file (config/models/<doc-type>.json) only stores a
small pointer to each evaluation run (env, model id/version, report path,
document count, average confidence) - never the full per-field results - so
it stays small no matter how many documents or runs accumulate. Full detail
lives in the (gitignored) reports/<doc-type>/ folder.

Example:
    python src/evaluate.py --env test --model-id invoice-v1.2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from azure.ai.documentintelligence import DocumentIntelligenceClient

from common import (
    ensure_reports_dir,
    get_environment_config,
    iter_validation_documents,
    load_model_state,
    record_evaluation_run,
    resolve_training_container_url,
    save_model_state,
)


def analyze_document(client: DocumentIntelligenceClient, model_id: str, content: bytes) -> dict:
    """Call Analyze Document and return {field_name: {value, confidence}}."""
    poller = client.begin_analyze_document(model_id, body=content, content_type="application/octet-stream")
    result = poller.result()

    fields = {}
    if result.documents:
        for name, field in result.documents[0].fields.items():
            fields[name] = {"value": field.content, "confidence": field.confidence}
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-type", required=True, help="e.g. invoice, receipt, lb200 - no default")
    parser.add_argument("--env", required=True, help="Environment to evaluate in, e.g. dev, test, staging, prod")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--validation-container-url",
        default=None,
        help="Container holding validation/ (defaults to the same container training reads from).",
    )
    args = parser.parse_args()

    state = load_model_state(args.doc_type)
    model_version = next(
        (h["version"] for h in state.get("history", []) if h["modelId"] == args.model_id), None
    )

    # 1. Connect to whichever environment we're evaluating.
    env = get_environment_config(args.env)
    client = DocumentIntelligenceClient(env.endpoint, env.credential)

    # 2. Read validation/ straight from Dev Blob Storage - the same
    #    identity that reads training/ also reads validation/, and neither
    #    is ever copied into this repo.
    container_url = args.validation_container_url or resolve_training_container_url(args.doc_type)
    dev = get_environment_config("dev")

    document_results = []
    for file_name, content in iter_validation_documents(container_url, dev.credential):
        fields = analyze_document(client, args.model_id, content)
        confidences = [f["confidence"] for f in fields.values() if f["confidence"] is not None]
        document_results.append({"file": file_name, "fields": fields})
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"  {file_name}: {len(fields)} field(s) extracted, avg confidence {avg:.0%}")

    if not document_results:
        raise SystemExit(f"No validation documents found under '{container_url}/validation/'.")

    all_confidences = [
        f["confidence"] for d in document_results for f in d["fields"].values() if f["confidence"] is not None
    ]
    average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # 3. Write the full extracted results to a report for humans to review,
    #    and record only a small pointer to it in this doc type's state
    #    file - not the full per-field detail, so it stays small however
    #    many documents/runs accumulate over time.
    report = {
        "docType": args.doc_type,
        "environment": args.env,
        "modelId": args.model_id,
        "modelVersion": model_version,
        "averageConfidence": average_confidence,
        "documents": document_results,
    }
    report_path = ensure_reports_dir(args.doc_type) / (
        f"{args.env}-{args.model_id}-eval-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2))

    record_evaluation_run(
        state,
        env_name=args.env,
        model_id=args.model_id,
        model_version=model_version,
        document_count=len(document_results),
        average_confidence=average_confidence,
        report_path=str(report_path.relative_to(report_path.parents[1])),
    )
    save_model_state(args.doc_type, state)

    print(f"\n{len(document_results)} document(s) evaluated, average confidence {average_confidence:.0%}.")
    print(f"Full extracted values written to: {report_path}")
    print(
        "Review the report (and the documents themselves) to judge accuracy - "
        "whether this version is ready to promote is a team decision, see copy_model.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
