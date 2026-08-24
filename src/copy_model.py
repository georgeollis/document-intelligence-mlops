"""
STEP 2 of the pipeline: promote a model by COPYING it to another
environment (never retraining). This sample uses dev/test/prod, but any
environment name works - see common.py.

Example:
    python src/copy_model.py --model-id invoice-v1.2.0 --source dev --target test
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient

from common import get_environment_config, load_model_state, model_state_path, save_model_state, update_environment_entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-type", required=True, help="e.g. invoice, receipt, lb200 - no default")
    parser.add_argument("--model-id", required=True, help="The exact model id trained in Dev (see train.py output)")
    parser.add_argument("--source", required=True, help="Environment to copy from, e.g. dev, test, staging")
    parser.add_argument("--target", required=True, help="Environment to copy to, e.g. test, staging, prod")
    args = parser.parse_args()

    if args.source == args.target:
        raise SystemExit("--source and --target must differ")

    source = get_environment_config(args.source)
    target = get_environment_config(args.target)
    source_client = DocumentIntelligenceAdministrationClient(source.endpoint, source.credential)
    target_client = DocumentIntelligenceAdministrationClient(target.endpoint, target.credential)

    # 1. Target issues a one-time authorization for the copy.
    print(f"Requesting copy authorization from '{args.target}'...")
    authorization = target_client.authorize_model_copy(
        {"modelId": args.model_id, "description": f"Copy from {args.source} to {args.target}"}
    )

    # 2. Source performs the copy using that authorization. The resulting
    #    model is byte-for-byte the same model that was trained/evaluated
    #    in the source environment - nothing is retrained.
    print(f"Copying model '{args.model_id}' from '{args.source}' to '{args.target}'...")
    poller = source_client.begin_copy_model_to(args.model_id, authorization)
    result = poller.result()
    print(f"Copy complete. Model '{result.model_id}' now available in '{args.target}'.")

    # 3. Record the copy in this doc type's state file for traceability,
    #    carrying forward the semantic version this model_id corresponds to
    #    (from history).
    state = load_model_state(args.doc_type)
    history_match = next((h for h in state.get("history", []) if h["modelId"] == args.model_id), None)
    update_environment_entry(
        state, args.target,
        modelId=result.model_id,
        modelVersion=history_match["version"] if history_match else None,
        resourceEndpoint=target.endpoint,
        copiedFrom=args.source,
        copiedAt=datetime.now(timezone.utc).isoformat(),
    )
    save_model_state(args.doc_type, state)
    print(f"State file updated: {model_state_path(args.doc_type)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
