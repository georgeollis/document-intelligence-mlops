"""
STEP 1 of the pipeline: train (Build Model) a custom model in Dev.

Dev is the ONLY environment that ever calls Build Model. Test and Prod never
train - they only ever receive a copy of a model that already passed
evaluation in Dev (see copy_model.py).

Reads training data directly from Blob Storage: the Dev Document
Intelligence resource authenticates to the training container using its own
system-assigned managed identity, which must be granted the "Storage Blob
Data Reader" role on that container:

    az cognitiveservices account identity assign -n <dev-di-resource> -g <rg>
    az role assignment create \
      --assignee <the-managed-identity-principal-id-from-above> \
      --role "Storage Blob Data Reader" \
      --scope <training-container-resource-id>

Every doc type's container is organized under two prefixes, created once
when its Document Intelligence Studio project is set up:

    <container>/training/    <- labeled documents (this script reads this prefix)
    <container>/validation/  <- held-out documents used later by evaluate.py

The container itself is resolved per doc type (see
common.resolve_training_container_url) so different models can use
different containers without any code changes - pass --training-container-url
to override it explicitly for a given run.

Every training run bumps the model's semantic version (default: minor) and
model_id becomes "<doc-type>-v<version>", e.g. "invoice-v1.2.0" - so every
model built in Document Intelligence is traceable back to a specific,
ordered version. Past runs are never overwritten: each one is appended to
the model's "history" in its state file (config/models/<doc-type>.json),
and "environments.dev" always points
at the latest.

Example:
    python src/train.py --doc-type invoice
    python src/train.py --doc-type receipt --training-container-url "https://otheraccount.blob.core.windows.net/receipt-train"
    python src/train.py --doc-type invoice --version-bump major
"""
from __future__ import annotations

import argparse
import sys

from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient
from azure.ai.documentintelligence.models import AzureBlobContentSource, BuildDocumentModelRequest

from common import (
    TRAINING_PREFIX,
    bump_version,
    get_environment_config,
    get_or_create_model_state,
    model_state_path,
    record_training_run,
    resolve_training_container_url,
    save_model_state,
    update_environment_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-type", required=True, help="e.g. invoice, receipt, lb200 - no default")
    parser.add_argument("--model-kind", choices=["template", "neural"], default="neural")
    parser.add_argument(
        "--training-container-url",
        default=None,
        help="Blob container URL for this doc type's Dev data. Defaults to the container "
        "resolved from DEV_TRAINING_CONTAINER_URL_<DOC_TYPE> or DEV_TRAINING_STORAGE_ACCOUNT.",
    )
    parser.add_argument(
        "--training-prefix",
        default=TRAINING_PREFIX,
        help=f"Blob name prefix (subfolder) within the container holding labeled training documents. "
        f"Default: '{TRAINING_PREFIX}'.",
    )
    parser.add_argument(
        "--version-bump",
        choices=["major", "minor", "patch"],
        default="minor",
        help="Which part of the model's semantic version to bump for this training run. Default: minor.",
    )
    parser.add_argument("--model-id", default=None, help="Defaults to '<doc-type>-v<version>'")
    args = parser.parse_args()

    training_container_url = args.training_container_url or resolve_training_container_url(args.doc_type)

    # 1. Look up (or create) this doc type's state file and bump its
    #    semantic version - the new version becomes part of the model_id so
    #    every Document Intelligence model is traceable to a state file entry.
    state = get_or_create_model_state(args.doc_type, args.model_kind)
    new_version = bump_version(state["version"], args.version_bump)
    model_id = args.model_id or f"{args.doc_type}-v{new_version}"

    # 2. Connect to the Dev Document Intelligence resource only.
    dev = get_environment_config("dev")
    client = DocumentIntelligenceAdministrationClient(dev.endpoint, dev.credential)

    # 3. Kick off training. build_mode "neural" or "template" controls model type.
    print(f"Training '{args.model_kind}' model '{model_id}' (version {new_version}) for doc type '{args.doc_type}' in Dev...")
    print(f"Reading training data from '{training_container_url}'" + (f" (prefix '{args.training_prefix}')" if args.training_prefix else ""))
    poller = client.begin_build_document_model(
        BuildDocumentModelRequest(
            model_id=model_id,
            build_mode=args.model_kind,
            azure_blob_source=AzureBlobContentSource(container_url=training_container_url, prefix=args.training_prefix),
            description=f"{args.doc_type} {args.model_kind} model v{new_version}",
        )
    )
    result = poller.result()
    print(f"Trained model '{result.model_id}'.")

    # 4. Record the new version in this doc type's state file: bump the
    #    version, append to history (past runs are never overwritten), and
    #    point Dev at the newly trained model.
    state["version"] = new_version
    record_training_run(
        state,
        version=new_version,
        model_id=result.model_id,
        model_kind=args.model_kind,
        trained_at=str(result.created_date_time),
    )
    update_environment_entry(
        state, "dev",
        modelId=result.model_id,
        modelVersion=new_version,
        resourceEndpoint=dev.endpoint,
        trainedAt=str(result.created_date_time),
    )
    save_model_state(args.doc_type, state)
    print(f"State file updated: {model_state_path(args.doc_type)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
