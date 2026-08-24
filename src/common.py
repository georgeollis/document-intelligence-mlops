"""Shared helpers for authenticating to Azure AI Document Intelligence
resources and locating repo-relative paths.

Every environment authenticates with `DefaultAzureCredential` - no resource
keys anywhere in this sample. In practice that resolves to:
  - a **user/system-assigned managed identity**, when running on Azure
    compute (e.g. an Azure DevOps/GitHub-hosted runner with a federated
    identity, or an Azure VM/Container App running the pipeline), or
  - a **service principal (SPN)** via federated OIDC credentials
    (`AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_SUBSCRIPTION_ID`), the
    recommended setup for GitHub Actions - see `.github/workflows/model-pipeline.yml`.

This sample ships with three environments - dev, test, prod - but any
environment name works: define `<NAME>_DI_ENDPOINT` and pass `--env <name>`
/ `--source <name>` / `--target <name>` to the scripts. For example, adding
a `staging` environment between test and prod needs no code changes, just a
`STAGING_DI_ENDPOINT` secret, the identity granted access to it, and an
extra `copy_model.py` step in the pipeline.

Each doc type gets its own state file (config/models/<doc-type>.json) and
its own reports folder (reports/<doc-type>/) - see model_state_path() and
ensure_reports_dir() - so a repo with many doc types never has one huge
shared file or folder that keeps growing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "config" / "models"
REPORTS_DIR = REPO_ROOT / "reports"

# Every doc type's Dev container is organized under two prefixes:
#   training/    - labeled documents Build Model trains on (train.py)
#   validation/  - held-out documents used to evaluate a trained model
#                  (evaluate.py). Neither prefix is ever copied into this
#                  repo - both live only in Blob Storage, created once when
#                  the Document Intelligence Studio project for that doc
#                  type is set up.
TRAINING_PREFIX = "training/"
VALIDATION_PREFIX = "validation/"

# How many past evaluation runs to keep per environment in each doc type's
# state file (oldest dropped first) so it stays small no matter how often
# evaluate.py runs. Full detail for every run always lives in reports/<doc-type>/.
MAX_EVALUATIONS_PER_ENVIRONMENT = 20


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    endpoint: str
    credential: TokenCredential


def get_environment_config(env: str) -> EnvironmentConfig:
    """Resolve endpoint + credential for any environment name from the
    `<ENV>_DI_ENDPOINT` env var. Credential is always DefaultAzureCredential
    (managed identity or federated SPN - see module docstring)."""
    env = env.lower()
    prefix = env.upper()
    endpoint = os.environ.get(f"{prefix}_DI_ENDPOINT")
    if not endpoint:
        raise EnvironmentError(f"Missing required env var {prefix}_DI_ENDPOINT")

    return EnvironmentConfig(name=env, endpoint=endpoint, credential=DefaultAzureCredential())


def resolve_training_container_url(doc_type: str) -> str:
    """Resolve the Dev container URL for a given doc type.

    Every doc type's Dev container follows the same layout:

        <container>/training/    -> labeled documents used to Build Model
        <container>/validation/  -> held-out documents used to evaluate a trained model

    The container itself is resolved in this order:
      1. `DEV_TRAINING_CONTAINER_URL_<DOC_TYPE>` - an explicit override,
         for a doc type whose container lives somewhere non-standard.
      2. `<DEV_TRAINING_STORAGE_ACCOUNT>/<doc-type>` - the default naming
         convention, one container per doc type in a shared Dev storage
         account.
    """
    override = os.environ.get(f"DEV_TRAINING_CONTAINER_URL_{doc_type.upper()}")
    if override:
        return override

    account = os.environ.get("DEV_TRAINING_STORAGE_ACCOUNT")
    if account:
        return f"https://{account}.blob.core.windows.net/{doc_type}"

    raise EnvironmentError(
        f"Set DEV_TRAINING_CONTAINER_URL_{doc_type.upper()} or DEV_TRAINING_STORAGE_ACCOUNT "
        f"to resolve the container for doc type '{doc_type}'."
    )


def iter_validation_documents(container_url: str, credential: TokenCredential):
    """Yield (blob_name, bytes) for every document under VALIDATION_PREFIX.
    Read-only - never writes."""
    container = ContainerClient.from_container_url(container_url, credential=credential)
    for blob in container.list_blobs(name_starts_with=VALIDATION_PREFIX):
        name = blob.name[len(VALIDATION_PREFIX):]
        if not name:
            continue
        yield name, container.download_blob(blob.name).readall()


def model_state_path(doc_type: str) -> Path:
    """Each doc type gets its own state file - config/models/<doc-type>.json -
    instead of one shared config/model-manifest.json. This keeps every doc
    type's history/evaluations bounded and independent: a repo with 50 doc
    types never makes any single file bigger than a repo with 1."""
    return MODELS_DIR / f"{doc_type}.json"


def ensure_reports_dir(doc_type: str) -> Path:
    """reports/<doc-type>/ - namespaced per doc type for the same reason:
    unbounded growth in one doc type's reports never affects another's."""
    directory = REPORTS_DIR / doc_type
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_model_state(doc_type: str) -> dict:
    path = model_state_path(doc_type)
    if not path.exists():
        raise FileNotFoundError(
            f"No state file for doc type '{doc_type}' at {path}. "
            f"Run train.py first, or copy config/models/_template.json to create one."
        )
    return json.loads(path.read_text())


def save_model_state(doc_type: str, state: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_state_path(doc_type).write_text(json.dumps(state, indent=2))


def get_or_create_model_state(doc_type: str, model_kind: str) -> dict:
    """One shared helper used by train.py / copy_model.py / evaluate.py so
    the per-doc-type state bookkeeping logic lives in exactly one place.
    """
    path = model_state_path(doc_type)
    if path.exists():
        state = json.loads(path.read_text())
    else:
        state = {
            "docType": doc_type,
            "modelKind": model_kind,
            "version": "0.0.0",
            "childModelIds": [],
            "environments": {},
            "history": [],
            "evaluations": {},
            "thresholds": {"minFieldConfidence": 0.80},
        }
    state["modelKind"] = model_kind
    state.setdefault("version", "0.0.0")
    state.setdefault("history", [])
    state.setdefault("evaluations", {})
    return state


def bump_version(current: str, part: str = "minor") -> str:
    """Semantic version bump (major/minor/patch) on a plain 'X.Y.Z' string.
    Every Dev training run bumps the version - this becomes part of the
    model_id (e.g. 'invoice-v1.1.0') so every model built in Document
    Intelligence is traceable back to a specific version in its state file."""
    major, minor, patch = (int(p) for p in current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown version part '{part}' - use major, minor, or patch")


def record_training_run(state: dict, *, version: str, model_id: str, model_kind: str, trained_at: str) -> None:
    """Append (never overwrite) a row to the model's training history."""
    state.setdefault("history", []).append(
        {"version": version, "modelId": model_id, "modelKind": model_kind, "trainedAt": trained_at}
    )


def record_evaluation_run(
    state: dict, *, env_name: str, model_id: str, model_version: str | None,
    document_count: int, average_confidence: float, report_path: str,
) -> None:
    """Append a small pointer (never the full per-field results) to this
    model's evaluation history for the given environment, keyed by model
    version. This is a record of what was measured - it does not decide
    whether to promote; that's left to a developer or team reviewing the
    full report (see report_path) and the documents themselves.

    Keeps only the most recent MAX_EVALUATIONS_PER_ENVIRONMENT runs per
    environment so this doc type's state file stays small; full history for
    every run always exists in reports/<doc-type>/ (or wherever CI archives
    that artifact).
    """
    runs = state.setdefault("evaluations", {}).setdefault(env_name, [])
    runs.append(
        {
            "modelId": model_id,
            "modelVersion": model_version,
            "documentCount": document_count,
            "averageConfidence": average_confidence,
            "reportPath": report_path,
        }
    )
    del runs[:-MAX_EVALUATIONS_PER_ENVIRONMENT]


def update_environment_entry(state: dict, env_name: str, **fields) -> None:
    """Merge fields (modelId, modelVersion, resourceEndpoint, trainedAt,
    copiedFrom, ...) into state["environments"][env_name]."""
    state.setdefault("environments", {}).setdefault(env_name, {})
    state["environments"][env_name].update(fields)
