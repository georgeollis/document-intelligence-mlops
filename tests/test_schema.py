"""Structural validation of each doc type's state file — no Azure calls,
safe for PR CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "config" / "models"


def test_model_state_files_are_valid_json_and_have_required_keys():
    if not MODELS_DIR.exists():
        return
    for state_path in MODELS_DIR.glob("*.json"):
        state = json.loads(state_path.read_text())
        for key in ("docType", "modelKind", "version", "environments", "thresholds"):
            assert key in state, f"{state_path.name} missing '{key}': {state}"
        assert state["docType"] == state_path.stem, (
            f"{state_path.name}: docType '{state['docType']}' must match the file name"
        )
        assert state["modelKind"] in ("template", "neural", "composed")
        assert len(state["version"].split(".")) == 3, f"version must be semver 'X.Y.Z': {state['version']}"
        assert 0 < state["thresholds"]["minFieldConfidence"] <= 1


def test_no_training_or_validation_data_is_committed_to_git():
    """Both training and validation documents belong only in Dev Blob
    Storage (under the training/ and validation/ prefixes of each doc
    type's container), never in this repo."""
    if not DATASETS_DIR.exists():
        return
    for doc_type_dir in DATASETS_DIR.iterdir():
        if not doc_type_dir.is_dir():
            continue
        for banned in ("train", "golden", "validation"):
            banned_dir = doc_type_dir / banned
            assert not banned_dir.exists(), (
                f"{doc_type_dir.name}: found a '{banned}' folder in the repo - training and validation "
                "data must live in Dev Blob Storage only, not in Git."
            )


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
