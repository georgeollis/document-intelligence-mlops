"""
Promotes every model that is newer upstream than it is here.

This is designed to run on a `test/*` branch (comparing against a `dev/*`
branch) or a `prod/*` branch (comparing against a `test/*` branch) right
after a merge: for each doc type, if the upstream branch's state file shows
a newer `modelId` in its own environment than this branch's state file
shows in ITS environment, copy that model here and evaluate it.

"Upstream" and "here" are both just a directory of `config/models/*.json`
state files at a given git ref - this script diffs one folder's worth of
those files, it never talks to two branches' worth of Document
Intelligence resources except the ones it's actually promoting between.

Example (run on a `test/*` branch, after merging a `dev/*` branch into it):
    mkdir -p /tmp/upstream-models
    git show dev/main:config/models > /tmp/upstream-models-list.txt   # or use git archive
    python src/promote_pending_models.py \
      --upstream-models-dir /tmp/upstream-models \
      --upstream-env dev --this-env test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import load_model_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-models-dir", required=True,
        help="Path to a checkout of config/models/ from the upstream branch (e.g. dev/*, or test/*).",
    )
    parser.add_argument("--upstream-env", required=True, help="Environment the upstream branch trains/promotes into, e.g. dev, test")
    parser.add_argument("--this-env", required=True, help="Environment this branch promotes into, e.g. test, prod")
    args = parser.parse_args()

    upstream_dir = Path(args.upstream_models_dir)
    upstream_files = sorted(upstream_dir.glob("*.json")) if upstream_dir.exists() else []

    promoted = []
    for upstream_file in upstream_files:
        doc_type = upstream_file.stem
        upstream_state = json.loads(upstream_file.read_text())
        upstream_env = upstream_state.get("environments", {}).get(args.upstream_env) or {}
        upstream_model_id = upstream_env.get("modelId")
        upstream_version = upstream_env.get("modelVersion")
        if not upstream_model_id:
            continue  # nothing trained/promoted upstream yet for this doc type

        try:
            this_state = load_model_state(doc_type)
        except FileNotFoundError:
            this_state = None
        this_env = (this_state or {}).get("environments", {}).get(args.this_env) or {}
        if this_env.get("modelId") == upstream_model_id:
            print(f"'{doc_type}': already at '{upstream_model_id}' in '{args.this_env}', skipping.")
            continue

        print(f"'{doc_type}': promoting '{upstream_model_id}' (v{upstream_version}) "
              f"from '{args.upstream_env}' to '{args.this_env}'...")
        subprocess.run(
            [
                sys.executable, "src/copy_model.py",
                "--doc-type", doc_type, "--model-id", upstream_model_id,
                "--source", args.upstream_env, "--target", args.this_env,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "src/evaluate.py",
                "--doc-type", doc_type, "--env", args.this_env, "--model-id", upstream_model_id,
            ],
            check=True,
        )
        promoted.append(doc_type)

    if not promoted:
        print(f"Nothing to promote into '{args.this_env}'.")
    else:
        print(f"\nPromoted into '{args.this_env}': {', '.join(promoted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
