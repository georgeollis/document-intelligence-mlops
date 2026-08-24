"""Summarize evaluation reports (produced by src/evaluate.py) for a quick
console overview of extraction confidence per document.

There is no pass/fail here: evaluate.py doesn't compare against
known-correct values, it just records what the model extracted and how
confident it was. Deciding whether a model version is accurate enough to
promote is a human judgment call - reviewing this summary, the full report,
and the source documents together (see reports/<doc-type>/*.json and copy_model.py).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "config" / "models"


def _reference_confidence_for(doc_type: str) -> float:
    state = json.loads((MODELS_DIR / f"{doc_type}.json").read_text())
    return state["thresholds"]["minFieldConfidence"]


def test_summarize_available_reports():
    if not REPORTS_DIR.exists():
        print("No reports/ directory yet - run evaluate.py first. Skipping.")
        return

    for report_path in sorted(REPORTS_DIR.glob("*/*-eval-*.json")):
        report = json.loads(report_path.read_text())
        reference = _reference_confidence_for(report["docType"])
        confidence = report["averageConfidence"]
        flag = "OK" if confidence >= reference else "BELOW REFERENCE CONFIDENCE"
        print(
            f"{report_path.relative_to(REPORTS_DIR)}: model={report['modelId']} version={report.get('modelVersion')} "
            f"env={report['environment']} documents={len(report['documents'])} "
            f"avgConfidence={confidence:.2%} (reference {reference:.2%}) [{flag}]"
        )


if __name__ == "__main__":
    test_summarize_available_reports()
