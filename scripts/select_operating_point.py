#!/usr/bin/env python
"""Lock a training-derived TRACE operating point for unchanged test use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_tfe3.evaluation import select_operating_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--minimum-sensitivity", type=float, default=0.80)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="trace_ct_score")
    args = parser.parse_args()

    source = Path(args.predictions_csv)
    frame = pd.read_csv(source)
    if "split" in frame:
        frame = frame[frame["split"].astype(str) == args.split]
    selected = select_operating_point(
        frame[args.label_col],
        frame[args.score_col],
        minimum_sensitivity=args.minimum_sensitivity,
    )
    record = {
        "threshold": selected["threshold"],
        "selection_split": args.split,
        "rule": "highest specificity with sensitivity at least the prespecified floor",
        "minimum_sensitivity": args.minimum_sensitivity,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n": int(len(frame)),
        "events": int(frame[args.label_col].sum()),
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
