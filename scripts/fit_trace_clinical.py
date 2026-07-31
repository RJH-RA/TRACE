#!/usr/bin/env python
"""Fit and freeze TRACE-Clinical using development-cohort TRACE-CT scores."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_tfe3.evaluation.clinical import fit_trace_clinical, save_clinical_specification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--regularization-c", type=float, default=1.0)
    args = parser.parse_args()

    source = Path(args.predictions_csv)
    frame = pd.read_csv(source)
    if "split" in frame:
        frame = frame[frame["split"].astype(str) == args.split].copy()
    specification = fit_trace_clinical(frame, regularization_c=args.regularization_c)
    specification.update(
        {
            "training_split": args.split,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    )
    save_clinical_specification(specification, args.output_json)


if __name__ == "__main__":
    main()
