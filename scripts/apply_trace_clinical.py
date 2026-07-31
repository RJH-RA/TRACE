#!/usr/bin/env python
"""Apply a frozen TRACE-Clinical specification without refitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_tfe3.evaluation.clinical import (
    apply_trace_clinical,
    load_clinical_specification,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions_csv)
    specification = load_clinical_specification(args.model_json)
    frame["trace_clinical_score"] = apply_trace_clinical(frame, specification)
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


if __name__ == "__main__":
    main()
