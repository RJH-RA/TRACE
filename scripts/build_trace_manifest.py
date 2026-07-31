#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patient-level TRACE manifest")
    parser.add_argument("--ct-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--slide-embedding-csv", default=None)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    from trace_tfe3.preprocessing import build_patient_manifest

    build_patient_manifest(
        ct_csv=args.ct_csv,
        labels_csv=args.labels_csv,
        slide_embedding_csv=args.slide_embedding_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
