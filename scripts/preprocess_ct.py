#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare registered tumour-centred TRACE CT")
    parser.add_argument("--cases-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import pandas as pd

    from trace_tfe3.preprocessing import CTPreprocessConfig, preprocess_ct_case
    from trace_tfe3.utils.config import load_config

    cfg = load_config(args.config)
    prep = cfg["preprocessing"]
    config = CTPreprocessConfig(
        target_spacing_mm=tuple(prep["target_spacing_mm"]),
        roi_shape_voxels=tuple(prep["roi_shape_voxels"]),
        hu_window=tuple(prep["hu_window"]),
    )
    cases = pd.read_csv(args.cases_csv)
    required = {"patient_id", "mask_path", "noncontrast", "arterial"}
    if missing := required.difference(cases.columns):
        raise ValueError(f"cases CSV is missing columns: {sorted(missing)}")

    records = [
        preprocess_ct_case(
            patient_id=str(row.patient_id),
            noncontrast_path=row.noncontrast,
            arterial_path=row.arterial,
            mask_path=row.mask_path,
            output_dir=args.output_dir,
            config=config,
        )
        for row in cases.itertuples(index=False)
    ]
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(destination, index=False)


if __name__ == "__main__":
    main()
