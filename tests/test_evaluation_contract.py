from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trace_tfe3.evaluation import (
    apply_trace_clinical,
    fit_trace_clinical,
    select_operating_point,
    threshold_metrics,
)


def _development_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "trace_ct_score": [0.05, 0.12, 0.28, 0.44, 0.47, 0.62, 0.80, 0.92],
            "age": [70, 61, 58, 65, 39, 46, 35, 43],
            "sex": ["M", "M", "F", "M", "F", "M", "F", "F"],
            "automated_maximum_tumour_diameter_cm": [
                2.8,
                3.4,
                4.1,
                3.6,
                4.2,
                5.0,
                3.3,
                4.7,
            ],
        }
    )


class EvaluationContractTests(unittest.TestCase):
    def test_operating_point_respects_sensitivity_floor(self) -> None:
        frame = _development_frame()
        selected = select_operating_point(
            frame["label"],
            frame["trace_ct_score"],
            minimum_sensitivity=0.75,
        )
        metrics = threshold_metrics(
            frame["label"],
            frame["trace_ct_score"],
            selected["threshold"],
        )
        self.assertGreaterEqual(metrics["sensitivity"], 0.75)

    def test_trace_clinical_round_trip(self) -> None:
        frame = _development_frame()
        specification = fit_trace_clinical(frame)
        scores = apply_trace_clinical(frame, specification)
        self.assertEqual(scores.shape, (len(frame),))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
