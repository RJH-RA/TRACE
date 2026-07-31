from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURES = (
    "trace_ct_score",
    "age",
    "sex_female",
    "automated_maximum_tumour_diameter_cm",
)


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "trace_ct_score",
        "age",
        "sex",
        "automated_maximum_tumour_diameter_cm",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"TRACE-Clinical input is missing columns: {sorted(missing)}")
    sex = frame["sex"].astype(str).str.strip().str.upper()
    invalid = ~sex.isin(["F", "M", "FEMALE", "MALE"])
    if invalid.any():
        raise ValueError("sex must be encoded as F/M or female/male")
    return np.column_stack(
        [
            frame["trace_ct_score"].astype(float),
            frame["age"].astype(float),
            sex.isin(["F", "FEMALE"]).astype(float),
            frame["automated_maximum_tumour_diameter_cm"].astype(float),
        ]
    )


def fit_trace_clinical(
    frame: pd.DataFrame,
    label_col: str = "label",
    regularization_c: float = 1.0,
) -> dict:
    """Fit the prespecified logistic TRACE-Clinical model on development data."""

    x = _design_matrix(frame)
    y = frame[label_col].astype(int).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("TRACE-Clinical predictors must be complete and finite")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=2000)
    model.fit((x - mean) / scale, y)
    return {
        "model": "TRACE-Clinical logistic regression",
        "features": list(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "regularization_c": regularization_c,
        "n": int(len(frame)),
        "events": int(y.sum()),
    }


def apply_trace_clinical(frame: pd.DataFrame, specification: dict) -> np.ndarray:
    if specification["features"] != list(FEATURES):
        raise ValueError("TRACE-Clinical feature contract does not match this implementation")
    x = _design_matrix(frame)
    mean = np.asarray(specification["mean"], dtype=float)
    scale = np.asarray(specification["scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    logit = ((x - mean) / scale) @ coefficient + float(specification["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))


def save_clinical_specification(specification: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    return destination


def load_clinical_specification(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
