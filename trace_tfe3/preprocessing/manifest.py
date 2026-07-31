from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_patient_manifest(
    ct_csv: str | Path,
    labels_csv: str | Path,
    slide_embedding_csv: str | Path | None,
    output_csv: str | Path,
) -> Path:
    ct = pd.read_csv(ct_csv)
    labels = pd.read_csv(labels_csv)
    required_ct = {"patient_id", "noncontrast", "arterial"}
    required_labels = {"patient_id", "split", "label"}
    if missing := required_ct.difference(ct.columns):
        raise ValueError(f"ct_csv is missing columns: {sorted(missing)}")
    if missing := required_labels.difference(labels.columns):
        raise ValueError(f"labels_csv is missing columns: {sorted(missing)}")

    output = labels.merge(ct, on="patient_id", how="left", validate="one_to_one")
    output["he_token_embeddings"] = ""
    if slide_embedding_csv:
        slides = pd.read_csv(slide_embedding_csv)
        required_slides = {"patient_id", "stain", "embedding_path"}
        if missing := required_slides.difference(slides.columns):
            raise ValueError(f"slide embedding CSV is missing columns: {sorted(missing)}")
        he = (
            slides[slides["stain"].str.lower().isin(["he", "h&e"])]
            .groupby("patient_id")["embedding_path"]
            .apply(lambda values: ";".join(map(str, values)))
            .rename("he_token_embeddings")
        )
        output = output.drop(columns=["he_token_embeddings"]).merge(
            he, on="patient_id", how="left"
        )

    train = output["split"].eq("train")
    if output.loc[train, "he_token_embeddings"].fillna("").eq("").any():
        raise ValueError("Every training patient must have paired H&E token embeddings")
    if output.duplicated("patient_id").any():
        raise ValueError("Patient identifiers must be unique across the manifest")

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination
