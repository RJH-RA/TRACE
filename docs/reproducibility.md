# Reproducibility protocol

## Scope of this version

`v0.1.0-pretest` freezes the executable method and analysis interfaces before
real-data training. It is designed to make a later locked experiment
traceable; it does not reproduce manuscript performance numbers because no
real training run or validated checkpoint is included.

## External components

Obtain DINOv3 and Prov-GigaPath from their official repositories under their
respective licences. Record repository commits and checkpoint SHA-256 values.
Do not redistribute third-party weights in this repository.

The production CT configuration uses:

- `ct_encoder.backend: torch_hub_local`
- `ct_encoder.model_name: dinov3_vitb16`
- an official local DINOv3 repository and checkpoint

The loader rejects missing files and does not silently replace DINOv3 with a
generic ImageNet ViT. `compact_test` is reserved for tests and must not be used
for reported experiments.

## Locked execution order

1. Validate de-identification, molecular labels, patient uniqueness, and the
   four patient-level split roles.
2. Register noncontrast CT to arterial CT and create the three locked channels.
3. Run renal-tumour localisation, retain its masks and measurements, and build
   the patient manifest.
4. Extract Prov-GigaPath tile tokens from development H&E slides. ASROT consumes
   token-level features; a pooled slide vector is not an acceptable substitute.
5. Train the pathology teacher using development rows only.
6. Train TRACE-CT with classification warm-up followed by ASROT transfer.
7. Run pathology-free inference separately for train, internal test, and
   external test.
8. Select the operating point only from training predictions and save its JSON.
9. Fit TRACE-Clinical only from training predictions and save its JSON.
10. Apply both frozen models and the unchanged threshold to held-out cohorts.
11. Generate a run manifest containing the code commit and SHA-256 values for
    the configuration, patient manifest, checkpoints, and prediction files.

## Required evidence for a future tested release

- exact cohort/split manifest hash and a leakage audit;
- external-component versions and checkpoint hashes;
- successful CI, unit tests, and synthetic smoke test;
- complete command log and environment lock;
- training curves and checkpoint-selection record;
- patient-level predictions for independently regenerated tables and figures;
- threshold and TRACE-Clinical JSON files showing training-only fitting;
- centre-separated discrimination, calibration, and operating metrics;
- segmentation/detection evaluation and failure analysis.

## Non-goals

The repository intentionally excludes DICOM, WSI, direct identifiers, linkage
tables, molecular reports, restricted embeddings, trained weights, and
manuscript result tables.
