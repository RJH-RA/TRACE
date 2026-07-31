# TRACE

Reference implementation of **TRACE**, a pathology-privileged learning framework
for preoperative identification of TFE3-rearranged renal cell carcinoma
(TFE3-rRCC) from paired noncontrast and arterial-phase CT.

TRACE uses postoperative H&E whole-slide images only during development. The
deployed `TRACE-CT` model receives CT alone and returns a continuous TFE3 score.
`TRACE-Clinical` combines the frozen CT score with age, sex, and automatically
measured maximum tumour diameter.

> **Pretest implementation status (`v0.1.0-pretest`)**
>
> This version freezes the code and data contracts before real-data training.
> Unit tests and a synthetic, dependency-light smoke test validate interfaces;
> they do not validate scientific performance. The repository contains no
> patient data, pretrained weights, molecular results, manuscript result
> tables, or clinically validated checkpoint. It is not a medical device.

## Method overview

TRACE separates development-time privileged information from application-time
inputs:

1. **TRACE-d** detects and segments kidneys and renal tumours from registered
   noncontrast and arterial-phase CT.
2. **Pathology teacher** represents postoperative H&E tissue with frozen
   Prov-GigaPath features and attention-based patient aggregation.
3. **TRACE-c** applies a shared slice-wise DINOv3 encoder to the CT tumour ROI
   and aggregates the axial tokens for TFE3-rRCC classification.
4. **ASROT transfer** aligns CT and H&E token sets with asymmetric semi-relaxed
   optimal transport. The pathology marginal is fixed; the CT marginal is
   relaxed with generalized KL regularisation.
5. **CT-only inference** removes the pathology teacher and transport branch.

The default method contract uses:

```text
CT phases:          noncontrast + arterial/corticomedullary
CT channels:        noncontrast, arterial, arterial-minus-noncontrast
ROI:                16 x 128 x 128 voxels
shared dimension:   256
ASROT epsilon:      0.05
ASROT tau:          0.10
ASROT loss weight:  0.20
```

## Installation

Python 3.10 or newer is required.

```bash
git clone <TRACE repository URL>
cd TRACE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Prov-GigaPath and DINOv3 source code and weights are not redistributed.
Configure their local paths in
[`configs/trace_default.yaml`](configs/trace_default.yaml). Relative paths in a
configuration file are resolved against that file's directory. The production
DINOv3 backend uses the official local repository through `torch.hub`; it does
not silently substitute a generic ViT.

## Data contract

The patient-level manifest contains no raw identifiers. Required columns are:

```text
patient_id,split,label,noncontrast,arterial,he_token_embeddings
```

- `label`: `1` for molecularly confirmed TFE3-rRCC and `0` for the prespecified
  clear-cell or papillary RCC control spectrum.
- `split`: patient-level `train`, `validation`, `internal_test`, or
  `external_test`. The validation split is development-only.
- `noncontrast`, `arterial`: paths to registered tumour-centred arrays.
- `he_token_embeddings`: a path or semicolon-separated paths to development-only
  H&E token tensors. This field is required only for training rows.

Optional deployable variables are:

```text
age,sex,automated_maximum_tumour_diameter_cm
```

See [`data/trace_manifest.example.csv`](data/trace_manifest.example.csv).
Never commit DICOM, WSI, linkage tables, direct identifiers, institution-local
mount paths, or checkpoints derived from restricted data.

## Pipeline

### 1. Prepare paired CT

```bash
python scripts/preprocess_ct.py \
  --cases-csv data/ct_cases.csv \
  --output-dir outputs/ct \
  --output-csv outputs/ct/ct_manifest.csv \
  --config configs/trace_default.yaml
```

### 2. Extract H&E token embeddings

```bash
python scripts/tile_wsi.py \
  --config configs/trace_default.yaml \
  --slides-csv data/slides.csv \
  --output-dir outputs/pathology
```

### 3. Build the locked patient manifest

```bash
python scripts/build_trace_manifest.py \
  --ct-csv outputs/ct/ct_manifest.csv \
  --labels-csv data/labels.csv \
  --slide-embedding-csv outputs/pathology/slide_embedding_manifest.csv \
  --output-csv data/trace_manifest.csv
```

### 4. Train the pathology teacher and TRACE-CT

```bash
python scripts/train_pathology.py \
  --config configs/trace_default.yaml

python scripts/train_trace.py \
  --config configs/trace_default.yaml \
  --pathology-checkpoint outputs/pathology_he/best.pt
```

### 5. Run CT-only inference

```bash
python scripts/infer_trace.py \
  --config configs/trace_default.yaml \
  --checkpoint outputs/trace_ct/final.pt \
  --split external_test \
  --output outputs/trace_scores_external.csv
```

The offline inference file contains:

```text
patient_id,split,label,trace_ct_score,age,sex,automated_maximum_tumour_diameter_cm
```

The label and clinical columns are copied from the de-identified analysis
manifest so the same file can be evaluated; they are not model inputs. No
pathology tensor is loaded by the inference entry point.

### 6. Lock the operating point in training data

```bash
python scripts/select_operating_point.py \
  --predictions-csv outputs/trace_scores_train.csv \
  --output-json outputs/trace_threshold.json \
  --minimum-sensitivity 0.80
```

The JSON record includes the selection rule, source-file SHA-256, sample size,
event count, and locked threshold.

### 7. Evaluate the unchanged operating point

```bash
python scripts/evaluate_predictions.py \
  --predictions-csv outputs/trace_scores_external.csv \
  --output-csv outputs/trace_metrics_external.csv \
  --score-col trace_ct_score \
  --threshold-json outputs/trace_threshold.json
```

### 8. Fit and apply TRACE-Clinical

```bash
python scripts/fit_trace_clinical.py \
  --predictions-csv outputs/trace_scores_train.csv \
  --output-json outputs/trace_clinical.json

python scripts/apply_trace_clinical.py \
  --predictions-csv outputs/trace_scores_external.csv \
  --model-json outputs/trace_clinical.json \
  --output-csv outputs/trace_clinical_external.csv
```

The four prespecified predictors are the TRACE-CT score, age, sex, and
automatically measured maximum tumour diameter. The frozen JSON stores feature
order, scaling statistics, coefficients, intercept, sample counts, and source
hash. Test cohorts are never used to refit the model.

### 9. Record a locked run

```bash
python scripts/write_run_manifest.py \
  --input configs/trace_default.yaml \
  --input data/trace_manifest.csv \
  --input outputs/trace_ct/final.pt \
  --output outputs/run_manifest.json
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete
reproduction sequence and current validation boundary.

## Repository structure

```text
configs/                    Locked experiment configuration
data/                       De-identified example manifests only
docs/                       Pipeline, governance, and method notes
trace_tfe3/
  data/                     Patient-level CT/H&E data contract
  models/                   DINOv3 CT encoder, pathology teacher, ASROT, TRACE
  preprocessing/            Paired CT and manifest preparation
  training/                 Losses, training loop, and metrics
  evaluation/               Diagnostic evaluation utilities
scripts/                    Reproducible command-line entry points
tests/                      Data-contract and ASROT regression tests
```

## Reproducibility boundaries

- Split at the patient level; never allow one patient's images, slides, lesions,
  or outcomes to cross cohorts.
- Fit models, preprocessing statistics, operating points, and calibration only
  in the training cohort.
- Keep H&E and molecular results out of internal/external inference.
- Record configuration, code commit, checkpoint hash, manifest hash, and output
  hash for every locked run.
- Treat the recurrence-free-survival analysis as a separate exploratory
  analysis, not as a training objective.

## Code validation

No real-data training is performed in this release. The repository can still
be checked without DINOv3 or Prov-GigaPath downloads:

```bash
ruff check trace_tfe3 scripts tests
pytest -q
python scripts/smoke_test.py
```

The smoke test exercises the paired-phase CT contract, enhancement channel,
deployable network tensor shapes, ASROT marginal constraint and gradients,
operating-point selection, and TRACE-Clinical fit/application.

## Citation

```bibtex
@article{trace_tfe3,
  title   = {Pathology-privileged learning for preoperative CT identification of TFE3-rearranged renal cell carcinoma},
  author  = {TRACE Investigators},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

TRACE was derived from the repository architecture of
[PIVOT](https://github.com/HepatoAI-Lab/PIVOT). The retained Git history and
`upstream-pivot` remote preserve that provenance.

## License

The source code is released under the Apache License 2.0. Third-party models,
weights, and datasets remain subject to their original licences and access
conditions.
