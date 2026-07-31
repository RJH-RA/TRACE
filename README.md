# TRACE

Reference implementation of **TRACE**, a pathology-privileged learning framework
for preoperative identification of TFE3-rearranged renal cell carcinoma
(TFE3-rRCC) from paired noncontrast and arterial-phase CT.

TRACE uses postoperative H&E whole-slide images only during development. The
deployed `TRACE-CT` model receives CT alone and returns a continuous TFE3 score.
`TRACE-Clinical` combines the frozen CT score with age, sex, and automatically
measured maximum tumour diameter.

> **Research-use status**
>
> This repository is an implementation scaffold aligned to the locked study
> protocol. It does not contain patient data, pretrained weights, molecular
> results, or a clinically validated checkpoint. It is not a medical device.

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

Prov-GigaPath and DINOv3 weights are not redistributed. Configure their local
paths in [`configs/trace_default.yaml`](configs/trace_default.yaml).

## Data contract

The patient-level manifest contains no raw identifiers. Required columns are:

```text
patient_id,split,label,noncontrast,arterial,he_token_embeddings
```

- `label`: `1` for molecularly confirmed TFE3-rRCC and `0` for the prespecified
  clear-cell or papillary RCC control spectrum.
- `split`: patient-level `train`, `internal_test`, or `external_test`.
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

The inference file contains:

```text
patient_id,trace_ct_score
```

No pathology tensor is loaded by the inference entry point.

### 6. Evaluate a frozen operating point

```bash
python scripts/evaluate_predictions.py \
  --predictions-csv outputs/trace_scores_external.csv \
  --output-csv outputs/trace_metrics_external.csv \
  --score-col trace_ct_score \
  --threshold <training-selected threshold>
```

The intended evaluation uses a threshold selected in the training cohort,
transported unchanged to the internal and external test cohorts, and 2,000
patient-level bootstrap resamples.

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
