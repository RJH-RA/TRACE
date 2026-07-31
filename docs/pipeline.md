# TRACE pipeline

## Information boundary

Postoperative H&E is privileged development information. Internal-test,
external-test, and clinical inference paths accept paired preoperative CT only.
Molecular results define labels but are never model inputs.

## 1. Cohort manifest

Create one de-identified patient-level manifest. The same patient must never
appear in more than one split. Training rows require paired H&E token
embeddings; held-out rows may leave that column empty.

## 2. Paired CT preparation

Register noncontrast CT rigidly to the arterial/corticomedullary phase, resample
to 1.0 x 1.0 x 3.0 mm, clip to -150 to 350 HU, scale to [0, 1], and crop a
16 x 128 x 128 tumour-centred ROI. The model receives noncontrast, arterial,
and arterial-minus-noncontrast channels.

## 3. TRACE-d integration

TRACE-d provides kidney and renal-tumour masks. Components must intersect the
kidney mask, exceed the configured minimum volume, and are ranked before a
maximum of three candidates is retained. Detection and segmentation are
evaluated separately from subtype classification.

## 4. Pathology teacher

Prov-GigaPath extracts H&E token embeddings. An adapter projects tokens to the
256-dimensional shared space, and attention pooling supplies a patient-level
teacher classifier. The token set, not only the pooled vector, is used by
ASROT.

## 5. TRACE-c and ASROT

The slice-wise DINOv3 CT encoder generates 16 axial CT tokens. ASROT fixes the
pathology marginal and relaxes the CT marginal with generalized KL
regularisation:

```text
<T, C> + epsilon * sum(T * (log(T) - 1))
       + tau * KL(T 1 || a),  subject to T^T 1 = b
```

where `C` is cosine distance, `a` and `b` are uniform CT and pathology masses,
and the default `(epsilon, tau, lambda_ot)` is `(0.05, 0.10, 0.20)`.

## 6. Locked training and inference

The first 20 epochs optimise classification without pathology transfer. The
remaining schedule adds ASROT while retaining the prespecified code, seed,
cohort, and final-epoch checkpoint rule. Inference loads only the CT encoder,
token aggregator, and classification head.

## 7. Evaluation

Choose the operating point in the training cohort, transport it unchanged, and
report internal and external estimates separately. Use 2,000 patient-level
bootstrap resamples. `TRACE-Clinical` is a separately locked logistic model
using `TRACE-CT score + age + sex + automated maximum tumour diameter`.
