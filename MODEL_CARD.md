# TRACE model card

## Intended use

Research evaluation of preoperative TFE3-rRCC triage in patients whose renal
mass is already considered likely to represent RCC on paired noncontrast and
arterial-phase CT. A high score is intended to prioritise expert pathology
review and molecular confirmation after tissue acquisition.

## Not intended for

- definitive diagnosis of TFE3-rRCC;
- benign-versus-malignant renal-mass classification;
- RCC subtypes outside the prespecified TFE3-rRCC, clear-cell RCC, and papillary
  RCC spectrum;
- treatment selection or prediction of treatment response;
- use without local validation, calibration, and governance review.

## Inputs and outputs

Deployment inputs are paired CT only. The output is a continuous `TRACE-CT`
score and, when a training-derived threshold is supplied, a referral flag.
Postoperative H&E is a development-only privileged input.

## Known risks

TFE3-rRCC is rare, so positive predictive value is prevalence dependent.
Scanner, reconstruction, demographic, referral, and molecular-testing shifts
may alter calibration. Detection errors precede subtype errors. The score may
also encode tumour size or other prognostic correlates.

## Validation required

Report centre-separated discrimination, calibration, operating characteristics,
false lesion detections, false molecular referrals, subgroup estimates, and
decision consequences. Do not recalibrate or select thresholds in held-out
cohorts.
