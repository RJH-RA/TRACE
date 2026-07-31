# Data governance

Only de-identified manifests and derived tensors approved for the relevant
research environment may be used. The public repository contains schema-only
examples.

Every locked run should record:

- manifest SHA-256 and cohort counts;
- code commit and configuration SHA-256;
- CT preprocessing version and segmentation checkpoint SHA-256;
- DINOv3 and Prov-GigaPath checkpoint identifiers;
- random seed, software environment, and execution host;
- final checkpoint, prediction, table, and figure hashes.

Keep the patient-linkage key outside the repository. Use opaque patient IDs and
one patient-level split. Confirm that multifocal lesions, serial examinations,
slides, molecular results, and follow-up from one patient cannot cross splits.
