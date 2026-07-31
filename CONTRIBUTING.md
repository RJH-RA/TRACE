# Contributing

Use a topic branch and keep each change tied to one method or reproducibility
question. Before opening a pull request:

1. run `ruff check trace_tfe3 scripts tests`;
2. run `pytest -q`;
3. confirm no patient data, DICOM, WSI, linkage table, checkpoint, or
   institution-local path is staged;
4. document any change to the manifest schema, preprocessing, model inputs,
   ASROT objective, cohort split, checkpoint rule, or operating-point rule;
5. update tests and the default configuration together.

Do not tune code against internal-test or external-test labels. Changes that
alter the locked scientific method should be proposed separately from bug fixes.
