# Canonical H5 Cross-Model Prediction Assembly

## 1. Purpose
- Build one immutable, matched-date canonical H5 package for Persistence, Ridge, ElasticNet, HGBR, and BCR-TCN v1.1.
- Provide the single source of truth for downstream H5 point metrics, retrospective alarm-budget, HybridRank, uncertainty, and sequential policy analyses.

## 2. Input Verification
- verification_pass: True
- dataset_sha256: 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- split_sha256: f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a
- git_branch: controlled-reruns-v1
- git_commit: 596f1d05ff477c317b9712d1233b64c6443e8c32

## 3. Source Models and Run Lineage
- models included: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1
- completion decisions: {"Persistence": "A", "Ridge": "A", "ElasticNet": "A", "HGBR": "A", "BCR-TCN v1.1": "A"}

## 4. Canonical Key Definition
- key: horizon, outer_fold, feature_date, target_date
- authoritative source: corrected_split_assignment.csv outer_test rows at H=5
- canonical H5 key count: 747

## 5. Date and Target Equality
- date-set equality across models: True
- target_date equality rule: target_date == feature_date + 5 days
- y_true cross-model equality (tight tolerance): True

## 6. Prediction Preservation
- Numeric prediction values were carried from source files without modification.
- source prediction preservation pass: True
- Representation-only normalization applied: horizon->int(5), outer_fold->int, feature_date/target_date->YYYY-MM-DD, model labels canonicalized.

## 7. Long-Format Canonical Table
- file: canonical_h5_predictions_long.csv
- includes model-wise predictions, risk scores, errors, event labels, and source lineage fields.

## 8. Wide-Format Canonical Table
- file: canonical_h5_predictions_wide.csv
- one canonical key row with side-by-side model predictions and BCR-TCN threshold probabilities.

## 9. Point-Metric Reconciliation
- pooled canonical metrics reconcile to approved source pooled metrics: True
- pooled RMSE is computed as sqrt(pooled MSE).

## 10. Event Prevalence
- event-count equality across models (by fold and pooled): True
- event labels are derived from shared y_true thresholds at 15, 16, and 17 mg/L.

## 11. Source and Checksum Registry
- source registry file: canonical_h5_source_registry.csv
- package checksums file: canonical_h5_checksums.sha256
- assembly_id: canonical_h5_cross_model_6ed147cd_f83baf57_596f1d05ff47

## 12. Automated Tests
- test_result: pass_21_of_21
- deterministic assembly status: pass

## 13. Deviations or Limitations
- No source prediction numeric value was altered.
- No alarm flags, top-k selections, rank normalization, hybrid weighting, or sequential policy cutoffs were produced.

## 14. Readiness Decision
- Automated tests passed; canonical assembly is release-ready.

FINAL DECISION

A. Canonical H5 cross-model assembly passed; retrospective alarm-budget evaluation and HybridRank regeneration may begin.

TERMINAL SUMMARY

1. input verification result: True
2. models included: Persistence, Ridge, ElasticNet, HGBR, BCR-TCN v1.1
3. canonical key count: 747
4. date-set equality: True
5. y_true equality: True
6. event-count equality: True
7. point-metric reconciliation result: True
8. source prediction preservation result: True
9. deterministic assembly result: pass
10. final decision: A
11. exactly one next action: Begin retrospective alarm-budget evaluation and HybridRank regeneration using this canonical package only.
