# Corrected Protocol Readiness Report

## 1. Dataset Lock
- Locked dataset file: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/00_provenance/recovered_inputs/figure2/Ulsan_Yongsan.csv
- Locked SHA-256: 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- Lock metadata recorded in canonical_dataset_lock.json (row count, date span, duplicate/missing-date checks, schema).
- Split generator verifies checksum and metadata before constructing splits.

## 2. Outer Split Reconstruction
- Canonical outer test windows were reconstructed from:
  - results/tables/blocked_cv_assignment_matrix_H1.csv
  - results/tables/blocked_cv_assignment_matrix_H3.csv
  - results/tables/blocked_cv_assignment_matrix_H5.csv
- Three expanding blocked outer folds are preserved for each horizon H in {1,3,5}.
- Output: corrected_split_assignment.csv and corrected_split_summary.csv.

## 3. Outer Purge Verification
- Horizon-aware outer purge was applied by date condition:
  - max(outer_train_target_date) < min(outer_test_feature_date)
- Total outer purged rows by horizon:
  - H1: 3
  - H3: 9
  - H5: 15
- All outer boundary pass flags are True in corrected_split_summary.csv.

## 4. Inner Split Construction
- Inner partition strategy: chronological validation tail from outer-train-after-outer-purge.
- Validation proportion: deterministic 15% tail (ceil-based count).
- Inner split labels are explicit in corrected_split_assignment.csv:
  - inner_train
  - inner_validation
  - not_applicable

## 5. Inner Purge Verification
- Inner boundary purge condition:
  - max(inner_train_target_date) < min(inner_validation_feature_date)
- Total inner purged rows by horizon:
  - H1: 3
  - H3: 9
  - H5: 15
- All inner boundary pass flags are True in corrected_split_summary.csv.

## 6. Common Test Dates
- For each horizon and outer fold, outer_test dates in corrected_split_assignment.csv exactly match canonical assignment matrices.
- The split artifact has no model-family dimension, enforcing shared canonical test-date sets across all model families.

## 7. Model Rerun Matrix
- Created: rerun_execution_plan.csv
- Rows: 24 (aligned to audited model_rerun_decision.csv)
- Decision classes preserved:
  - recalculate metrics from existing predictions: 3 rows
  - rerun using fixed existing hyperparameters on corrected folds: 4 rows
  - retrain and revalidate because leakage affected fitting: 5 rows
  - missing model artifact: 12 rows

## 8. Retrospective Alarm Benchmark
- Retrospective top-k benchmark is explicitly retained as non-deployable diagnostic only.
- Rule fixed as k = ceil(r x N) with mandatory retrospective labeling.

## 9. Prospective Online Policy
- Prospective protocol defined in alarm_evaluation_protocol.md:
  - cutoff calibrated on train/validation only
  - cutoff frozen before outer-test outcomes
  - chronological outer-test processing
  - dated alarm log and realized-rate reporting

## 10. Automated Test Results
- Test file: test_purged_nested_splits.py
- Executed tests: 10
- Result: all passed
- Verified checks include leakage inequalities, test-window preservation, determinism, and checksum-identical repeated execution.

## 11. Remaining Blockers
- No protocol-construction blockers remain.
- Model artifact gaps and rerun workloads remain execution tasks, not split-protocol validation blockers.
- Training must still follow rerun_execution_plan.csv and corrected_evaluation_protocol.md without held-out-informed selection.

## 12. Training Authorization Decision
A. Corrected protocol validated; controlled model reruns may begin.
