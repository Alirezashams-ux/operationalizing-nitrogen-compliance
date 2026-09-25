# Corrected Evaluation Protocol

## 1. Purpose
This protocol defines a leakage-safe, nested, horizon-aware temporal evaluation framework for Ulsan TNout forecasting that can be validated before any model training begins. It addresses decision D from the leakage audit by preventing held-out-informed model selection and by enforcing purge conditions through explicit date arithmetic.

## 2. Canonical Dataset
- Locked dataset: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/00_provenance/recovered_inputs/figure2/Ulsan_Yongsan.csv
- Lock metadata: revision_2026/03_corrected_protocol/canonical_dataset_lock.json
- Required checks before split generation:
  - SHA-256 must equal 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb.
  - Row count, date range, and unique-date count must match the lock file.
  - Date column is Date and target column is TNout.
- No dataset substitution is allowed.

## 3. Forecast Issue Date and Target Date
For each observation and horizon H in {1, 3, 5}:
- feature_date = forecast issue date
- target_date = date of TNout being predicted
- target_date = feature_date + H days

All leakage conditions are validated on feature_date and target_date, not only on row counts.

## 4. Outer Expanding Blocked Evaluation
- Outer folds: 3 expanding blocked folds for each horizon H.
- Canonical outer test windows are preserved from:
  - results/tables/blocked_cv_assignment_matrix_H1.csv
  - results/tables/blocked_cv_assignment_matrix_H3.csv
  - results/tables/blocked_cv_assignment_matrix_H5.csv
- Outer test dates are immutable within each horizon and must remain identical across all compared model families.

## 5. Horizon-Aware Outer Purge
For each horizon and outer fold:
- Start from canonical outer train/test assignment.
- Purge outer-train rows that violate:
  - max(outer_train_target_date) < min(outer_test_feature_date)
- Equivalent row-level rule:
  - drop any outer-train row where target_date >= min(outer_test_feature_date)
- Fail immediately if the inequality is not satisfied after purging.

## 6. Inner Purged Model Selection
Within each outer fold and horizon:
- Construct a chronological validation partition from outer-train-after-outer-purge.
- Purge inner-train rows that violate:
  - max(inner_train_target_date) < min(inner_validation_feature_date)
- Equivalent row-level rule:
  - drop any inner-train row where target_date >= min(inner_validation_feature_date)
- Hyperparameter selection is performed only inside outer-training data, using the purged inner split.
- Fail immediately if inner inequality is not satisfied after purging.

## 7. Fold-Local Preprocessing
All preprocessing must be fitted only inside the corresponding training partition:
- For inner selection: fit only on inner-train-after-inner-purge.
- For final outer-fold fit: fit only on outer-train-after-outer-purge (or policy-defined fold-local training partition).
- No scaler, imputer, transformer, rank normalizer, or threshold calibrator may use outer-test outcomes.

## 8. Model-Specific Selection Rules
- Persistence:
  - no fitted parameters; regenerate metrics on corrected canonical dates.
- Boundary-affected models:
  - refit on corrected purged folds.
- Held-out-informed selection affected models:
  - require nested revalidation; no global post-hoc selection from concatenated outer-test metrics.
- Missing artifacts:
  - rerun under this protocol with documented seeds and fixed scope.
- Final manuscript model selection rule must be predeclared and independent of outer-test comparisons.

## 9. Common Test-Date Requirement
For each horizon and outer fold:
- Test dates are fixed by canonical assignment matrices.
- Every model family must emit predictions for exactly that test-date set.
- Any missing test-date predictions must be flagged as incomplete and excluded from cross-model claims until resolved.

## 10. Retrospective Alarm-Budget Benchmark
Retrospective benchmarking is allowed only as explicitly non-deployable analysis:
- Global top-k rule: k = ceil(r x N)
- Full held-out score distribution may be used for retrospective ranking only.
- Results must be labeled retrospective test-set benchmark and must not be used for model or operating-point selection.

## 11. Prospective Online Alarm Policy
Deployable alarm policy must be evaluated separately:
- Calibrate risk cutoff using only training/validation data within each outer fold.
- Freeze cutoff before any outer-test outcomes are viewed.
- Process outer-test observations chronologically, without ranking the complete test block retrospectively.
- Report realized alarm rate, TP, FP, FN, precision, recall, false-alarm rate, and miss rate.
- Maintain a date-stamped alarm log.
- Permit updates only using information that would be available by that date.

## 12. Reproducibility Controls
- Locked dataset checksum and metadata verification is mandatory.
- Deterministic split generation with fixed parameters and stable sort order.
- Split assignments and summaries are written as revision-only artifacts:
  - corrected_split_assignment.csv
  - corrected_split_summary.csv
- Automated leakage assertions must pass before training authorization.
- Repeated split generation must yield checksum-identical outputs.

## 13. Acceptance Criteria
The protocol is accepted only if all are true:
- Canonical dataset lock checks pass.
- Outer test windows are preserved from canonical matrices.
- Outer purge inequality passes for all H and folds.
- Inner purge inequality passes for all H and folds.
- No train/validation/test date overlap within fold.
- target_date = feature_date + H is satisfied for every row.
- Split outputs are deterministic and checksum-identical across repeated runs.
- Model rerun plan follows audit classifications without overrides lacking evidence.
