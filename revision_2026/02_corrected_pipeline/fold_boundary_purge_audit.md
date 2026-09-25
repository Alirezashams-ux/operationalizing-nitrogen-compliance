# Fold Boundary and Purge Audit

## Part A: Canonical Temporal Index
- Canonical split source: results/tables/blocked_cv_split_summary_all_horizons.csv
- Assignment matrices used: blocked_cv_assignment_matrix_H1.csv, blocked_cv_assignment_matrix_H3.csv, blocked_cv_assignment_matrix_H5.csv
- Target shift verified in feature builders as y = TNout.shift(-h), so target_date = feature_date + horizon.

## Part B: Outer Train-Test Boundary
- H=1: FAIL folds=3/3, mean rows requiring purge=1.0.
- H=3: FAIL folds=3/3, mean rows requiring purge=3.0.
- H=5: FAIL folds=3/3, mean rows requiring purge=5.0.
- Existing predictions were generated before any explicit horizon purge-gap was applied.

## Part C: Inner Train-Validation Boundary
- TCN/BCR-TCN uses a chronological 15% validation tail, but no H-day purge between subtrain and validation.
- TCN H=1: FAIL folds=3/3, max overlap target dates=1.
- TCN H=3: FAIL folds=3/3, max overlap target dates=3.
- TCN H=5: FAIL folds=3/3, max overlap target dates=5.
- HGBR uses internal validation_fraction=0.1 random holdout (reconstructed), not chronological tail validation.

## Notes
- Full row-level evidence, offending rows, and purge-day calculations are in fold_boundary_purge_audit.csv.