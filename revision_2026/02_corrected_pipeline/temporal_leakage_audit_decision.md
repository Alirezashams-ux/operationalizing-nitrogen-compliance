# Temporal Leakage Audit Decision

## 1. Canonical Temporal Index
- Canonical fold/date sources are blocked_cv_split_summary_all_horizons.csv and blocked_cv_assignment_matrix_H1/H3/H5.csv.
- Feature-to-target mapping is confirmed as target_date = feature_date + horizon from y = TNout.shift(-h) in feature builders.
- Inner validation tails are reconstructed at 15% of outer training length for chronological deep-model validation checks.

## 2. Outer Train–Test Purge
- Outer boundary failures: 9/9 fold-horizon combinations.
- H=1 requires 1-row purge at each fold boundary; H=3 requires 3-row purge; H=5 requires 5-row purge.
- Existing prediction artifacts were generated before applying these horizon purge removals.

## 3. Inner Train–Validation Purge
- TCN/BCR-TCN inner boundary failures: 9/9 (chronological tail present, horizon purge absent).
- HGBR inner boundary failures (reconstructed random validation splits): 9/9.
- ElasticNet/Ridge/HybridRank blends have no explicit chronological validation tail in final fold-fit scripts.

## 4. Feature Availability
- TNout lag/rolling memory features are backward-looking and do not include future TNout values.
- Process and weather rolling features include current-day values (t) but not future values; issue-time availability must be explicitly declared.
- No forward-fill/interpolation feature operations were found in primary feature builders.

## 5. Preprocessing Isolation
- ElasticNet/Ridge scaling is fit on fold-local training data only.
- HGBR uses random internal validation fraction during fitting, violating strict chronological validation isolation.
- TCN normalization and class-weighting are subtrain-local, but validation-tail purge is missing.

## 6. Hyperparameter and Model Selection
- ElasticNet and Ridge variant choice relies on blocked OOF metrics from the same data context used for reporting, not nested revalidation.
- HGBR Optuna tuning is similarly non-nested for final comparative reporting.
- Table 3 mixes ElasticNet variants by operating point and mixes H5 run contexts, indicating post-hoc selection behavior.

## 7. Retrospective Alarm-Budget Classification
- k = ceil(r*N) with global held-out N is classified as retrospective fixed-budget benchmarking.
- Rank normalization over complete fold score distributions is retrospective block scoring.
- Risk-score weighting itself is train-fold-only, but representative reporting choices are held-out-informed.

## 8. Cross-Model Date Equality
- H1 max common-date set: 762 (families incomplete: no H1 BCR-TCN/HGBR artifacts).
- H3 max common-date set among available families: 759, but this is 759 and differs from canonical 747 split.
- H5 max common-date set across four families: 747; manuscript artifacts still mix 747, 759, and 735 contexts.

## 9. Confirmed Leakage Risks
- Missing horizon purge at outer train-test boundaries for all horizons.
- Missing horizon purge at chronological inner subtrain-validation boundary for TCN/BCR-TCN.
- Non-chronological internal validation in HGBR fitting.
- Held-out-informed model/run/operating-point selection in final manuscript lineage.

## 10. Models and Horizons That Can Be Retained
- Persistence H1: recalculate metrics from existing predictions.
- Persistence H3: recalculate metrics from existing predictions.
- Persistence H5: recalculate metrics from existing predictions.

## 11. Models and Horizons That Must Be Rerun
- TCN/BCR-TCN H5: rerun fixed existing hyperparameters on corrected folds.
- HybridRank H5: rerun fixed existing hyperparameters on corrected folds.
- point blend H5: rerun fixed existing hyperparameters on corrected folds.
- risk-score blend H5: rerun fixed existing hyperparameters on corrected folds.
- ElasticNet H1: full nested revalidation required.
- ElasticNet H3: full nested revalidation required.
- ElasticNet H5: full nested revalidation required.
- HGBR H3: full nested revalidation required.
- HGBR H5: full nested revalidation required.
- Missing artifacts requiring regeneration before fair comparison:
  - Ridge H1
  - Ridge H3
  - Ridge H5
  - HGBR H1
  - TCN/BCR-TCN H1
  - TCN/BCR-TCN H3
  - HybridRank H1
  - HybridRank H3
  - point blend H1
  - point blend H3
  - risk-score blend H1
  - risk-score blend H3

## 12. Corrected Evaluation Requirements
- Enforce H-day purge between outer training targets and test feature windows for every horizon/fold.
- For models with inner validation, enforce H-day purge between subtrain targets and validation feature windows.
- Re-run model selection in nested fashion when reporting comparative performance tables.
- Separate retrospective fixed-budget benchmark reporting from deployable online policy claims.

## 13. Final Decision
D. Hyperparameter or model selection used held-out test information; affected configurations require nested revalidation.