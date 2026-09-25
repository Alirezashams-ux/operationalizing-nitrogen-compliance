# Preprocessing and Model-Selection Audit

- PASS rows: 7
- FAIL rows: 7
- UNCLEAR rows: 1

## Preprocessing Isolation
- ElasticNet/Ridge scaler fitting is fold-local in pipeline code.
- HGBR uses random internal validation (validation_fraction=0.1), not chronological blocked validation.
- TCN standardization uses subtrain-only statistics, but no horizon purge before validation tail.

## Hyperparameter and Model Selection
- ElasticNet and Ridge variant choice is based on the same blocked folds later used for reported performance (not nested).
- HGBR Optuna tuning also uses the same blocked data context for selection and reporting.
- HybridRank manuscript lineage mixes guarded and normalized H5 run contexts, indicating post-hoc variant selection risk.