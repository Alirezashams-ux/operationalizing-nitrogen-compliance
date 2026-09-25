# HGBR Definition and Provenance Recovery

## Scope
This document records recoverable submitted HGBR lineage for H1, H3, and H5 before any corrected-protocol fitting.

Model family under review:
- Estimator: sklearn.ensemble.HistGradientBoostingRegressor
- Training scripts recovered:
  - src/train_hgbr_optuna_multi.py
  - src/train_hgbr_optuna_v2.py
- Prediction scripts recovered:
  - src/save_hgbr_predictions.py
  - src/save_hgbr_predictions_v2.py

## Common recovered implementation details
Recovered from source scripts:
- Loss: default HistGradientBoostingRegressor loss (squared_error)
- Early stopping: True
- validation_fraction: 0.1
- random_state: 42
- Missing-value handling: native tree handling; no explicit imputer in HGBR scripts
- Candidate search design: Optuna trial-based search over continuous/int ranges
- Candidate selection metric in training scripts: mean MAE over blocked TimeSeriesSplit folds
- Outer-test usage in original pipeline: submitted artifacts were generated in non-nested workflows and are not authorized for corrected selection

## Horizon-specific provenance recovery

### H1
Recovered evidence:
- Feature artifact exists: features/ulsan_H1_features.npz
- Missing submitted prediction artifact: results/predictions/hgbr_optuna_H1_preds.csv (not present)
- Missing submitted selected-params JSON: results/hgbr/hgbr_optuna_H1.json (not present)
- Recoverable candidate search space from src/train_hgbr_optuna_multi.py:
  - learning_rate: log-uniform [0.01, 0.2]
  - max_iter: int [200, 1500]
  - max_leaf_nodes: int [15, 255]
  - min_samples_leaf: int [10, 160]
  - l2_regularization: log-uniform [1e-6, 20.0]
  - max_bins: int [64, 255]

Decision:
- Classification: B
- Rationale: exact submitted best config is missing, but a documented candidate space and feature artifact are recoverable; corrected nested fold-local selection is therefore authorized.

### H3
Recovered evidence:
- Submitted prediction artifact exists: results/predictions/hgbr_optuna_H3_preds.csv
- Submitted selected-params artifact exists: results/hgbr/hgbr_optuna_H3.json
- Feature artifacts exist in two lineages:
  - features/ulsan_H3_features.npz
  - features/ulsan_H3_features_v2.npz
- final_chosen_hyperparameters_by_horizon_model.csv references ordinary lineage (hgbr_optuna_H3.json)
- Source-level candidate spaces recoverable:
  - ordinary: src/train_hgbr_optuna_multi.py
  - v2: src/train_hgbr_optuna_v2.py

Lineage resolution for corrected rerun:
- Use ordinary H3 lineage for provenance continuity with traceable submitted H3 prediction + selected-params artifacts.
- Apply corrected canonical split assignment (747 held-out rows total under corrected protocol), not legacy 759 reporting context.

Decision:
- Classification: B
- Rationale: candidate space and implementation are recoverable; corrected nested selection is required.

### H5
Recovered evidence:
- Submitted prediction artifacts exist in two lineages:
  - ordinary: results/predictions/hgbr_optuna_H5_preds.csv (759)
  - v2: results/predictions/hgbr_optuna_H5_v2_preds.csv (747)
- Submitted selected-params artifacts exist in two lineages:
  - ordinary: results/hgbr/hgbr_optuna_H5.json
  - v2: results/hgbr/hgbr_optuna_H5_v2.json
- Feature artifacts exist in two lineages:
  - features/ulsan_H5_features.npz
  - features/ulsan_H5_features_v2.npz
- corrected_split_assignment.csv canonical H5 held-out total is 747.

Lineage resolution for corrected rerun:
- Use v2 H5 lineage for corrected-run fitting because corrected canonical date regime for H5 is 747 and the v2 H5 artifact is the traceable 747 submitted lineage.
- Keep ordinary H5 lineage only for post-run comparison (not for corrected-run selection).

Decision:
- Classification: B
- Rationale: although multiple historical lineages exist, the corrected protocol provides a deterministic tie-break for authorized rerun lineage (canonical 747 context), and the v2 candidate space is fully recoverable.

## Horizon classification summary
- H1: B
- H3: B
- H5: B

No horizon is classified C or D, so all three horizons are authorized for corrected nested fitting under this predeclared provenance decision.

## Explicit H1 authorization decision
H1 is authorized for corrected nested rerun under status B.
Reason: documented candidate search space and feature artifact are recoverable from source and locked feature files, despite missing submitted H1 prediction/config artifacts.
