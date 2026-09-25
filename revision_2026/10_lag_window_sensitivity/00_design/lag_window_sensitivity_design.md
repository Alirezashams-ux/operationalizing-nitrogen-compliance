# Lag-Window Sensitivity Design (Stage 2, Design Gate)

## Objective
- Establish a predeclared, leakage-safe TNout-memory sensitivity design without refitting in this stage.

## Reviewer Issue Addressed
- Assess whether forecast robustness depends materially on TNout memory depth while preserving corrected protocol constraints.

## Authoritative Feature Lineage
- Ordinary corrected (H1/H3 and Ridge/ElasticNet H5): TNout_lag1 + TNout_roll7 + TNout_roll14.
- V2 corrected H5 lineage (HGBR/BCR-TCN): TNout_lag1/3/5/7 + TNout_roll7/14/30.
- H3 v2 artifact exists but is not used by final corrected H3.

## Configuration Definitions
- SHORT: TNout features=['TNout_lag1', 'TNout_roll7'] max_history_days=7 earliest_possible_date=2021-01-11 role=reduced-memory sensitivity
- REFERENCE: TNout features=['TNout_lag1', 'TNout_roll14', 'TNout_roll7'] max_history_days=14 earliest_possible_date=2021-01-18 role=ordinary corrected feature reference
- LONG: TNout features=['TNout_lag1', 'TNout_lag3', 'TNout_lag5', 'TNout_lag7', 'TNout_roll14', 'TNout_roll30', 'TNout_roll7'] max_history_days=30 earliest_possible_date=2021-02-03 role=extended-memory sensitivity and final HGBR-H5/BCR-TCN-H5 TN-memory lineage

## Model and Horizon Selection
- Fitted sensitivity models (future execution only): Ridge and HGBR across H1/H3/H5.
- Context-only baseline: Persistence (no fitting).
- ElasticNet: frozen contextual evidence only; no refit in minimal design.
- BCR-TCN v1.1: lineage documented for H5 LONG context only; not refit in minimal design.

## Final Reference Mapping
- Ridge H1/H3/H5 -> REFERENCE
- HGBR H1/H3 -> REFERENCE
- HGBR H5 -> LONG

## Non-TN Feature Freeze Rule
- Preserve final corrected non-TN predictors for each model/horizon.
- Vary only TNout memory columns between SHORT/REFERENCE/LONG.
- Do not add/remove weather, influent, or seasonal predictors during sensitivity runs.

## Hyperparameter Freeze Rule
- Preserve corrected fold-specific hyperparameters from frozen rerun manifests.
- No retuning, no inner/outer-test hyperparameter search in sensitivity execution.

## Common-Date Requirement
- Canonical outer-test dates must be identical across configurations.
- Any configuration requiring canonical test-date removal is not authorized.

## Warm-Up and Sample-Retention Expectations
- SHORT warm-up=7 days; REFERENCE warm-up=14 days; LONG warm-up=30 days from raw start 2021-01-04.
- Computed earliest dates: SHORT=2021-01-11, REFERENCE=2021-01-18, LONG=2021-02-03.
- LONG may reduce early H1 training availability while preserving canonical H1 outer-test dates.

## Planned Outcomes
- Future execution will report MAE/MSE/RMSE/MASE and reference-correlations (Pearson/Spearman).
- Representative event-ranking diagnostic: offline Recall@5% at tau=16 mg/L, fold-wise top-k.

## Interpretation Rules
- This is sensitivity analysis, not model optimization.
- Test performance will not be used to choose lag windows.
- Compare robustness patterns; do not declare an optimal configuration.

## Exclusions
- No model fitting, no prediction generation, no sensitivity metric calculation in this stage.
- No manuscript editing in Stage 2 execution gate.

## Limitations
1. Offline sensitivity cannot substitute for prospective operational validation.
2. BCR-TCN behavior is not directly validated by minimal Ridge/HGBR refits.
3. Mixed sensitivity outcomes remain possible despite strict protocol controls.

## Execution Stop Conditions
- Abort if embargo/common-date/hyperparameter-freeze constraints are violated.
- Abort on any source-preservation or checksum integrity failure.

## Future Decision Options
- Robust: proceed with confidence statement while preserving non-optimization framing.
- Mixed/Sensitive: report conditional robustness with explicit caution and no post-hoc changes.
