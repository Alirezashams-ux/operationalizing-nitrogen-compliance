# Fully Nested Corrected H5 HybridRank

## 1. Purpose
- Build a fully chronological and fully nested H5 HybridRank pathway with strict temporal isolation.

## 2. Submitted Temporal-Selection Problem
- Submitted tuned HybridRank pooled non-held-out folds, allowing later chronology to influence earlier-fold weight tuning.

## 3. Input Verification
- dataset_sha256: 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- split_sha256: f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a
- git_commit: d62721ac686f9419a0fda5e08a3eae38a7eb53b4

## 4. Fully Nested Split Design
- Fold 1: outer_train_N=244, component_subtrain_N_after_embargo=168, component_validation_N=30, hybrid_validation_N=36, outer_test_N=249
- Fold 2: outer_train_N=493, component_subtrain_N_after_embargo=348, component_validation_N=62, hybrid_validation_N=73, outer_test_N=249
- Fold 3: outer_train_N=742, component_subtrain_N_after_embargo=528, component_validation_N=93, hybrid_validation_N=111, outer_test_N=249

## 5. Five-Day Embargoes
- inner_target_separation_pass: True
- hybrid_target_separation_pass: True
- outer_target_separation_pass: True

## 6. Component-Model Selection
- Fold 1 BCR-TCN selected_configuration=not_selected_guard_block objective=checkpoint_selection_blocked_recall_at_5pct_undefined
- Fold 1 ElasticNet selected_configuration={"alpha":0.1,"l1_ratio":0.5,"selection_metric":"inner_validation_MAE","selection_value":1.156095991955227} objective=minimize_inner_validation_MAE_tie_smallest_alpha_then_l1_ratio
- Fold 1 HGBR selected_configuration={"candidate_configuration_id":"h5_cfg_014","parameters":{"early_stopping":true,"l2_regularization":0.29991055508755193,"learning_rate":0.021667009613060388,"max_bins":84,"max_iter":1595,"max_leaf_nodes":180,"min_samples_leaf":15,"random_state":42,"validation_fraction":0.1},"selection_metric":"inner_validation_MAE","selection_value":1.0683731763162192} objective=minimize_inner_validation_MAE_tie_smallest_candidate_configuration_id
- BCR checkpoint gate (component-validation tau16 events and k@5%):
  fold1 N=30 tau16_events=0 k_r005=2 recall_defined=False blocked=True
- Fold 1 tau16 support diagnostic artifact: fold1_component_validation_tau16_diagnostic.csv
  component_validation feature_span=2021-07-26..2021-08-24 target_span=2021-07-31..2021-08-29 N=30 tau16_events=0 prevalence=0.000000
  counts by split boundary are listed per nested role in the diagnostic CSV.
- Decision note: Fold 1 blocked before BCR training: tau16_event_count_zero; recall_at_5pct_undefined; no_checkpoint_selection_permitted

## 7. HybridRank Validation Predictions
- Predictions generated from refit component models on component-development only.

## 8. Event-Support Safeguard
- minimum events required: 5
- minimum alarm slots required: 3
- Guard uses fixed policy at r <= 0.05.
- This sparse-support rule is a conservative revised safeguard, not an exact submitted feature.

## 9. HybridRank Weight Selection
- Candidate grid: 0.0..1.0 step 0.1; nonzero vectors normalized to sum 1.
- Objective order: recall, precision, TP.
- Precision floor: fixed-policy precision on HybridRank validation.

## 10. Fixed and Guarded Policies
- Fixed weights: BCR-TCN=0.50, ElasticNet=0.25, Persistence=0.25, HGBR=0.00.
- Guard condition: r <= 0.05.

## 11. Final Hybrid-Specific Component Refits
- Final component fits use complete corrected outer-training blocks and produce hybrid-specific outer-test predictions.

## 12. Outer-Test Hybrid Scores
- Outer-test scores are rank-only retrospective outputs; no outer-test alarm decisions were computed.

## 13. Test and Future-Fold Isolation
- No outer-test outcomes were used for component selection or weight selection.
- No future folds were used in earlier-fold model/weight selection.

## 14. Difference from Canonical Point Models
- Canonical base-model package remains unchanged; this stage adds hybrid-specific nested predictions and scores only.

## 15. Determinism and Automated Tests
- Build completed deterministically under fixed seeds; 62-assertion test script is generated separately.

## 16. Limitations
1. Submitted tuned HybridRank pooled non-held-out folds, allowing future chronology to influence earlier folds.
2. The revised procedure separates component selection and ensemble selection.
3. Five-day target-date embargoes separate every stage.
4. The fixed policy is used at r <= 0.05.
5. The fixed policy is also used when HybridRank-validation event support is insufficient.
6. Full outer-test fold rank normalization remains retrospective and non-deployable.
7. A separate sequential policy will be evaluated later.
8. Hybrid-specific component predictions are not substituted for the canonical point-forecast package.
9. No outer-test alarm metrics were calculated in this stage.

## 17. Readiness Decision
- run_id: fully_nested_hybridrank_h5_d62721ac686f_6ed147cd_f83baf57

FINAL DECISION

<!-- AUTO_DECISION_START -->
C. Fully nested component or ensemble selection could not be completed safely with the available data.
<!-- AUTO_DECISION_END -->

TERMINAL SUMMARY
- event-support rows: 0
