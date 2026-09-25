# Corrected HGBR Nested Revalidation Report

## 1. Input Verification
- Dataset lock file: revision_2026/03_corrected_protocol/canonical_dataset_lock.json
- Dataset path verified: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/00_provenance/recovered_inputs/figure2/Ulsan_Yongsan.csv
- Dataset SHA-256 expected/observed: 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb / 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb
- Protocol checksum gate: PASS for all files listed in corrected_protocol_v1.sha256
- Split path verified: revision_2026/03_corrected_protocol/corrected_split_assignment.csv
- Split SHA-256: f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a
- Active branch verified: controlled-reruns-v1
- Git commit recorded: 1db2f3472cd48274a5afa4010f446952d76c6042
- Verification status: PASS

## 2. Submitted HGBR Provenance
Recovered provenance sources include:
- src/train_hgbr_optuna_multi.py
- src/train_hgbr_optuna_v2.py
- src/save_hgbr_predictions.py
- src/save_hgbr_predictions_v2.py
- results/predictions/hgbr_optuna_H3_preds.csv
- results/predictions/hgbr_optuna_H5_preds.csv
- results/predictions/hgbr_optuna_H5_v2_preds.csv
- results/hgbr/hgbr_optuna_H3.json
- results/hgbr/hgbr_optuna_H5.json
- results/hgbr/hgbr_optuna_H5_v2.json
- deliverables/si_upload_packages_20260329/generated/final_chosen_hyperparameters_by_horizon_model.csv

Recovered model definition:
- Estimator: HistGradientBoostingRegressor
- Loss: squared_error (default)
- Fixed settings in recovered scripts: early_stopping=True, validation_fraction=0.1, random_state=42
- Search family: Optuna trial-based search over recovered parameter bounds

## 3. H1 Artifact Recovery Decision
- H1 submitted prediction and selected-params artifacts are missing.
- A recoverable candidate search space exists in src/train_hgbr_optuna_multi.py.
- H1 feature artifact exists: features/ulsan_H1_features.npz.
- Decision: H1 authorized under status B (candidate space recovered; fold-local nested selection required).

## 4. Candidate Space and Feature Sets
Predeclared in revision_2026/04_controlled_reruns/hgbr/hgbr_selection_plan.json.

Provenance status by horizon:
- H1: B
- H3: B
- H5: B

Authorized feature sets:
- H1: hgbr_npz_ordinary_H1_18col -> features/ulsan_H1_features.npz
- H3: hgbr_npz_ordinary_H3_18col -> features/ulsan_H3_features.npz
- H5: hgbr_npz_v2_H5_34col -> features/ulsan_H5_features_v2.npz

Search spaces:
- H1/H3: ordinary recovered bounds from train_hgbr_optuna_multi.py
- H5: v2 recovered bounds from train_hgbr_optuna_v2.py

Selection controls:
- Metric: inner_validation_MAE
- Method: deterministic_random_search_over_recovered_optuna_space
- Budget: 16 candidates per horizon
- Tie-break: lexicographically smallest candidate_configuration_id among MAE ties
- Seed policy: fixed candidate sampling seed per horizon + model random_state=42

## 5. Purged Nested Selection Strategy
Per horizon and outer fold:
1. Candidate generation from predeclared recovered search space.
2. Candidate evaluation only on purged inner_validation rows from corrected_split_assignment.csv.
3. Candidate selection before any outer-test outcome access.
4. Refit selected candidate on complete purged outer_train rows.
5. Emit exactly one prediction per outer_test row.

Isolation controls:
- No fold recreation was performed.
- Outer-test keys were validated against the locked assignment and uniqueness constraints.

## 6. Selected Configuration by Fold
From revision_2026/04_controlled_reruns/hgbr/hgbr_selected_configurations.csv:

- H1 fold1: h1_cfg_014
- H1 fold2: h1_cfg_010
- H1 fold3: h1_cfg_007
- H3 fold1: h3_cfg_010
- H3 fold2: h3_cfg_007
- H3 fold3: h3_cfg_004
- H5 fold1: h5_cfg_006
- H5 fold2: h5_cfg_003
- H5 fold3: h5_cfg_012

All selected rows record:
- loss=squared_error
- early_stopping=True
- random_seed=42
- selection_source=corrected_split_assignment_inner_validation

## 7. H1 Results
Pooled (N=762):
- MAE=1.664918
- MSE=4.351177
- RMSE=2.085948
- MASE=1.331702

Fold-level N:
- fold1: 254
- fold2: 254
- fold3: 254

## 8. H3 Results
Pooled (N=747):
- MAE=1.763640
- MSE=5.092504
- RMSE=2.256658
- MASE=1.410724

Fold-level N:
- fold1: 249
- fold2: 249
- fold3: 249

## 9. H5 Results
Pooled (N=747):
- MAE=1.807192
- MSE=5.237965
- RMSE=2.288660
- MASE=1.444501

Fold-level N:
- fold1: 249
- fold2: 249
- fold3: 249

## 10. Event Prevalence and Date Equality
From revision_2026/04_controlled_reruns/hgbr/hgbr_event_prevalence.csv:
- H1 folds: events_tau15=[66,19,56], events_tau16=[34,6,34], events_tau17=[17,1,12]
- H3 folds: events_tau15=[59,20,55], events_tau16=[31,7,33], events_tau17=[16,1,12]
- H5 folds: events_tau15=[59,20,55], events_tau16=[31,7,33], events_tau17=[16,1,12]

Equality checks:
- Event counts and canonical dates match Persistence, Ridge, and ElasticNet on identical corrected outer-test rows.

## 11. Comparison with Persistence, Ridge, and ElasticNet
Pooled corrected metrics summary:

H1 (N=762)
- Persistence: MAE 1.546053, RMSE 2.060279
- Ridge: MAE 1.558675, RMSE 1.958182
- ElasticNet: MAE 1.582431, RMSE 1.985505
- HGBR: MAE 1.664918, RMSE 2.085948

H3 (N=747)
- Persistence: MAE 2.025676, RMSE 2.598513
- Ridge: MAE 1.797504, RMSE 2.254861
- ElasticNet: MAE 1.776871, RMSE 2.234165
- HGBR: MAE 1.763640, RMSE 2.256658

H5 (N=747)
- Persistence: MAE 2.074511, RMSE 2.666128
- Ridge: MAE 1.773188, RMSE 2.261799
- ElasticNet: MAE 1.787277, RMSE 2.273309
- HGBR: MAE 1.807192, RMSE 2.288660

## 12. Comparison with Submitted HGBR Results
Submitted traces considered:
- H3 ordinary: results/predictions/hgbr_optuna_H3_preds.csv (759)
- H5 ordinary: results/predictions/hgbr_optuna_H5_preds.csv (759)
- H5 v2: results/predictions/hgbr_optuna_H5_v2_preds.csv (747)

Separated effects:
- Corrected horizon purge and canonical-date enforcement:
  - H3 submitted native N=759 -> corrected-date overlap N=747
  - H5 ordinary submitted native N=759 -> corrected-date overlap N=747
- Nested selection vs held-out-informed historical context:
  - Corrected rerun performs fold-local purged inner selection with outer-test isolation.
- N=759 vs N=747:
  - H3 submitted ordinary MAE: 1.776702 (native 759), 1.778690 (restricted 747)
  - H3 corrected MAE: 1.763640 (747)
  - H5 ordinary MAE: 1.754919 (native 759), 1.761086 (restricted 747)
  - H5 corrected MAE: 1.807192 (747)
- H5 ordinary vs v2 lineage:
  - H5 v2 submitted MAE (747): 1.795448
  - H5 corrected MAE (747): 1.807192
- Shared-date intersection:
  - H3 corrected overlaps all 747 dates with submitted H3 ordinary.
  - H5 corrected overlaps all 747 dates with submitted H5 v2 and the 747-date subset of submitted H5 ordinary.
- Pooled vs fold-averaged metrics:
  - Corrected protocol reports pooled metrics from concatenated held-out predictions (primary), not mean fold RMSE.
- Different feature sets:
  - H1/H3 corrected use ordinary NPZ feature lineage.
  - H5 corrected uses v2 NPZ feature lineage selected by corrected canonical 747 context.

No post-hoc global superiority claim is made from outer-test performance.

## 13. Automated Test Results
Executed:
- revision_2026/04_controlled_reruns/hgbr/run_corrected_hgbr.py
- revision_2026/04_controlled_reruns/hgbr/test_corrected_hgbr.py

Result:
- 20/20 assertions passed
- Determinism checks passed
- Prediction checksum stability checks passed
- Manifest test_result updated to pass_20_of_20

## 14. Deviations or Blockers
- Deviation: candidate search execution uses deterministic predeclared random search over recovered Optuna spaces rather than importing Optuna runtime state from historical runs.
- Reconciliation outcome: the submitted H3/H5 hyperparameter table-vs-JSON mismatch is confirmed as a submitted reporting provenance issue, not a corrected-run calculation error.
- No blocker prevented corrected HGBR artifact generation, validation, or reconciliation.

## 15. Readiness Decision
Readiness status: Reconciliation complete, no unresolved validity-critical issue.

FINAL DECISION
A. HGBR nested revalidation passed; causal TCN controlled reruns may begin.

TERMINAL SUMMARY
1. Provenance classification by horizon: H1=B, H3=B, H5=B.
2. H1 authorization decision: authorized via recovered documented candidate space (status B).
3. Input verification result: PASS (dataset lock, protocol checksums, split checksum, branch, commit).
4. Candidate search space: recovered ordinary bounds for H1/H3 and recovered v2 bounds for H5; deterministic predeclared search budget=16 per horizon.
5. Selected configuration by horizon/fold: H1={h1_cfg_014,h1_cfg_010,h1_cfg_007}; H3={h3_cfg_010,h3_cfg_007,h3_cfg_004}; H5={h5_cfg_006,h5_cfg_003,h5_cfg_012}.
6. Prediction counts: H1=762, H3=747, H5=747.
7. Pooled MAE/MSE/RMSE/MASE: H1=(1.664918,4.351177,2.085948,1.331702), H3=(1.763640,5.092504,2.256658,1.410724), H5=(1.807192,5.237965,2.288660,1.444501).
8. Event-count equality across completed models: PASS versus Persistence, Ridge, ElasticNet on identical corrected dates.
9. Outer-test isolation result: PASS (selection uses purged inner validation only; outer_test_accessed_during_selection=False for all candidates).
10. Deterministic test result: PASS (repeated execution outputs stable).
11. Comparison with submitted values: reconciliation completed; differences are explained by canonical-date regime, lineage context (ordinary vs v2), aggregation context, and leakage-safe nested selection.
12. Completion decision: A (nested revalidation passed; no unresolved validity-critical issue).
13. Exactly one next action: begin causal TCN controlled reruns under the locked corrected protocol using the finalized HGBR artifacts.
