# Canonical Corrected H1/H3 Cross-Model Reconciliation

## 1. Scope
- Reconciled horizons: H1 and H3 only.
- Reconciled models: Persistence, Ridge, ElasticNet, HGBR.
- Explicit exclusions: H5, BCR-TCN, TCN, point blend, HybridRank, fixed rank ensemble, alarm flags, top-k, classification metrics
- No model fitting, tuning, retraining, alarm policies, or classification metrics were run.

## 2. Preflight and Source Integrity
- verification_pass: True
- source test results: {"ElasticNet": "pass_15_of_15", "HGBR": "pass_20_of_20", "Persistence": "pass_12_of_12", "Ridge": "pass_15_of_15"}
- source completion decisions: {"ElasticNet": "A", "HGBR": "A", "Persistence": "A", "Ridge": "A"}
- local freeze-tag commits: {"corrected-protocol-v1": "2bbf5eba419b0650693f1e88326107bf9a346c68", "elasticnet-corrected-v1": "1db2f3472cd48274a5afa4010f446952d76c6042", "hgbr-corrected-v1": "4d16b4d513e4fd9aa7c1acd85d0b5894ce4d5d62", "persistence-corrected-v1": "41f184a29ff88aaf275f3365267b8c909ce7598a", "ridge-corrected-v1": "69cfb2bf8bc181dc1de1e09f0c7477a1a5de7585"}
- remote freeze-tag presence (revision-origin): {"corrected-protocol-v1": true, "elasticnet-corrected-v1": true, "hgbr-corrected-v1": true, "persistence-corrected-v1": true, "ridge-corrected-v1": true}
- source preservation against local tags: True
- provenance warnings: ["locked_protocol_registry_missing_files_noncritical: build_purged_nested_splits.py,test_purged_nested_splits.py"]

## 3. Canonical Sample Sizes and Date Ranges
- H1: N=762, fold_counts={1: 254, 2: 254, 3: 254}, feature_date=2021-09-29..2023-10-30, target_date=2021-09-30..2023-10-31
- H3: N=747, fold_counts={1: 249, 2: 249, 3: 249}, feature_date=2021-10-12..2023-10-28, target_date=2021-10-15..2023-10-31

## 4. Event Counts
- H1 pooled events: tau15=141 tau16=74 tau17=30 (prevalence: 0.18503937007874016, 0.09711286089238845, 0.03937007874015748)
- H3 pooled events: tau15=134 tau16=71 tau17=29 (prevalence: 0.17938420348058903, 0.09504685408299866, 0.038821954484605084)

## 5. Metric Definitions
- MAE = mean(abs(y_true - y_pred)).
- MSE = mean((y_true - y_pred)^2).
- RMSE = sqrt(MSE).
- Pooled MAE/MSE/RMSE are computed directly over pooled canonical rows.
- MASE period is m=1 with fold-local training-only denominators.
- MASE method: mean_absolute_scaled_error_over_pooled_rows_using_fold_local_train_denominator
- MASE convention verification pass: True

## 6. Provenance Distinction
- source_run_id and artifact_generation_commit are preserved from each source manifest.
- package_freeze_tag and package_freeze_commit are recorded separately from source-generation commits.
- HGBR provenance-B issue registry status: resolved=TRUE for all rows and no validity-critical row.

## 7. Reconciliation and Determinism
- controlled-rerun metric reconciliation pass: True
- checksum registry verification pass: True
- checksum coverage missing: []
- checksum coverage extra: []
- test_result: pass_56_of_56
- deterministic_result: pass

## 8. Remaining Limitations
1. This stage reconciles historical held-out predictions only; no prospective deployment validation is provided.
2. Remote tag availability for revision-origin may be incomplete; this is recorded as provenance metadata, not silently converted into computational pass/fail.
3. This stage excludes H5 and all alarm-policy analyses by design.

FINAL DECISION

A. Canonical H1/H3 reconciliation passed; lag-window sensitivity may begin.

TERMINAL SUMMARY

1. repository: /home/alrezshams/acs_tnout_ulsan_revision
2. branch: controlled-reruns-v1
3. starting commit: 045d265f54d29eb1212758307161072c25379541
4. authorized output directory: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/09_h1_h3_reconciliation
5. input checksum status: True
6. source decision status: {"ElasticNet": "A", "HGBR": "A", "Persistence": "A", "Ridge": "A"}
7. source-tag and remote-tag status: local={"corrected-protocol-v1": "2bbf5eba419b0650693f1e88326107bf9a346c68", "elasticnet-corrected-v1": "1db2f3472cd48274a5afa4010f446952d76c6042", "hgbr-corrected-v1": "4d16b4d513e4fd9aa7c1acd85d0b5894ce4d5d62", "persistence-corrected-v1": "41f184a29ff88aaf275f3365267b8c909ce7598a", "ridge-corrected-v1": "69cfb2bf8bc181dc1de1e09f0c7477a1a5de7585"} remote={"corrected-protocol-v1": true, "elasticnet-corrected-v1": true, "hgbr-corrected-v1": true, "persistence-corrected-v1": true, "ridge-corrected-v1": true}
8. H1 N and fold sizes: N=762 folds={1: 254, 2: 254, 3: 254}
9. H3 N and fold sizes: N=747 folds={1: 249, 2: 249, 3: 249}
10. event counts: {"1": {"events_tau15": 141, "events_tau16": 74, "events_tau17": 30, "prevalence_tau15": 0.18503937007874016, "prevalence_tau16": 0.09711286089238845, "prevalence_tau17": 0.03937007874015748}, "3": {"events_tau15": 134, "events_tau16": 71, "events_tau17": 29, "prevalence_tau15": 0.17938420348058903, "prevalence_tau16": 0.09504685408299866, "prevalence_tau17": 0.038821954484605084}}
11. models included: ['Persistence', 'Ridge', 'ElasticNet', 'HGBR']
12. models explicitly excluded: ['H5', 'BCR-TCN', 'TCN', 'point blend', 'HybridRank', 'fixed rank ensemble', 'alarm flags', 'top-k', 'classification metrics']
13. long and wide row counts: H1 long=3048 wide=762; H3 long=2988 wide=747
14. metric-reconciliation status: True
15. MASE-convention status: True
16. number of tests run and passed: pass_56_of_56
17. deterministic result: pass
18. source-preservation result: True
19. checksum result: file_ok=True coverage_ok=True
20. final decision: A
21. generated files: ["revision_2026/09_h1_h3_reconciliation/build_canonical_h1_h3_reconciliation.py", "revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_point_table2_source.csv", "revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_reconciliation_completion_report.md", "revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_reconciliation_manifest.json", "revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_source_registry.csv", "revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_table1_reconciliation.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_event_prevalence.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_key_audit.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_point_metrics_by_fold.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_point_metrics_pooled.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_predictions_long.csv", "revision_2026/09_h1_h3_reconciliation/h1/canonical_h1_predictions_wide.csv", "revision_2026/09_h1_h3_reconciliation/h1/h1_controlled_rerun_reconciliation.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_event_prevalence.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_key_audit.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_point_metrics_by_fold.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_point_metrics_pooled.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_predictions_long.csv", "revision_2026/09_h1_h3_reconciliation/h3/canonical_h3_predictions_wide.csv", "revision_2026/09_h1_h3_reconciliation/h3/h3_controlled_rerun_reconciliation.csv", "revision_2026/09_h1_h3_reconciliation/input_verification.json", "revision_2026/09_h1_h3_reconciliation/test_canonical_h1_h3_reconciliation.py"]
