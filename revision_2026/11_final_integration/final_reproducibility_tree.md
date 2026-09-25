# Final Reproducibility Tree

## raw_dataset_lock
- directory: revision_2026/00_provenance/recovered_inputs/figure2
- role: raw dataset lock
- manifest: revision_2026/03_corrected_protocol/canonical_dataset_lock.json
- checksum registry: revision_2026/03_corrected_protocol/corrected_protocol_v1.sha256
- builder: revision_2026/03_corrected_protocol/build_purged_nested_splits.py
- standalone test: revision_2026/03_corrected_protocol/test_purged_nested_splits.py
- test_result: captured upstream
- deterministic_result: captured upstream
- tag: corrected-protocol-v1
- peeled commit: 2bbf5eba419b0650693f1e88326107bf9a346c68
- child outputs: corrected_protocol, controlled_reruns
- limitations: Raw lock only; not a model-evaluation artifact.

## corrected_protocol
- directory: revision_2026/03_corrected_protocol
- role: canonical corrected protocol and split freeze
- manifest: 
- checksum registry: revision_2026/03_corrected_protocol/corrected_protocol_v1.sha256
- builder: revision_2026/03_corrected_protocol/build_purged_nested_splits.py
- standalone test: revision_2026/03_corrected_protocol/test_purged_nested_splits.py
- test_result: 
- deterministic_result: 
- tag: corrected-protocol-v1
- peeled commit: 2bbf5eba419b0650693f1e88326107bf9a346c68
- child outputs: persistence_corrected, ridge_corrected, elasticnet_corrected, hgbr_corrected, bcr_tcn_v11_h5_corrected
- limitations: Protocol freeze only.

## persistence_corrected
- directory: revision_2026/04_controlled_reruns/persistence
- role: controlled persistence rerun inputs
- manifest: revision_2026/04_controlled_reruns/persistence/persistence_run_manifest.json
- checksum registry: 
- builder: 
- standalone test: revision_2026/04_controlled_reruns/persistence/test_corrected_persistence.py
- test_result: pass_12_of_12
- deterministic_result: pass
- tag: persistence-corrected-v1
- peeled commit: 41f184a29ff88aaf275f3365267b8c909ce7598a
- child outputs: canonical_h1_h3_reconciliation, canonical_h5_assembly
- limitations: Point model rerun source.

## ridge_corrected
- directory: revision_2026/04_controlled_reruns/ridge
- role: controlled ridge rerun inputs
- manifest: revision_2026/04_controlled_reruns/ridge/ridge_run_manifest.json
- checksum registry: 
- builder: 
- standalone test: revision_2026/04_controlled_reruns/ridge/test_corrected_ridge.py
- test_result: pass_15_of_15
- deterministic_result: pass
- tag: ridge-corrected-v1
- peeled commit: 69cfb2bf8bc181dc1de1e09f0c7477a1a5de7585
- child outputs: canonical_h1_h3_reconciliation, canonical_h5_assembly, lag_window_sensitivity_execution
- limitations: Point model rerun source.

## elasticnet_corrected
- directory: revision_2026/04_controlled_reruns/elasticnet
- role: controlled elasticnet rerun inputs
- manifest: revision_2026/04_controlled_reruns/elasticnet/elasticnet_run_manifest.json
- checksum registry: 
- builder: 
- standalone test: revision_2026/04_controlled_reruns/elasticnet/test_corrected_elasticnet.py
- test_result: pass_15_of_15
- deterministic_result: pass
- tag: elasticnet-corrected-v1
- peeled commit: 1db2f3472cd48274a5afa4010f446952d76c6042
- child outputs: canonical_h1_h3_reconciliation, canonical_h5_assembly
- limitations: Point model rerun source.

## hgbr_corrected
- directory: revision_2026/04_controlled_reruns/hgbr
- role: controlled hgbr rerun inputs
- manifest: revision_2026/04_controlled_reruns/hgbr/hgbr_run_manifest.json
- checksum registry: 
- builder: 
- standalone test: revision_2026/04_controlled_reruns/hgbr/test_corrected_hgbr.py
- test_result: pass_20_of_20
- deterministic_result: pass
- tag: hgbr-corrected-v1
- peeled commit: 4d16b4d513e4fd9aa7c1acd85d0b5894ce4d5d62
- child outputs: canonical_h1_h3_reconciliation, canonical_h5_assembly, lag_window_sensitivity_execution
- limitations: Point model rerun source.

## bcr_tcn_v11_h5_corrected
- directory: revision_2026/04_controlled_reruns/bcr_tcn_v11_h5
- role: controlled bcr-tcn h5 rerun inputs
- manifest: revision_2026/04_controlled_reruns/bcr_tcn_v11_h5/bcr_tcn_v11_h5_run_manifest.json
- checksum registry: 
- builder: 
- standalone test: revision_2026/04_controlled_reruns/bcr_tcn_v11_h5/test_corrected_bcr_tcn_v11_h5.py
- test_result: pass_25_of_25
- deterministic_result: pass
- tag: bcr-tcn-v11-h5-corrected-v1
- peeled commit: 596f1d05ff477c317b9712d1233b64c6443e8c32
- child outputs: canonical_h5_assembly
- limitations: H5-only model source.

## canonical_h5_assembly
- directory: revision_2026/05_canonical_predictions/h5_cross_model
- role: canonical h5 predictions and pooled metrics
- manifest: revision_2026/05_canonical_predictions/h5_cross_model/canonical_h5_assembly_manifest.json
- checksum registry: revision_2026/05_canonical_predictions/h5_cross_model/canonical_h5_checksums.sha256
- builder: revision_2026/05_canonical_predictions/h5_cross_model/build_canonical_h5_predictions.py
- standalone test: revision_2026/05_canonical_predictions/h5_cross_model/test_canonical_h5_assembly.py
- test_result: pass_21_of_21
- deterministic_result: 
- tag: canonical-h5-assembly-v1
- peeled commit: ea6e72f891afc7a62a9ca8206242f946047edbab
- child outputs: fixed_hybridrank_scores, retrospective_alarm_budget, sequential_past_only
- limitations: Canonical H5 rows and metrics.

## fully_nested_hybridrank_feasibility
- directory: revision_2026/06_corrected_hybridrank/h5/fully_nested
- role: hybridrank feasibility diagnostic branch
- manifest: revision_2026/06_corrected_hybridrank/h5/fully_nested/fully_nested_hybridrank_h5_manifest.json
- checksum registry: revision_2026/06_corrected_hybridrank/h5/fully_nested/fully_nested_hybridrank_h5_checksums.sha256
- builder: revision_2026/06_corrected_hybridrank/h5/fully_nested/build_fully_nested_hybridrank_h5.py
- standalone test: revision_2026/06_corrected_hybridrank/h5/fully_nested/test_fully_nested_hybridrank_h5.py
- test_result: pass_guard_block_28_of_28
- deterministic_result: pass
- tag: fully-nested-hybridrank-h5-feasibility-v1
- peeled commit: ea738f9bca32b3bf5648a860c35cd84fbe4e205b
- child outputs: fixed_hybridrank_scores
- limitations: Diagnostic infeasibility/guarded non-selection branch only.

## fixed_hybridrank_scores
- directory: revision_2026/06_corrected_hybridrank/h5/fixed_policy
- role: fixed rank ensemble scores for rank/alarm stages
- manifest: revision_2026/06_corrected_hybridrank/h5/fixed_policy/fixed_hybridrank_h5_manifest.json
- checksum registry: revision_2026/06_corrected_hybridrank/h5/fixed_policy/fixed_hybridrank_h5_checksums.sha256
- builder: revision_2026/06_corrected_hybridrank/h5/fixed_policy/build_fixed_hybridrank_h5_scores.py
- standalone test: revision_2026/06_corrected_hybridrank/h5/fixed_policy/test_fixed_hybridrank_h5_scores.py
- test_result: pass_67_of_67
- deterministic_result: pass
- tag: fixed-hybridrank-h5-scores-v1
- peeled commit: 1a7fa526c26397ac50810bca24f57d3765aa4af4
- child outputs: retrospective_alarm_budget, sequential_past_only, sequential_quota_enforced
- limitations: Fixed rank ensemble for alarm evaluations only.

## retrospective_alarm_budget
- directory: revision_2026/07_retrospective_alarm_budget/h5
- role: retrospective fold-local top-k alarm-budget evaluation
- manifest: revision_2026/07_retrospective_alarm_budget/h5/retrospective_h5_alarm_budget_manifest.json
- checksum registry: revision_2026/07_retrospective_alarm_budget/h5/retrospective_h5_alarm_budget_checksums.sha256
- builder: revision_2026/07_retrospective_alarm_budget/h5/build_retrospective_h5_alarm_budget.py
- standalone test: revision_2026/07_retrospective_alarm_budget/h5/test_retrospective_h5_alarm_budget.py
- test_result: pass_66_of_66
- deterministic_result: pass
- tag: retrospective-h5-alarm-budget-v1
- peeled commit: 140b8193f31425f7506fae4f36987246eedab729
- child outputs: final_integration
- limitations: Retrospective offline top-k only.

## sequential_past_only
- directory: revision_2026/08_sequential_alarm_policy/h5
- role: historical past-only sequential threshold simulation
- manifest: revision_2026/08_sequential_alarm_policy/h5/sequential_h5_alarm_policy_manifest.json
- checksum registry: revision_2026/08_sequential_alarm_policy/h5/sequential_h5_alarm_policy_checksums.sha256
- builder: revision_2026/08_sequential_alarm_policy/h5/build_sequential_h5_alarm_policy.py
- standalone test: revision_2026/08_sequential_alarm_policy/h5/test_sequential_h5_alarm_policy.py
- test_result: pass_65_of_65
- deterministic_result: pass
- tag: sequential-h5-past-quantile-v1
- peeled commit: e98a21d77033d746e7b77f348af8dc34be0768fd
- child outputs: sequential_quota_enforced, final_integration
- limitations: Historical past-only simulation only.

## sequential_quota_enforced
- directory: revision_2026/08_sequential_alarm_policy/h5/quota_enforced
- role: historical quota-enforced sequential simulation
- manifest: revision_2026/08_sequential_alarm_policy/h5/quota_enforced/quota_enforced_h5_manifest.json
- checksum registry: revision_2026/08_sequential_alarm_policy/h5/quota_enforced/quota_enforced_h5_checksums.sha256
- builder: revision_2026/08_sequential_alarm_policy/h5/quota_enforced/build_quota_enforced_h5_alarm_policy.py
- standalone test: revision_2026/08_sequential_alarm_policy/h5/quota_enforced/test_quota_enforced_h5_alarm_policy.py
- test_result: pass_25_of_25
- deterministic_result: pass
- tag: quota-enforced-sequential-h5-v1
- peeled commit: 045d265f54d29eb1212758307161072c25379541
- child outputs: final_integration
- limitations: Historical quota-enforced simulation only.

## canonical_h1_h3_reconciliation
- directory: revision_2026/09_h1_h3_reconciliation
- role: canonical h1/h3 corrected predictions and metrics
- manifest: revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_reconciliation_manifest.json
- checksum registry: revision_2026/09_h1_h3_reconciliation/canonical_h1_h3_reconciliation_checksums.sha256
- builder: revision_2026/09_h1_h3_reconciliation/build_canonical_h1_h3_reconciliation.py
- standalone test: revision_2026/09_h1_h3_reconciliation/test_canonical_h1_h3_reconciliation.py
- test_result: pass_56_of_56
- deterministic_result: pass
- tag: canonical-h1-h3-reconciliation-v1
- peeled commit: 15cc6f8414df3e60692700963afc296e7653236e
- child outputs: lag_window_sensitivity_design, final_integration
- limitations: Canonical H1/H3 source.

## lag_window_sensitivity_design
- directory: revision_2026/10_lag_window_sensitivity/00_design
- role: lag-window sensitivity design freeze
- manifest: revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design_manifest.json
- checksum registry: revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design_checksums.sha256
- builder: revision_2026/10_lag_window_sensitivity/00_design/build_lag_window_sensitivity_design.py
- standalone test: revision_2026/10_lag_window_sensitivity/00_design/test_lag_window_sensitivity_design.py
- test_result: pass_44_of_44
- deterministic_result: pass
- tag: lag-window-sensitivity-design-v1
- peeled commit: dee06bcfeca0c3c224897a2a92261c13f4fea26e
- child outputs: lag_window_sensitivity_execution
- limitations: Design freeze.

## lag_window_sensitivity_execution
- directory: revision_2026/10_lag_window_sensitivity/01_execution
- role: lag-window sensitivity execution freeze
- manifest: revision_2026/10_lag_window_sensitivity/01_execution/lag_window_sensitivity_execution_manifest.json
- checksum registry: revision_2026/10_lag_window_sensitivity/01_execution/lag_window_sensitivity_execution_checksums.sha256
- builder: revision_2026/10_lag_window_sensitivity/01_execution/run_lag_window_sensitivity_execution.py
- standalone test: revision_2026/10_lag_window_sensitivity/01_execution/test_lag_window_sensitivity_execution.py
- test_result: pass_75_of_75
- deterministic_result: pass
- tag: lag-window-sensitivity-execution-v1
- peeled commit: f61278b3e3fc2cb0b69564b346a4bf32e037b4fb
- child outputs: final_integration
- limitations: Execution freeze for lag sensitivity outputs.

## final_integration
- directory: revision_2026/11_final_integration
- role: stage3 final computational integration
- manifest: revision_2026/11_final_integration/final_computational_integration_manifest.json
- checksum registry: revision_2026/11_final_integration/final_computational_integration_checksums.sha256
- builder: revision_2026/11_final_integration/build_final_computational_integration.py
- standalone test: revision_2026/11_final_integration/test_final_computational_integration.py
- test_result: pending
- deterministic_result: pending
- tag: none
- peeled commit: f61278b3e3fc2cb0b69564b346a4bf32e037b4fb
- child outputs: final tables, final figures, final registries, final audits
- limitations: Computational package only; manuscript editing performed separately.
