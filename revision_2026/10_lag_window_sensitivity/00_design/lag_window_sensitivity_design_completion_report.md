# Stage 2 Design-Gate Completion Report

## Gate Summary
- preflight_pass: True
- feature_lineage_established_pass: True
- common_date_feasibility_pass: True
- source_preservation_pass: True
- checksum_registry_pass: True
- checksum_coverage_pass: True
- test_result: pass_44_of_44
- deterministic_result: pass

## Context
- repository: /home/alrezshams/acs_tnout_ulsan_revision
- branch: controlled-reruns-v1
- starting commit: 15cc6f8414df3e60692700963afc296e7653236e
- Stage 1 tag canonical-h1-h3-reconciliation-v1 resolves to 15cc6f8414df3e60692700963afc296e7653236e

## Feature Lineage by Model/Horizon
- BCR-TCN_H5: v2_npz_context_only | feature_set_id=ulsan_H5_features_v2_npz_34col | feature_count=34 | used=True
- ElasticNet_H1: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H1_18col | feature_count=18 | used=True
- ElasticNet_H3: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H3_18col | feature_count=18 | used=True
- ElasticNet_H5: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H5_18col | feature_count=18 | used=True
- HGBR_H1: ordinary_npz | feature_set_id=hgbr_npz_ordinary_H1_18col | feature_count=18 | used=True
- HGBR_H3: v2_artifact_exists_not_used | feature_set_id=ulsan_H3_features_v2_npz_34col | feature_count=34 | used=False
- HGBR_H5: v2_npz | feature_set_id=hgbr_npz_v2_H5_34col | feature_count=34 | used=True
- Persistence_H1: direct_naive_context_only | feature_set_id=none_naive_lag_h | feature_count=0 | used=True
- Persistence_H3: direct_naive_context_only | feature_set_id=none_naive_lag_h | feature_count=0 | used=True
- Persistence_H5: direct_naive_context_only | feature_set_id=none_naive_lag_h | feature_count=0 | used=True
- Ridge_H1: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H1_18col | feature_count=18 | used=True
- Ridge_H3: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H3_18col | feature_count=18 | used=True
- Ridge_H5: ordinary_submitted_non_v2 | feature_set_id=submitted_main_linear_features_H5_18col | feature_count=18 | used=True

## Predeclared TN-Memory Definitions
- SHORT: TNout_lag1; TNout_roll7
- REFERENCE: TNout_lag1; TNout_roll14; TNout_roll7
- LONG: TNout_lag1; TNout_lag3; TNout_lag5; TNout_lag7; TNout_roll14; TNout_roll30; TNout_roll7

## Final-Reference Mapping
- HGBR_H1 -> REFERENCE
- HGBR_H3 -> REFERENCE
- HGBR_H5 -> LONG
- Ridge_H1 -> REFERENCE
- Ridge_H3 -> REFERENCE
- Ridge_H5 -> REFERENCE

## NPZ Inspection Status
- all_required_npz_present: True
- schema_ok: True
- hashes_recorded: True

## Training Retention Findings
- LONG H1 outer-train warm-up losses by fold={1: 16, 2: 16, 3: 16} (pooled=48); all canonical outer-test rows are preserved across SHORT/REFERENCE/LONG for H1/H3/H5.

FINAL DECISION

A. Feature lineage and predeclared lag-window sensitivity design passed; controlled sensitivity execution may be prepared.

Generated files:
- revision_2026/10_lag_window_sensitivity/00_design/build_lag_window_sensitivity_design.py
- revision_2026/10_lag_window_sensitivity/00_design/candidate_memory_configurations.csv
- revision_2026/10_lag_window_sensitivity/00_design/common_date_feasibility.csv
- revision_2026/10_lag_window_sensitivity/00_design/feature_artifact_registry.csv
- revision_2026/10_lag_window_sensitivity/00_design/feature_lineage_audit.csv
- revision_2026/10_lag_window_sensitivity/00_design/feature_lineage_audit.md
- revision_2026/10_lag_window_sensitivity/00_design/input_verification.json
- revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design.json
- revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design.md
- revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design_completion_report.md
- revision_2026/10_lag_window_sensitivity/00_design/lag_window_sensitivity_design_manifest.json
- revision_2026/10_lag_window_sensitivity/00_design/model_horizon_reference_registry.csv
- revision_2026/10_lag_window_sensitivity/00_design/test_lag_window_sensitivity_design.py
- revision_2026/10_lag_window_sensitivity/00_design/training_retention_audit.csv
