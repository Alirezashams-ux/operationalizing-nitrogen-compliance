# Table 3 Alarm-Rule Reconciliation

## k-rule diagnostic
- Submitted rows align with global-concatenated k = ceil(r * total N).
- Tie rows where global equals per-fold are marked as global_concatenated_tie_with_per_fold based on script behavior and non-tie rows in same table.
- Disambiguating rows include H3 at r=0.10 (k=76 global, not 78 per-fold with 253-size folds) and H5 at r=0.05 (k=38 global, not 39 per-fold).

- Full row-level reconciliation data: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/01_original_audit/table3_alarm_rule_reconciliation.csv