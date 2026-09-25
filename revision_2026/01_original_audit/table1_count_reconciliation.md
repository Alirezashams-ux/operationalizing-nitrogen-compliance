# Table 1 Count Reconciliation

## Candidate Identity Check
- Submitted counts (H1/H3/H5 N and tau-event totals) match Table2_exceedance_prevalence_random_baseline.csv exactly.
- Point-summary/leaderboard candidates do not contain tau-event count schema and therefore cannot directly represent submitted count table.
- Conclusion: submitted Main Table 1 count block is an exceedance prevalence/random-baseline table lineage (not the 3-row point-summary table).

## 759 versus 747 Date-Level Delta
- H3: 12 dates exist in 759-row non-v2 predictions but not in canonical 747-row SI split.
- H5: 12 dates exist in 759-row non-v2 predictions but not in canonical 747-row SI split.
- Cause: non-v2 feature builders include earlier test-window rows than v2 canonical split definitions.

- Full reconciliation data: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/01_original_audit/table1_count_reconciliation.csv