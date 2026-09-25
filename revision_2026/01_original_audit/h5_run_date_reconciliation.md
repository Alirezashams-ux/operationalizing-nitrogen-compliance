# H5 Run Date Reconciliation

## Run-level Comparison
- Compared runs: 20260219_195429_H5_hybrid_rank_v2 and 20260220_131820_H5_hybrid_rank_v2.
- Both runs use policy_guarded objective and inner-join alignment across base model prediction files.
- 20260219 run uses v2 base files and preserves canonical 747-date H5 test set (249/249/249 by fold).
- 20260220 run uses non-v2 ENet/Persistence/HGBR files and reduces intersection to 735 rows (241/245/249 by fold).

## N=747 versus N=735
- Exact excluded-date count: 12
- Excluded dates (present in 747 and absent in 735):
  - 2022-06-08
  - 2022-06-09
  - 2022-06-10
  - 2022-06-11
  - 2022-06-12
  - 2022-06-13
  - 2022-06-14
  - 2022-06-15
  - 2023-02-16
  - 2023-02-17
  - 2023-02-18
  - 2023-02-19
- Exclusion mechanism: inner merge on (fold,date) dropped rows missing in at least one 20260220 base prediction file.
- Normalization, purge, or embargo removals: none detected in script path.

## File-level Metadata
- Full per-file reconciliation table: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/01_original_audit/h5_run_date_reconciliation.csv