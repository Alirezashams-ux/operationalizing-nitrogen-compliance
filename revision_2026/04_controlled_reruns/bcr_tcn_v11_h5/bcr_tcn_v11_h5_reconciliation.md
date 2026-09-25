# BCR-TCN v1.1 H5 Reconciliation

## 1. Scope
- Model: BCR-TCN v1.1
- Horizon: H=5 only
- Corrected run_id: bcr_tcn_v11_h5_corrected_4d16b4d513e4_6ed147cd_f83baf57
- Split source: corrected_split_assignment.csv (locked)

## 2. Context Separation
- submitted_759_date_context: historical TCN/BCR-TCN row not directly traceable in available H5 alarm table artifacts.
- submitted_747_v11_context: alarm_budget_bcr_tcn_v11_H5_v2.csv and bcr_tcn_v11_H5_v2_preds.csv.
- submitted_735_shared_normalized_context: hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2/tables/alarm_budget_metrics_H5.csv.
- corrected_747_canonical_context: controlled rerun with locked corrected split and five-day boundary purge.

## 3. Key Recall@5% tau16 Comparison
| context | submitted_value | corrected_value | delta | status |
|---|---:|---:|---:|---|
| submitted_747_v11_context | 0.211267605634 | 0.014084507042 | -0.197183098592 | ok |
| submitted_747_hybrid_context | 0.211267605634 | 0.014084507042 | -0.197183098592 | ok |
| submitted_735_shared_normalized_context | 0.214285714286 | 0.014084507042 | -0.200201207243 | ok |
| submitted_759_date_context | NA | 0.014084507042 | NA | missing_historical_artifact |
| corrected_747_canonical_context | NA | 0.014084507042 | NA | reference |

## 4. Pooled vs Fold-Averaged Point Metrics
- pooled_RMSE_direct = 12.786255575706; fold_mean_RMSE = 12.779040110715; delta = +0.007215464992.
- Pooled metrics are computed directly from all held-out rows, not by averaging fold RMSE as the primary value.

## 5. Purge and Determinism Effects
- Outer boundary purge removes 5 boundary rows per fold (15 total from outer-training candidate rows).
- Test horizon count is fixed at 747 (249 per fold), versus historical 759 and shared-normalized 735 contexts.
- Deterministic execution uses seed 42 with torch.use_deterministic_algorithms(True) on CPU.

## 6. Missing Historical Artifact Notes
- A direct submitted TCN/BCR-TCN H5 row in the 759-date alarm table context was not found in available project artifacts; this is recorded as missing_historical_artifact in the comparison CSV.

## 7. Boundary Audit Summary
- Fold 1: inner_boundary_pass=True, outer_boundary_pass=True, inner_purge_days=5, outer_purge_days=5.
- Fold 2: inner_boundary_pass=True, outer_boundary_pass=True, inner_purge_days=5, outer_purge_days=5.
- Fold 3: inner_boundary_pass=True, outer_boundary_pass=True, inner_purge_days=5, outer_purge_days=5.
