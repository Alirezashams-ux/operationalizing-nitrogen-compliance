# Final Figures and Tables for Manuscript (ACS-ready)

## Main Text Tables

1. **Table 1. Leakage-safe point forecasting performance across horizons (H=1,3,5)**  
   Source: `results/tables/leaderboard_point_by_horizon.csv`

2. **Table 2. Exceedance prevalence and random-baseline recall by horizon/threshold**  
   Source: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table2_exceedance_prevalence_random_baseline.csv`

3. **Table 3. Alarm-budget metrics at H=5 (core operating points)**  
   Source: `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv`

4. **Table 4. Normalized H5 comparison on shared folds/windows (fold mean ± 95% CI)**  
   Source: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table4_H5_normalized_context_comparison.csv`

5. **Table 5. Key operating-point fold uncertainty (precision/recall/TP with 95% CI)**  
   Source: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv`

## Main Text Figures

1. **Figure 1. Study workflow and leakage-safe blocked-CV protocol**  
   (Conceptual schematic; prepare in manuscript graphics workflow)

2. **Figure 2. Cross-horizon point-performance comparison (MAE/RMSE trend by H)**  
   (Render from `results/tables/leaderboard_point_by_horizon.csv`)

3. **Figure 3. Budget–recall curves at H=5 (τ=15,16)**  
   Source: `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_panel_tau15_16_budget_recall.pdf`

4. **Figure 4. Fold-uncertainty bars at key operating points (recall & precision, 95% CI)**  
   Source: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig4_key_operating_points_uncertainty.pdf`

5. **Figure 5. Decision figure at H=5, τ=16: TP captured vs k (top models, fold 95% CI)**  
   Source: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig5_decision_curve_H5_tau16_tp_vs_k.pdf`

## Supporting Information (recommended)

- **Table S1. Full fold rows for key operating points**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_rows.csv`

- **Table S2. Figure/Table manifest with status and section mapping**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/paper_figures_tables_manifest.csv`

- **Figure S1 (optional). Single-threshold variants of Fig. 3 (τ=15 only or τ=16 only)**  
  `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_single_tau15_budget_recall.pdf`  
  `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_single_tau16_budget_recall.pdf`

## Notes on benchmark context normalization

- Normalized H5 context artifacts were regenerated using a shared fold/window context via run:  
  `results/hybrid_rank_v2_runs/20260220_131820_H5_hybrid_rank_v2`
- Use Table 4 when comparing hybrid vs non-hybrid H5 claims in main text to avoid apples-vs-oranges reviewer concerns.
