# Results and Discussion Complete Report (Manuscript-Ready)

Date: 2026-03-27

## 1) What was reviewed exactly
- Source narrative reviewed: `results/methods_figure_csvs/Results and Discussion..docx`
- Results artifacts cross-checked from:
  - `results/tables/`
  - `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/`
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/`
  - `reports/paper_figures_tables_final_list.md`
  - `results/tables/discussion_key_results.csv`

This report is aligned to what has already been produced in your project and avoids adding non-existent outputs.

## 2) Confirmed core results to report

### 2.1 Data reality check and exceedance context
From `Table2_exceedance_prevalence_random_baseline.csv`:
- H1 prevalence:
  - tau=15: 141/762 = 0.1850 (18.50%)
  - tau=16: 74/762 = 0.0971 (9.71%)
  - tau=17: 30/762 = 0.0394 (3.94%)
- H3 prevalence:
  - tau=15: 140/759 = 0.1845 (18.45%)
  - tau=16: 73/759 = 0.0962 (9.62%)
  - tau=17: 29/759 = 0.0382 (3.82%)
- H5 prevalence:
  - tau=15: 140/759 = 0.1845 (18.45%)
  - tau=16: 73/759 = 0.0962 (9.62%)
  - tau=17: 29/759 = 0.0382 (3.82%)

Interpretation to keep: prevalence is stable by horizon and decreases sharply with threshold, which justifies budgeted ranking evaluation.

### 2.2 Point-forecast performance (what to claim)
From `results/tables/leaderboard_point_by_horizon.csv` (main linear context):
- H1 best MAE: ElasticNet (a=0.1, l1=0.5), MAE=1.5373 mg/L
- H3 best MAE: ElasticNet (a=0.1, l1=0.5), MAE=1.7191 mg/L
- H5 best MAE: ElasticNet (a=1.0, l1=0.2), MAE=1.7944 mg/L

From guarded H5 run context in same table:
- Hybrid point blend MAE=1.7127 mg/L (best in that guarded H5 run family)

Interpretation to keep: point error rises with horizon; regularized linear models remain strong; point-optimal model is not always alarm-optimal.

### 2.3 Alarm-budget utility (operational novelty)
From `results/tables/discussion_key_results.csv` and Table 3 artifacts:
- At r=0.05 random baseline recall is 0.05 by design.
- Achieved recalls commonly in 0.14 to 0.27 range at tight budgets, corresponding to about 3x to 5x enrichment vs random.

This is the central operational novelty and should be emphasized in Discussion and Conclusion.

### 2.4 Operating-point dependent model preference
Supported by `results/tables/leaderboard_alarm_winners.csv`, `results/tables/model_alarm_performance_full.csv`, and Table 3:
- H1: persistence is often strongest for immediate triage.
- H3: ElasticNet variants frequently lead under practical budgets.
- H5: winner depends on (tau, r), and tie/near-tie behavior is common at top-k cutoffs.

### 2.5 Guarded hybrid finding at H5
From `results/tables/discussion_key_results.csv` (H5_v2_guarded rows):
- tau=16, r=0.10:
  - hybrid_rank_v2 recall=0.3662 (TP=26/71)
  - hybrid_paper_fixed recall=0.2958 (TP=21/71)
  - persistence recall=0.3239 (TP=23/71)

Claim to make: guarded rank policy improves exceedance capture at fixed alarm count for key extended-horizon operating points.

### 2.6 Uncertainty and decision interpretation
From `Table5_key_operating_points_fold_uncertainty.csv` and `Fig5_decision_curve_H5_tau16_data.csv`:
- Example key point (H5, tau=16, r=0.10):
  - hybrid_rank_v2 precision_mean=0.2933 with precision_ci95=0.2493
  - hybrid_rank_v2 recall_mean=0.3177 with recall_ci95=0.0964
- Decision curves show model separation is most meaningful in low-to-mid k ranges; convergence increases at larger k.

Discussion point: uncertainty overlap means avoid absolute winner language outside specified operating points.

## 3) Achievements and key findings to highlight
1. Built and validated a leakage-safe blocked-CV early-warning framework across 1, 3, and 5 day horizons.
2. Demonstrated that operational utility under bounded attention differs from point-error rankings.
3. Quantified practical enrichment over random selection under strict alarm budgets.
4. Showed operating-point dependent model superiority instead of one-model-fits-all conclusions.
5. Added fold-level uncertainty and TP-vs-k decision curves for transparent operational claims.
6. Demonstrated a guarded hybrid policy that can recover additional exceedances at fixed alarm load in extended-horizon use.

## 4) Exact tables to use in Results and Discussion

### Table 1 (Point forecasting across horizons)
- Use CSV: `results/tables/leaderboard_point_by_horizon.csv`

### Table 2 (Exceedance prevalence + random baseline)
- Use CSV: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table2_exceedance_prevalence_random_baseline.csv`

### Table 3 (H5 alarm-budget core operating points)
- Use CSV: `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv`

### Table 4 (Normalized H5 comparison in shared context)
- Use CSV: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table4_H5_normalized_context_comparison.csv`

### Table 5 (Fold uncertainty at key operating points)
- Use CSV: `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv`
- Optional companion with row-level detail:
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_rows.csv`

## 5) Figures 3, 4, 5, and 6 (with exact CSV inputs)

### Figure 3 (Budget-recall panel at H5 for tau=15,16)
- Primary figure file:
  - `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_panel_tau15_16_budget_recall.pdf`
- Exact CSV input:
  - `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv`
  - (raw equivalent) `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/tables/table3_alarm_budget_H5_v2_long.csv`

### Figure 4 (Key operating-point uncertainty bars)
- Primary figure file:
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig4_key_operating_points_uncertainty.pdf`
- Exact CSV input:
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv`

### Figure 5 (Decision curve TP captured vs alarm count k at H5 tau=16)
- Primary figure file:
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig5_decision_curve_H5_tau16_tp_vs_k.pdf`
- Exact CSV input:
  - `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Fig5_decision_curve_H5_tau16_data.csv`

### Figure 6 (Recommended addition: Silence-Safety Frontier)
Rationale: You asked for Figures 3-6. Current main manifest has up to Figure 5. A fully generated and manuscript-ready additional figure already exists and is strongly aligned with Discussion framing.

- Recommended figure file:
  - `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_silence_safety_frontier_tau15_16.pdf`
- Exact CSV input:
  - `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/tables/table3_alarm_budget_H5_v2_long.csv`
- Suggested caption:
  - "Silence-safety frontier at H=5 for tau=15 and 16 mg/L, showing missed-event fraction (1-recall) versus quiet-days fraction (1-r) across models."

## 6) Discussion text guidance (what should be made explicit)

### 6.1 What to emphasize strongly
1. Operational evaluation objective is different from regression objective.
2. Under bounded attention, top-k ranking quality near cutoff is the key practical property.
3. Gains should be reported as workload-linked benefits (TP captured at k alarms), not only metric deltas.

### 6.2 What to phrase cautiously
1. Avoid universal winner statements at H5.
2. State operating point for every comparative claim (horizon, tau, r, and k).
3. Pair precision/recall with TP counts and uncertainty intervals.

### 6.3 Limitations to keep in Discussion
1. Single-site primary dataset for main analysis.
2. Fold-level variability at longer horizons can alter local rankings.
3. Top-k thresholding can produce ties/near-ties in TP counts across models.

### 6.4 Strong close-out narrative
Use this structure in the final discussion paragraphs:
- Paragraph 1: Data reality + why bounded-attention evaluation is needed.
- Paragraph 2: Point-forecast findings by horizon.
- Paragraph 3: Alarm-budget findings and enrichment over random.
- Paragraph 4: Extended-horizon uncertainty and decision-curve interpretation.
- Paragraph 5: Practical deployment implication and external-validation roadmap.

## 7) Exact CSV package checklist (copy into manuscript workflow)
- `results/tables/leaderboard_point_by_horizon.csv`
- `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table2_exceedance_prevalence_random_baseline.csv`
- `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv`
- `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table4_H5_normalized_context_comparison.csv`
- `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv`
- `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Fig5_decision_curve_H5_tau16_data.csv`
- `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/tables/table3_alarm_budget_H5_v2_long.csv`

## 8) Final recommendation on figure numbering
For your requested numbering in Results and Discussion:
- Figure 3: Budget-recall panel (H5, tau 15 and 16)
- Figure 4: Fold uncertainty bars
- Figure 5: TP-vs-k decision curve
- Figure 6: Silence-safety frontier (recommended addition from existing generated artifact)

This gives a coherent progression: performance under budget -> uncertainty of claims -> operational decision utility -> risk-silence tradeoff view.
