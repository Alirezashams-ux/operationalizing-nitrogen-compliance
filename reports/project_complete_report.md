# Complete Project Report: Leakage-Safe, Alarm-Budget TNout Early Warning Framework

**Project**: Operationalizing Nitrogen Compliance in WWTPs (Ulsan main study + Seoul external robustness in SI)  
**Prepared on**: 2026-03-04  
**Primary manuscript source used**: `reports/Manuscript.docx` (clean extraction: `reports/_tmp_manuscript_docx_clean.txt`)  
**Cross-check sources**: `reports/*.md`, `src/*.py`, `results/tables/*.csv`, `results/paper_artifacts/*`

---

## 1) Problem Definition

Wastewater treatment plants (WWTPs) must prevent effluent TN exceedances, but operators can only investigate a limited number of alarms each day. Most ML studies optimize average error (MAE/RMSE), while real operations require **ranking a small top-k list** under bounded attention and alarm-fatigue constraints.

This project addresses the practical question:

> If operators can act on only 5–10% of days, how many true exceedance days can a model capture at fixed workload?

The technical risk being addressed is **optimistic bias from temporal leakage**, especially in small daily datasets.

---

## 2) Aims and Objectives

### 2.1 Main aim
Develop a leakage-safe, decision-aligned framework for TNout early warning that is scientifically valid and operationally deployable.

### 2.2 Specific objectives
1. Build leakage-safe horizon-specific prediction pipelines for `H = {1,3,5}` days.
2. Benchmark baseline, linear, tree, deep, and hybrid models under blocked time-series CV.
3. Evaluate both point-forecast performance (MAE/RMSE/MASE) and alarm-budget utility (precision/recall at fixed `k`).
4. Introduce and evaluate guarded rank-ensemble policy for long-horizon bounded-attention decisions.
5. Produce manuscript-ready and SI-ready artifacts (tables/figures + uncertainty + decision curves + external transferability check).

---

## 3) Novelty and Scientific Contribution

### 3.1 Core novelty
- Combines **strict leakage control** with **alarm-budget decision evaluation** in one reproducible framework.
- Moves model selection from “best global MAE” to “best top-k exceedance capture under real staffing constraints.”

### 3.2 Methodological novelty
- Policy-specific and guarded rank fusion (`hybrid_rank_v2`) tuned on training folds only.
- Explicit precision floor and guarded fallback at conservative budgets (`r <= 0.05`) to avoid degradation where false alarms are most costly.
- Dual reporting: point metrics + decision metrics + fold-level uncertainty + decision curves.

### 3.3 Practical novelty
- Outputs are directly interpretable for operations as: “with `k` alarms, how many exceedances are captured?”
- Provides publication-grade artifact traceability (`run_config.json`, timestamped folders, normalized benchmark context tables).

---

## 4) Research Gaps Filled

This work closes the following gaps in typical WWTP forecasting studies:

1. **Leakage transparency gap**  
   Filled by strict temporal boundary in feature engineering and fold-wise preprocessing.

2. **Decision-metric gap**  
   Filled by top-k alarm-budget precision/recall at operational thresholds (`tau = 15/16/17 mg L^-1`).

3. **Bounded-attention policy gap**  
   Filled by guarded hybrid rank policy that preserves low-budget behavior while improving selected higher-budget operating points.

4. **Uncertainty reporting gap**  
   Filled by fold-level CI reporting (Table 5, Figure 4).

5. **Manuscript reproducibility gap**  
   Filled by explicit table/figure manifests and source-file mapping to generated artifacts.

---

## 5) Methodology (Process Form)

## 5.1 End-to-end pipeline

1. **Raw data ingestion**  
   Ulsan daily operational + meteorological data (`data/raw/Ulsan_Yongsan.csv`), plus Seoul external set for SI (`data/raw/Seoul_Tancheon_1.csv`).

2. **Leakage-safe feature construction**  
   Horizon-specific target shifting and feature engineering (`src/build_ulsan_npz_v2.py`):
   - autoregressive memory (lags, rolling means)
   - influent/load proxies
   - weather aggregates
   - seasonality terms
   - interactions
   Output: `features/ulsan_H{1,3,5}_features_v2.npz`.

3. **Chronological evaluation protocol**  
   Blocked `TimeSeriesSplit` (3 splits), no shuffling, train-only scaling/preprocessing.

4. **Model training and prediction generation**
   - Linear baselines (`ElasticNet`, `Ridge`, persistence)
   - Nonlinear (`HGBR`, Optuna tuning)
   - Deep sequence (`BCR-TCN v1.1`)
   - Hybrid point blend + hybrid rank ensemble (v2 guarded)
   Outputs in `results/predictions/`, `results/metrics/`, `results/hybrid_rank_v2_runs/`.

5. **Decision-layer evaluation**  
   For each `(H, tau, r)`: compute top-`k = ceil(r * N_test)` alarms and evaluate precision/recall/TP.

6. **Manuscript artifact generation**  
   Table/Figure production scripts under `src/paper_*` and uncertainty/decision artifacts under `results/paper_artifacts/`.

7. **SI transferability assessment**  
   External Seoul validation with point/alarm metrics and transferability figure (`src/si_make_seoul_external_eval.py`).

---

## 6) Complete Explanation of Models and Justification

## 6.1 Persistence baseline
- Rule-based carry-forward predictor.
- **Why included**: strong short-horizon operational baseline in autocorrelated daily process data.
- **Observed role**: dominant early-warner at `H=1` for multiple operating points.

## 6.2 ElasticNet / Ridge linear family
- Regularized linear models with fold-safe scaling.
- **Why included**: robust under small data, interpretable, low variance, strong generalization.
- **Observed role**: strongest and most stable point forecasting across horizons in main linear runs.

## 6.3 HGBR (Optuna-tuned)
- Histogram Gradient Boosting Regressor with tuned hyperparameters.
- **Why included**: captures nonlinearities/interactions missed by linear models.
- **Observed role**: competitive in selected alarm and external-transfer points, but context dependent.

## 6.4 BCR-TCN v1.1
- Causal dilated temporal convolutions + multi-head outputs (regression + threshold logits).
- Composite objective (Smooth L1 + weighted BCE + pairwise ranking emphasis near decision threshold).
- **Why included**: designed for temporal dependence and ranking behavior near operational thresholds.
- **Observed role**: useful risk component for hybrid ranking, not always the best standalone model.

## 6.5 Hybrid point predictor
- Convex blend of point predictors (trained on train folds only by MAE minimization).
- **Why included**: stabilize long-horizon point forecasts by combining complementary biases.
- **Observed role**: best MAE within guarded H5-v2 context (`MAE=1.713`).

## 6.6 Hybrid rank ensemble (paper-fixed and v2 guarded)
- Rank-normalized fusion of TCN + regression-based scores.
- Policy-guarded mode: conservative fallback at low budgets; tuned policy at higher budgets.
- **Why included**: direct optimization for top-k decision utility under bounded attention.
- **Observed role**: preserves low-budget behavior and improves selected high-budget operating points at `H=5`.

---

## 7) Results and Achievements

## 7.1 Main point-forecast achievements (leakage-safe)
From `results/tables/leaderboard_point_by_horizon.csv`:

- `H=1`: best MAE = **1.537** (`enet_a0.1_l0.5`)
- `H=3`: best MAE = **1.719** (`enet_a0.1_l0.5`)
- `H=5`: best MAE = **1.794** (`enet_a1.0_l0.2`)

Key achievement: robust linear baseline outperforms persistence for longer horizons while staying simple and deployable.

## 7.2 Alarm-budget achievements (main context)
From `results/tables/leaderboard_alarm_winners.csv`:

- `H=1`: persistence dominates key `tau=15,16` operating points.
- `H=3`: ElasticNet variants dominate major operating points.
- `H=5`: winners are operating-point dependent; ties and near-ties appear near top-k boundaries.

## 7.3 Guarded-hybrid achievements (H5 targeted context)
From `results/tables/discussion_key_results.csv` and `leaderboard_alarm_by_operating_point.csv`:

- At `tau=15, r=0.10`: `hybrid_rank_v2`/ENet context achieves recall **0.246** vs paper-fixed **0.201** (same `k=75`; +6 TP vs paper-fixed in corresponding compare table context).
- At `tau=16, r=0.10`: `hybrid_rank_v2` achieves recall **0.366** and precision **0.347**, improving over paper-fixed recall **0.296** and precision **0.280**.
- At strict budget `r=0.05`, guarded strategy preserves conservative performance (no degradation vs paper-fixed at key points).

## 7.4 Decision-layer achievement (scientific impact)
- Demonstrated that MAE-optimal model and top-k recall-optimal model are often different.
- Quantified operational enrichment: recall at `r=0.05` frequently around `0.14–0.27`, i.e., about **3x–5x random baseline**.

## 7.5 External robustness (SI Seoul)
From `reports/main_manuscript_assets/SI/tables/TableS15_external_validation_summary_metrics_seoul.csv`:

- Point performance ranking changes by horizon (e.g., ENet strongest at `H=1`, persistence stronger at longer horizons), confirming context dependence.
- Alarm utility still provides enrichment over random in multiple settings, supporting transferability of the decision-evaluation framework.

---

## 8) Outputs for Main Manuscript and SI

## 8.1 Main manuscript core outputs

1. **Table 1 (point performance by horizon)**  
   `results/tables/leaderboard_point_by_horizon.csv`

2. **Table 2 (prevalence + random baseline)**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table2_exceedance_prevalence_random_baseline.csv`

3. **Table 3 (H5 alarm-budget metrics)**  
   `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv`

4. **Table 4 (normalized H5 context comparison)**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table4_H5_normalized_context_comparison.csv`

5. **Table 5 (fold uncertainty at key operating points)**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Table5_key_operating_points_fold_uncertainty.csv`

6. **Figure 2 (TNout reality check)**  
   generated by `src/paper_make_fig2_tnout_reality_check.py`

7. **Figure 3 (H5 budget–recall curves)**  
   `results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_panel_tau15_16_budget_recall.pdf`

8. **Figure 4 (uncertainty bars)**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig4_key_operating_points_uncertainty.pdf`

9. **Figure 5 (decision curve TP vs k)**  
   `results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/figures/Fig5_decision_curve_H5_tau16_tp_vs_k.pdf`

## 8.2 SI outputs
- `reports/main_manuscript_assets/SI/tables/TableS14_cross_site_variable_mapping_ulsan_to_seoul.csv`
- `reports/main_manuscript_assets/SI/tables/TableS15_external_validation_summary_metrics_seoul.csv`
- `reports/main_manuscript_assets/SI/tables/TableS15a_seoul_budget_recall_curve_data.csv`
- `reports/main_manuscript_assets/SI/figures/FigureS_Seoul_budget_recall_transferability.pdf`

## 8.3 Best-decision protocol for Seoul external verification (SI)

### Decision
Use Seoul as a **strict external verification set** (single holdout context), with model training/selection frozen from Ulsan.

### Should all models be tested?
Not all hyperparameter variants should drive conclusions. The best decision is a two-tier strategy:

1. **Primary SI verification set (claim-supporting, mandatory)**
   - Persistence
   - ElasticNet (main manuscript setting)
   - HGBR (Ulsan-tuned settings, no Seoul retuning)

2. **Exploratory SI sensitivity set (optional, clearly labeled exploratory)**
   - Additional model variants (other ElasticNet/Ridge/TCN/hybrid policies), reported transparently but not used to redefine main claims.

### Rules to keep verification valid
1. No feature/threshold/hyperparameter tuning on Seoul.
2. Keep Seoul results in SI only; do not replace main-text winner logic derived from Ulsan CV.
3. Report both point metrics and alarm-budget metrics (`tau = 15/16/17`, `r = 0.05/0.10`).
4. Interpret differences as transferability/robustness evidence, not as new model-selection evidence.

### Implemented SI consolidation path
Use `src/si_make_seoul_external_eval.py`, which generates:
- `TableS14_cross_site_variable_mapping_ulsan_to_seoul.csv`
- `TableS15_external_validation_summary_metrics_seoul.csv`
- `TableS15a_seoul_budget_recall_curve_data.csv`
- `FigureS_Seoul_budget_recall_transferability.(png|pdf)`

---

## 9) Comprehensive Consistency Check (Manuscript vs Artifacts)

## 9.1 Verified and consistent claims

1. **Dataset span and size**  
   Manuscript claim (`n ≈ 1031`, 2021-01-04 to 2023-10-31) matches project artifacts.

2. **Point MAE headline values**  
   Manuscript values (`1.537`, `1.719`, `1.794`) match leaderboard files.

3. **Bounded-attention enrichment claim (3x–5x)**  
   Supported by recall values relative to random baseline at `r=0.05`.

4. **Guarded-hybrid benefit at key H5 points**  
   Direction and magnitude are supported in H5 comparison tables.

## 9.2 Issues needing correction/clarification

1. **Outdated extracted manuscript helper file**  
   `reports/_tmp_manuscript_extracted.txt` is older than `reports/Manuscript.docx`.  
   Action taken: created updated clean extraction `reports/_tmp_manuscript_docx_clean.txt`.

2. **Potential metric mismatch in current manuscript text**  
   In current extracted text, one sentence states for `H=3, tau=16, r=0.10` values around precision `0.513`, recall `0.279`; these values correspond to `tau=15, r=0.10`, not `tau=16, r=0.10` in main alarm table context.

3. **Context mixing risk (main alarm vs H5-v2 normalized context)**  
   Main alarm tables and H5-v2 guarded/normalized runs use different event totals (`events=73/140` vs `71/134` at H5 key thresholds).  
   Recommendation: keep all cross-model comparisons within the same context table (Table 4 for normalized H5 comparisons).

4. **Config metadata gap**  
   `configs/models.yaml` and `configs/features.yaml` are empty.  
   Not a blocker, but filling them would improve reproducibility metadata and reviewer transparency.

---

## 10) Scientific Achievements (Publication-Ready Framing)

1. **Decision-aligned evaluation framework established**  
   Demonstrated that ranking under fixed alarm budgets is a fundamentally different and operationally superior evaluation lens than MAE-only reporting.

2. **Leakage-safe methodology operationalized**  
   Enforced strict temporal boundaries at feature, scaling, and validation stages in small daily WWTP data.

3. **Operating-point-aware model selection demonstrated**  
   Showed horizon- and threshold-dependent winner shifts (persistence short horizon, regularized linear at intermediate horizon, guarded hybrid improvements at long horizon/key budgets).

4. **Bounded-attention value quantified in practical units**  
   Gains reported as additional captured exceedances at fixed `k`, making results directly interpretable for plant operations.

5. **Uncertainty-aware evidence strengthened**  
   Added fold-level CI and decision curves (TP vs k), improving robustness and reviewer readiness.

6. **Main + SI artifact ecosystem completed**  
   End-to-end outputs exist for manuscript and SI, including external transferability analysis and manifest-style traceability.

---

## 11) Recommended Final Manuscript Edits Before Submission

1. Correct any `tau=16`/`tau=15` swapped metric references in Results text.
2. Explicitly label benchmark contexts whenever mixing main alarm and H5-v2/normalized comparisons.
3. Keep key claims tied to fixed operating points `(H, tau, r)` and corresponding table IDs.
4. Cite Table 4 when making guarded-hybrid superiority claims to avoid apples-to-oranges reviewer concerns.

---

## 12) Final Status

The project is scientifically strong and near submission-ready. The methodology, model justification, decision framing, and artifact generation pipeline are coherent and reproducible. The remaining work is mainly editorial consistency tightening (numeric alignment and context labeling), not core method development.
