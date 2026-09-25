# Methodology Report (Leakage-Safe Forecasting and Alarm-Budget Evaluation)

## 1) Study Objective and Framing
This study develops and evaluates daily effluent TNout forecasting models for operational early warning under staffing constraints. The methodological goal is twofold:

1. Produce leakage-safe point forecasts for forecast horizons $H \in \{1,3,5\}$ days.
2. Evaluate practical alarm utility under fixed alert capacity (alarm budget), not only conventional regression metrics.

The decision context is WWTP operations where only a fraction of days can be acted upon. Therefore, model ranking quality at the top-$k$ risk days is critical.

---

## 2) Dataset and Scope
- **Site/Data**: Ulsan WWTP daily records.
- **Primary file used**: `data/raw/Ulsan_Yongsan.csv`.
- **Period**: 2021-01-04 to 2023-10-31.
- **Sample size**: 1031 daily rows (before horizon-dependent usable-row reductions).
- **Target**: TNout (mg/L).

### Core variables used in this project
- **Process variables**: Inflow, TNin, TOCin, BODin.
- **Climate variables**: temp_mean_c, precip_total_mm, wet_hours.

Descriptive summary for these variables is provided in:
- `results/tables/Ulsan_Yongsan_main_feature_descriptive_stats.csv`
- `reports/Ulsan_Yongsan_initial_descriptive_analysis.md`

---

## 3) Leakage-Safe Feature Construction
The project enforces a strict temporal information boundary:
- For prediction at day $t+H$, features are computed from information available at or before day $t$.
- No future observations are included in lag, rolling, or engineered covariates.

### Feature classes
1. **Autoregressive memory**
   - TNout lags and rolling summaries (e.g., lag-1, rolling 7/14 where available by script).
2. **Influent/process load signals**
   - Inflow, TNin, TOCin, BODin and rolling smoothers.
3. **Weather/exogenous signals**
   - Temperature and precipitation-derived summaries.
4. **Calendar/seasonality**
   - Cyclic day-of-year encoding (`sin_doy`, `cos_doy`) where used.

Feature pipelines are implemented via horizon-specific and general scripts in `src` (e.g., `build_ulsan_npz_H1.py`, related horizon scripts/runs).

---

## 4) Train/Validation Protocol (Time-Series Safe)
Evaluation is based on **blocked time-series cross-validation** (`TimeSeriesSplit` style), typically 3 splits:
- Earlier contiguous block(s) used for training.
- Later contiguous block used for validation/testing in each split.
- No random shuffling.

This protocol preserves chronology and minimizes leakage from future to past.

---

## 5) Models Evaluated
### 5.1 Baselines
- **Persistence** (carry-forward style baseline in horizon-consistent form).

### 5.2 Linear models
- **Ridge/ElasticNet** (multiple alpha/l1_ratio settings in linear experiments).

### 5.3 Tree-based model
- **HGBR** (Optuna-tuned variant in v2 runs where available).

### 5.4 Deep/sequence model
- **TCN/BCR-TCN v1.1** (probability-like risk outputs per threshold in v2 analysis).

### 5.5 Hybrid models
- **Hybrid rank ensemble (paper v2)**: weighted rank fusion of TCN + linear/persistence components.
- **Hybrid rank ensemble v2**: policy-driven rank tuning with guarded fallback.
- **Hybrid point predictor**: weighted blend for TNout point forecast quality.

Comprehensive model inventory and naming map:
- `results/tables/model_registry_with_full_names.csv`

---

## 6) Metrics
## 6.1 Point-forecast metrics (regression)
- MAE
- RMSE
- MASE(1)
- MASE(7)

These are reported by fold and overall (horizon-specific outputs).

## 6.2 Operational alarm metrics (classification-by-ranking)
For each threshold $\tau \in \{15,16,17\}$ mg/L and budget $r \in \{0.05,0.10\}$:
- Compute $k = \lceil r \cdot n \rceil$.
- Rank days by model risk score.
- Flag top-$k$ as alarms.
- Compute:
  - Precision = TP / k
  - Recall = TP / events

This directly measures actionability under constrained alarm capacity.

---

## 7) Hybrid Ranking Strategy and Guarded Policy
### 7.1 Core concept
Different models can dominate at different operating points. A single global weight vector is often suboptimal.

### 7.2 Policy-specific tuning
In v2 methodology, rank weights are tuned per $(\tau, r)$ in a leakage-safe manner:
- Tune on training folds only.
- Apply selected weights to held-out fold.

### 7.3 Guarded fallback
To avoid regressions at conservative low-budget settings, `policy_guarded` mode uses:
- Paper-fixed hybrid behavior for low budget (default guard $r \le 0.05$).
- Policy-specific tuned weights for higher budget.

This preserves key conservative operating-point performance while improving higher-budget recall/precision trade-offs.

Implementation is in:
- `src/make_hybrid_rank_ensemble_v2.py`

---

## 8) Reproducibility and Artifact Management
Recent scripts use timestamped run directories to prevent overwriting:
- `results/hybrid_rank_v2_runs/<timestamp>_H*_hybrid_rank_v2/`

Typical outputs per run:
- `models/`: saved weight tables and model registry.
- `predictions/`: per-model prediction/score CSVs.
- `tables/`: point metrics, alarm metrics, curve tables.
- `figures/`: learning curves and budget–recall plots.
- `run_config.json`: run metadata.

This supports auditability and manuscript traceability.

---

## 9) Current Empirical Pattern (High-Level)
From project outputs:
- Point forecasting: hybrid point blends are strong and generally beat single baselines in H5 v2 runs.
- Alarm ranking: best model depends on $(\tau, r)$; guarded policy avoids low-budget degradation while improving higher-budget points.

Reference summaries:
- `results/tables/final_model_summary_for_paper.csv`
- `reports/final_model_summary.md`
- `results/tables/leaderboard_alarm_winners.csv`

---

## 10) What Is Missing or Vague in the Provided Introduction
Your current introduction sentence is directionally strong but methodologically underspecified for publication-grade reproducibility. Missing/vague items to add explicitly:

1. **Dataset identity and span**
   - Exact site, date range, and sample size.
2. **Prediction target and horizons**
   - TNout; $H=1,3,5$ days.
3. **Leakage definition and controls**
   - Explicit statement that all features at day $t$ use only information up to $t$.
4. **Validation protocol**
   - Blocked time-series CV (number of folds and rationale).
5. **Operational evaluation layer**
   - Alarm-budget setup with top-$k$ ranking and fixed $r$.
6. **Decision thresholds and budgets**
   - $\tau \in \{15,16,17\}$, $r \in \{0.05,0.10\}$.
7. **Primary comparison classes**
   - Persistence, linear models, tree model, TCN/BCR-TCN, and hybrids.
8. **Main contribution claim precision**
   - Clarify whether contribution is (a) framework, (b) specific hybrid, or (c) both.
9. **Uncertainty / robustness statement**
   - How stability across folds/horizons is handled.
10. **Reproducibility statement**
   - Mention timestamped artifacts and no-overwrite protocol.

---

## 11) Suggested Revision of Your Intro Sentence (Methodology-Ready)
"In small daily WWTP datasets, insufficiently specified evaluation protocols can induce temporal data leakage and optimistic bias; therefore, we adopt leakage-safe feature construction, blocked time-series cross-validation, and alarm-budget top-$k$ ranking metrics to provide decision-grade TNout early warning under fixed operational capacity."

---

## 12) Files to Cite in Methodology Section
- Data/description:
  - `data/raw/Ulsan_Yongsan.csv`
  - `results/tables/Ulsan_Yongsan_main_feature_descriptive_stats.csv`
- Pipeline and modeling:
  - `src/make_hybrid_rank_ensemble_v2.py`
- Main summaries:
  - `results/tables/final_model_summary_for_paper.csv`
  - `results/tables/model_point_performance_full.csv`
  - `results/tables/model_alarm_performance_full.csv`
  - `results/tables/leaderboard_alarm_winners.csv`

