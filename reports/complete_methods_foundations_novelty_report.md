# Complete Technical Report: Methods, Mathematical Foundations, Model Justification, Stability Controls, Findings, and Novel Contributions

**Project context**: TNout early warning for WWTP operation under bounded alarm capacity (Ulsan main study, Seoul external robustness in SI)  
**Prepared on**: 2026-03-04

---

## 1) Problem definition

### 1.1 Real-world problem

In plant operation, teams cannot investigate every day as an alarm day. The practical decision problem is:

> Given a limited daily alarm budget (e.g., 5–10% of days), which model best ranks days so exceedance events are captured early with manageable false alarms?

So this is not only a regression problem (minimizing MAE), but also a constrained ranking/decision problem.

### 1.2 Technical risks to solve

1. **Temporal leakage** (future information contaminating training).  
2. **Metric mismatch** (best MAE model not always best top-k alarm model).  
3. **Stability risk** in deep training (overfit/underfit, large gradients, optimization instability).  
4. **Operational misalignment** if evaluation ignores budgeted alarm selection.

---

## 2) Data, targets, and formulation

### 2.1 Supervised learning target

For forecast horizon $H \in \{1,3,5\}$:

$$
y_t^{(H)} = \mathrm{TNout}_{t+H}
$$

Features at day $t$ use only information available up to day $t$ (or earlier).

### 2.2 Alarm decision labels

For threshold $\tau \in \{15,16,17\}$:

$$
e_t^{(\tau)} = \mathbb{1}[y_t^{(H)} \ge \tau]
$$

For budget ratio $r \in \{0.05, 0.10\}$ and test size $n$, alarm capacity is:

$$
k = \lceil r n \rceil
$$

Models are ranked by score, and top-$k$ are flagged as alarms.

---

## 3) Validation protocol (exact method used)

### 3.1 Outer blocked time-series CV

The project uses **expanding-window blocked forward-chaining CV** through:

- `TimeSeriesSplit(n_splits=3)`
- no shuffle
- contiguous chronological test folds
- expanding train history

This is the main anti-leakage backbone and was used across linear, HGBR, and TCN workflows.

Evidence artifacts:

- `results/tables/blocked_cv_split_summary_all_horizons.csv`
- `results/tables/blocked_cv_assignment_matrix_H1.csv`
- `results/tables/blocked_cv_assignment_matrix_H3.csv`
- `results/tables/blocked_cv_assignment_matrix_H5.csv`

### 3.2 Inner time-safe validation for TCN

Inside each outer train fold for BCR-TCN:

- earliest $(1-\text{val\_frac})$ used as sub-train,
- latest `val_frac=0.15` used as sub-validation,
- standardization fit on sub-train only.

This prevents leakage from validation/test periods into preprocessing.

---

## 4) Window size explanation (what window means in this project)

There are **two distinct windows** in this pipeline.

### 4.1 Feature engineering windows (for tabular models)

From `build_ulsan_npz_v2.py`, rolling/time windows include:

- lag windows: 1, 3, 5, 7 days
- moving means/sums: 3, 7, 14, 30 days

Scientific rationale:

- 3–7 d: short process memory and weekly operational rhythm
- 14 d: bi-weekly process/load persistence
- 30 d: slower process and weather smoothing

### 4.2 Sequence window for BCR-TCN (`L`)

The TCN is trained with sequence length `L=60` (default used in v1.1 run metadata).

Interpretation:

- each prediction sees up to 60 historical daily feature vectors,
- enough to include roughly two 30-day process/weather cycles while preserving enough training samples.

Model structure details:

- kernel size $k=3$
- blocks $B=5$
- dilations $d \in \{1,2,4,8,16\}$ (two causal conv layers per block)

Theoretical receptive field of stacked dilated convolutions exceeds 60 steps, so the full input window is exploitable by the network.

---

## 5) Models used and scientific justification

## 5.1 Persistence baseline

### Algorithm

For horizon $H$, use a lagged value as prediction:

$$
\hat{y}_t = y_{t-H}
$$

### Why scientifically justified

- Strong baseline in autocorrelated environmental time series.
- Establishes minimum acceptable utility against naive carry-forward behavior.

---

## 5.2 ElasticNet / Ridge family

### Algorithms

Ridge:

$$
\min_\beta \frac{1}{N}\|y-X\beta\|_2^2 + \lambda\|\beta\|_2^2
$$

ElasticNet:

$$
\min_\beta \frac{1}{N}\|y-X\beta\|_2^2 + \alpha\left(\rho\|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|_2^2\right)
$$

with train-fold-only standardization.

### Why scientifically justified

- Appropriate bias–variance tradeoff for modest sample sizes.
- Robust generalization and interpretability.
- Serves as a strong low-complexity baseline before nonlinear/deep methods.

---

## 5.3 HistGradientBoostingRegressor (HGBR, Optuna tuned)

### Algorithmic foundation

Additive tree boosting approximates:

$$
\hat{f}(x) = \sum_{m=1}^{M} \eta \cdot h_m(x)
$$

where each tree $h_m$ is fit to reduce residual loss from current ensemble prediction.

### Why scientifically justified

- Captures nonlinear interactions among hydraulic, influent, weather, and seasonal features.
- Handles tabular heterogeneity effectively.
- Optuna tuning under blocked CV improves fairness of comparison.

---

## 5.4 BCR-TCN v1.1 (causal temporal deep model)

### Architecture

- Causal dilated convolutional blocks (no future look-ahead)
- Residual connections
- Two heads:
  - regression head for TNout value
  - classification head producing logits for exceedance thresholds

### Loss function (used in code)

$$
\mathcal{L}_{\text{total}} = w_{reg}\mathcal{L}_{\text{Huber}} + w_{bce}\mathcal{L}_{\text{BCE}} + w_{rank}\mathcal{L}_{\text{Rank}}
$$

with project config (H5 v2 metadata):

- $w_{reg}=1.0$, $w_{bce}=0.6$, $w_{rank}=1.4$
- optimizer: AdamW
- gradient clipping: `clip_grad_norm_=1.0`
- early stopping patience: 25

### Ranking loss concept

Pairwise objective encourages higher score for positive-event samples than negatives:

$$
\mathcal{L}_{rank} = \mathbb{E}_{(i,j)}\left[\log\left(1+e^{-(s_i-s_j)}\right)\right],\quad i\in\text{pos}, j\in\text{neg}
$$

### Why scientifically justified

- Causal TCN is suitable for temporal dependence and nonlinear dynamics.
- Joint regression + exceedance classification + ranking directly aligns representation with decision tasks.

---

## 5.5 Hybrid models

### Hybrid point predictor

Convex blend of base point predictors (ENet, persistence, optional HGBR), weights tuned on training folds to minimize MAE.

### Hybrid rank ensemble (paper-fixed and policy-guarded v2)

- Rank-normalize model scores and combine with nonnegative weights.
- Guarded objective imposes conservative precision floor at strict budget points.

Why scientifically justified:

- Different models capture complementary structure.
- Ensembling in rank space is natural for top-k selection tasks.
- Guarding prevents deterioration at high-cost low-budget operation.

---

## 6) How overfit, underfit, gradient vanishing/exploding were prevented and checked

## 6.1 Prevention mechanisms built into training

### Overfitting prevention

1. blocked time-series CV (chronological generalization only)
2. inner time-safe validation split
3. early stopping (patience)
4. regularization (ElasticNet penalties, AdamW weight decay)
5. dropout in TCN
6. train-only scaling

### Underfitting mitigation

1. model family diversity (linear, tree, deep, hybrid)
2. nonlinear capacity via HGBR and TCN
3. rank-aware objective to optimize decision signal, not only average error

### Gradient stability prevention

1. causal residual architecture
2. gradient clipping with norm cap = 1.0
3. AdamW optimizer and conservative learning rate

## 6.2 Diagnostic checks that were executed

Generated diagnostics artifacts:

- `results/tables/si_training_epoch_diagnostics_H5.csv`
- `results/tables/si_gradient_health_epoch_flags_H5.csv`
- `results/tables/si_gradient_ve_summary_H5.csv`
- `results/tables/si_overfit_underfit_epoch_flags_H5.csv`
- `results/tables/si_overfit_underfit_counts_H5.csv`
- `results/tables/si_failure_bars_H5.csv`
- `results/tables/si_failure_bars_overall_H5.csv`

### Quantitative check outcomes (H5 diagnostics replay)

- Vanishing-risk epochs: **0 across folds** under the implemented flag rule.
- Exploding-risk flags often appear in **pre-clip gradients**, while post-clip norms remain bounded by design.
- Fit regime labels across epochs show majority as `stable_or_improving`, with smaller subsets flagged as under/overfit risk.

Interpretation:

- clipping is actively engaged and functioning as intended,
- optimization can produce large raw gradients, but controlled update norms reduce instability risk,
- overfit/underfit behavior is episodic, not dominant across epochs.

---

## 7) Evaluation mathematics (point + alarm decision)

## 7.1 Point metrics

MAE:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|
$$

RMSE:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}
$$

MASE (seasonal period $m$):

$$
\mathrm{MASE} = \frac{\frac{1}{n}\sum_t |y_t-\hat{y}_t|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|}
$$

## 7.2 Alarm metrics at budget $k$

After sorting by score and selecting top-$k$ alarms:

$$
\mathrm{Precision} = \frac{TP}{TP+FP},\quad
\mathrm{Recall} = \frac{TP}{TP+FN}
$$

Additional SI failure diagnostics:

$$
\mathrm{Miss\ Rate} = \frac{FN}{TP+FN},\quad
\mathrm{False\ Alarm\ Rate} = \frac{FP}{FP+TN}
$$

---

## 8) Findings and achievements

## 8.1 Empirical findings (main)

1. **Point forecasting**: ElasticNet variants provide strongest and most stable MAE across horizons in main linear evaluation.
2. **Decision forecasting**: Top-k alarm winners vary by $(H,\tau,r)$, confirming no universal winner.
3. **Hybrid ranking**: Guarded rank policy improves selected moderate-budget operating points while preserving strict-budget behavior in key comparisons.
4. **Validation integrity**: chronological blocked CV and fold-safe preprocessing provide leakage-safe estimation.

## 8.2 Achievements relevant to knowledge

Within this project’s scope, the work contributes:

1. A reproducible **decision-aligned** benchmark where model selection is tied to constrained alarm capacity rather than only global regression error.
2. A practical demonstration that **metric-optimal model depends on operating policy** (budget + threshold), not just on MAE ranking.
3. A leakage-safe hybrid ranking framework with **guarded policy behavior** at conservative budgets.
4. SI-ready stability diagnostics linking optimization behavior (gradients, fit regimes) to operational outcome diagnostics (failure bars).

## 8.3 What was newly understood (project-level novelty insight)

1. Better MAE does not automatically imply better constrained exceedance capture.
2. Alarm-policy context changes model preference; therefore model choice should be policy-specific.
3. Stability control (especially clipping + early stopping + fold-safe validation) is essential for reliable deep ranking behavior in relatively small daily environmental datasets.

---

## 9) Fundamental scientific justification of model portfolio

The final model set is scientifically justified as a **structured ladder of inductive bias and complexity**:

1. persistence: autocorrelation anchor and operational baseline,
2. linear regularized models: robust low-variance baseline,
3. gradient boosting: nonlinear tabular interactions,
4. causal deep temporal model: sequential nonlinear dynamics + ranking-aware objective,
5. hybrid ensembles: controlled bias-complementarity for both point and ranking objectives.

This hierarchy enables interpretable ablation-like comparison while maintaining operational relevance.

---

## 10) Reproducibility and evidence files

Core supporting files:

- Blocked CV artifacts: `results/tables/blocked_cv_split_summary_all_horizons.csv`
- Main model synthesis: `reports/project_complete_report.md`
- SI diagnostics report: `reports/SI_complete_blockedCV_training_validation_diagnostics.md`
- Diagnostics exporter: `src/si_export_training_diagnostics.py`
- Guarded-hybrid comparison rows: `results/tables/discussion_key_results.csv`

Diagnostic generation command:

```bash
cd /home/alrezshams/acs_tnout_ulsan
/bin/python3 src/si_export_training_diagnostics.py --h 5 --epochs 60 --splits 3 --device cpu
```

---

## 11) Suggested manuscript phrasing for window size

You can describe it as:

> “We used dual temporal windows: (i) engineered rolling/lagged covariates over 3–30 day scales to represent short-to-medium process memory, and (ii) a 60-day causal sequence window for TCN training, chosen to capture multi-week dependencies while preserving sufficient effective sample size under blocked time-series cross-validation.”
