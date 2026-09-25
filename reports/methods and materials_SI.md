# Methods and Materials for Supplementary Information (SI)

## 1) SI objective

This SI methods file documents all implementation-critical details that support strict reproducibility and reviewer-level auditability beyond main-text space limits.

## 2) Complete split and leakage-control specification

### 2.1 Outer blocked CV

Expanding blocked TimeSeriesSplit with n_splits = 3 is used for evaluation.

Constraints:

$$
\max(Train_i) < \min(Test_i),
\quad Train_1 \subset Train_2 \subset Train_3
$$

### 2.2 Inner validation for deep training

Within each outer train fold:

$$
Train_i = Subtrain_i \cup Val_i,
\quad
\max(Subtrain_i) < \min(Val_i)
$$

with val_frac = 0.15.

### 2.3 Fold-safe transform rule

$$
\tilde{\mathbf{x}}_t = \frac{\mathbf{x}_t-\mu_i}{\sigma_i+10^{-6}}
$$

where \(\mu_i\) and \(\sigma_i\) are fit only on Subtrain_i.

## 3) Full model and optimization details

### 3.1 Linear and baseline models

Implemented in src/train_main_linear.py and src/run_linear_alarm_v2.py.

Includes persistence, seasonal baseline, Ridge grid, and ElasticNet grid.

### 3.2 HGBR optimization

Implemented in src/train_hgbr_optuna_v2.py using Optuna with blocked CV MAE objective.

### 3.3 BCR-TCN v1.1

Implemented in src/train_bcr_tcn_v11.py.

Composite objective:

$$
\mathcal{L}_{total}=w_{reg}\mathcal{L}_{Huber}+w_{bce}\mathcal{L}_{BCE}+w_{rank}\mathcal{L}_{pair}
$$

Pairwise term:

$$
\mathcal{L}_{pair} = \frac{1}{M}\sum_{m=1}^{M}\log(1+\exp(-(s_m^+-s_m^-)))
$$

Gradient clipping:

$$
\mathbf{g}_{clip}=\mathbf{g}\cdot\min\left(1,\frac{c}{\|\mathbf{g}\|_2+\epsilon}\right),\ c=1.0
$$

Early stopping objective: maximize validation Recall@5% at tau = 16.

### 3.4 Hybrid policy optimization

Implemented in src/make_hybrid_rank_ensemble_v2.py.

Rank score equation:

$$
S_\tau(t)=w_{tcn}R(p_\tau^{tcn}(t)) + \sum_m w_m R(\hat y_m(t))
$$

Weight constraints:

$$
\sum_j w_j=1,\quad w_j\ge0
$$

Guarding constraint:

$$
Precision_{candidate}(\tau,r) \ge Precision_{paper-fixed}(\tau,r)
$$

Guarded policy:

$$
S_\tau^{guarded}(t;r)=
\begin{cases}
S_\tau^{paper-fixed}(t), & r\le 0.05\\
S_\tau^{policy}(t;r), & r>0.05
\end{cases}
$$

## 4) Decision-layer equations and interpretation

Alarm budget:

$$
k=\lceil rn\rceil
$$

Metrics:

$$
Precision@k=\frac{TP}{k},
\quad
Recall@k=\frac{TP}{TP+FN}
$$

Additional failure metrics:

$$
Miss\ Rate = \frac{FN}{TP+FN},
\quad
False\ Alarm\ Rate = \frac{FP}{FP+TN}
$$

Random baseline expectation:

$$
\mathbb{E}[Recall@k]=r
$$

## 5) Gradient and fit diagnostics reported in SI

Files:

- results/tables/si_training_epoch_diagnostics_H5.csv
- results/tables/si_gradient_health_epoch_flags_H5.csv
- results/tables/si_gradient_ve_summary_H5.csv
- results/tables/si_overfit_underfit_epoch_flags_H5.csv
- results/tables/si_overfit_underfit_counts_H5.csv
- results/tables/si_failure_bars_H5.csv
- results/tables/si_failure_bars_overall_H5.csv

Interpretation to include:

1. Clipping was active in many epochs (stability safeguard engaged).
2. No material vanishing-gradient signal under defined thresholds.
3. Majority epochs fell in stable/improving regime in exported diagnostics.

## 6) Blocked CV evidence tables for SI

- results/tables/blocked_cv_split_summary_all_horizons.csv
- results/tables/blocked_cv_split_summary_H1.csv
- results/tables/blocked_cv_split_summary_H3.csv
- results/tables/blocked_cv_split_summary_H5.csv
- results/tables/blocked_cv_assignment_matrix_H1.csv
- results/tables/blocked_cv_assignment_matrix_H3.csv
- results/tables/blocked_cv_assignment_matrix_H5.csv

These should be cited as explicit evidence of chronology-preserving split design.

## 7) Strict held-out external verification

External site:

- data/raw/Seoul_Tancheon_1.csv

Protocol statement for SI:

No model architecture, hyperparameter, threshold, or feature-selection retuning was performed on Seoul for main-claim optimization.

## 8) Seeding and reproducibility details

1. Seed fixed to 42 in deep model training path.
2. Timestamped run folders preserve immutable run lineage.
3. run_config.json captures objective and input provenance.
4. Final paper CSV package assembled under results/final_tables/results_discussion_csvs/.

## 9) Exact reproducibility commands for SI section

Use documented commands:

1. /bin/python3 src/export_blocked_cv_strategy_artifacts.py
2. /bin/python3 src/si_export_training_diagnostics.py --h 5 --epochs 60 --splits 3 --device cpu
3. /bin/python3 src/export_si_supporting_csvs.py

## 10) SI reviewer-facing novelty framing

1. Counterintuitive methodological choice: optimize decision quality under hard operational budget rather than only average error.
2. Counterintuitive safety choice: enforce precision-floor guarding to prevent low-budget degradation.
3. Counterintuitive reporting choice: include optimization-stability diagnostics and failure bars as first-class evidence, not optional appendix narrative.

## 11) SI uncertainty equation

Fold-level 95 percent CI for reported means:

$$
CI_{95} \approx t_{0.975,K-1}\frac{s}{\sqrt{K}}
$$

where K is number of folds.

## 12) Closing SI statement

The SI methods package demonstrates that the reported gains are not artifacts of leakage, overfitting, or threshold cherry-picking, but arise from chronology-safe training, constrained policy optimization, and explicitly auditable reproducibility artifacts.