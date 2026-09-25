# Model Name and Attribution Report

Date: 2026-04-06  
Project: TNout leakage-safe forecasting and alarm-budget decision framework

## 1) Scope

This report documents, for the three requested models:

1. the complete model name,
2. which parts are implemented/developed in this project,
3. which parts are standard methods that should be referenced in manuscript text.

Requested models:

1. HistGradientBoostingRegressor
2. BCR-TCN v1.1
3. HybridRank

---

## 2) Model 1: HistGradientBoostingRegressor

## 2.1 Complete name

- HistGradientBoostingRegressor (scikit-learn)
- Expanded form: Histogram-based Gradient Boosting Regressor

## 2.2 Where it appears in this repository

- src/train_hgbr_optuna_v2.py
- src/save_hgbr_predictions.py
- src/si_make_seoul_external_eval.py
- reports/model_performance_full_report.md (listed as hgbr)

## 2.3 Parts developed in this project (claim as our implementation decisions)

1. Leakage-safe blocked CV objective wiring for tuning with TimeSeriesSplit.
2. Project-specific Optuna search space and bounds for learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, and max_bins.
3. Horizon-wise training/evaluation integration with this TNout feature pipeline and alarm-budget reporting stack.
4. Reproducible run outputs and file conventions used in this repo.

## 2.4 Parts that are standard and should be referenced

1. Gradient boosting trees and histogram-based boosting algorithm.
2. scikit-learn implementation class HistGradientBoostingRegressor.
3. Optuna framework for black-box hyperparameter optimization.

## 2.5 Suggested citation targets

1. scikit-learn paper and/or official API docs for HistGradientBoostingRegressor.
2. Original gradient boosting literature (Friedman 2001).
3. Optuna paper/tool citation.

---

## 3) Model 2: BCR-TCN v1.1

## 3.1 Complete name

- BCR-TCN v1.1
- Recommended expanded form for manuscript: Budget-Constrained Recall Temporal Convolutional Network, version 1.1

Note: The repository uses the acronym BCR-TCN consistently, while the explicit expansion is not centrally declared in a single definition file. The expansion above matches the implemented training objective and evaluation focus.

## 3.2 Where it appears in this repository

- src/train_bcr_tcn_v11.py (core training/inference)
- src/si_export_training_diagnostics.py (diagnostic replay classes)
- reports/methods and materials_SI.md (method equations)
- reports/Optimization_Process_Exact_Trustable_Results.md
- reports/model_performance_full_report.md

## 3.3 Parts developed in this project (claim as our method contributions)

1. Composite objective design and weighting in current version:
   - Huber regression term,
   - weighted BCE threshold-classification term,
   - pairwise rank loss emphasizing tau16 (with mild tau15 support).
2. Decision-aligned early stopping target:
   - maximize validation Recall@5% at tau=16.
3. Time-safe inner validation split inside outer blocked CV folds.
4. Fold-local feature standardization fit on sub-train only.
5. Project-specific hyperparameter set in v1.1 (channels, blocks, dropout, learning rate, etc.).
6. End-to-end export format with p_tau15/p_tau16/p_tau17 risk outputs used by downstream decision analysis.

## 3.4 Parts that are standard and should be referenced

1. Temporal Convolutional Network concepts:
   - causal dilated convolution,
   - residual temporal blocks.
2. Rank-based pairwise logistic objective family (RankNet-style pairwise ranking concept).
3. BCEWithLogitsLoss, Huber/SmoothL1 loss, AdamW optimizer, gradient clipping.
4. PyTorch framework and standard deep-learning training primitives.

## 3.5 Suggested citation targets

1. TCN references (for causal dilated temporal conv networks; commonly Bai et al., 2018 and related TCN sources).
2. Pairwise ranking loss reference (RankNet family).
3. PyTorch citation.
4. AdamW reference (decoupled weight decay).

---

## 4) Model 3: HybridRank

## 4.1 Complete name

- Hybrid rank ensemble
- In repo naming variants:
  - HybridRank v2
  - hybrid_rank_ens
  - hybrid_paper_fixed
- Recommended complete manuscript name:
  - HybridRank v2 (Guarded Budget-Constrained Rank Ensemble)

## 4.2 Where it appears in this repository

- src/make_hybrid_rank_ensemble.py (paper-fixed style rank blending)
- src/make_hybrid_rank_ensemble_v2.py (policy/guarded optimization)
- reports/methods and materials_SI.md
- reports/methodology_full_report.md
- reports/model_performance_full_report.md

## 4.3 Parts developed in this project (claim as our method contributions)

1. Multi-model rank fusion implementation combining TCN risk score ranks with ENet/persistence/(optional HGBR) prediction ranks.
2. Policy-driven weight search under leakage-safe fold separation.
3. Guarded policy logic:
   - paper-fixed fallback at low budget,
   - policy-specific tuning at higher budget.
4. Precision-floor constraint relative to baseline fixed weights at key conservative operating points.
5. Full operating-point export and manuscript artifact pipeline integration.

## 4.4 Parts that are standard and should be referenced

1. Rank normalization and weighted rank aggregation as a general ensembling strategy.
2. Top-k decision metrics (precision@k, recall@k) and budgeted selection framing.
3. General ensemble-learning rationale (combining complementary model families).

## 4.5 Suggested citation targets

1. Ensemble-learning foundations (e.g., model averaging/stacking references).
2. Learning-to-rank / rank aggregation references where appropriate.
3. Any decision-curve or top-k detection literature you cite for budgeted alarm operations.

---

## 5) Ready-to-use attribution statements

Use these concise manuscript-safe statements.

1. HistGradientBoostingRegressor: "We used the scikit-learn Histogram-based Gradient Boosting Regressor and optimized hyperparameters with Optuna under blocked time-series cross-validation; the core boosting algorithm is standard prior work."
2. BCR-TCN v1.1: "Our BCR-TCN v1.1 implementation combines standard causal TCN components with a project-specific composite objective and a decision-aligned early-stopping criterion (validation Recall@5% at tau=16)."
3. HybridRank: "HybridRank v2 is our policy-guarded rank-fusion framework that blends model ranks and enforces conservative low-budget precision safeguards; rank aggregation and top-k metrics are standard methods."

---

## 6) Citation checklist by model

Before submission, verify that the reference list includes:

1. HGBR/scikit-learn source citation and gradient boosting foundation.
2. Optuna citation.
3. TCN architecture citation(s).
4. Pairwise ranking loss citation.
5. PyTorch and AdamW citations.
6. Ensemble/rank-aggregation and top-k decision-metric references.

---

## 7) Evidence links inside this repository

Primary evidence files used for this attribution:

1. src/train_hgbr_optuna_v2.py
2. src/train_bcr_tcn_v11.py
3. src/make_hybrid_rank_ensemble.py
4. src/make_hybrid_rank_ensemble_v2.py
5. reports/methods and materials_SI.md
6. reports/Methods and materials_main.md
7. reports/model_performance_full_report.md
