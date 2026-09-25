# BCR-TCN v1.1 Definition and Provenance (H5 Controlled Regeneration)

## 1. Name and Version
- Full name: Budget-Constrained Recall Temporal Convolutional Network.
- Model token in code and artifacts: bcr_tcn_v11.
- v1.1 meaning: internal project-specific implementation/model version 1.1 used in submitted experiments.

## 2. What BCR Means Operationally
- BCR means checkpoint selection is aligned to a fixed-budget recall objective on validation data.
- The concrete selection criterion is validation Recall@5% at tau=16 mg/L.
- This uses r_budget=0.05 and tau=16 threshold exceedance labels.

## 3. What BCR Does Not Mean
- The 5% budget is not implemented as a hard differentiable inequality constraint in the final v1.1 training loss.
- BCR is a decision-aligned model-selection rule, not a claim of constrained optimization feasibility in loss-space.

## 4. Fixed Sequence and Feature Provenance
- Sequence length L: 60.
- Feature-set source: features/ulsan_H5_features_v2.npz.
- Input matrix shape in submitted context: 996 rows x 34 features.
- Feature dates span 2021-02-03 to 2023-10-26.

## 5. Backbone and Heads
- Backbone type: causal dilated residual TCN blocks.
- Causal convolution implementation: left-only padding equal to (kernel-1)*dilation per Conv1d.
- Number of blocks: 5.
- Dilation sequence: 1, 2, 4, 8, 16.
- Channels: 48.
- Kernel size: 3.
- Dropout: 0.25.
- Residual connections: identity or 1x1 projection when channel dimensions differ.
- Representation for heads: last-timestep feature vector (not temporal average pooling).
- Regression head: MLP ending in scalar y_pred.
- Classification head: MLP ending in three logits for thresholds.
- Thresholds: 15, 16, and 17 mg/L.

## 6. Composite Objective and Optimization
- Regression loss: SmoothL1/Huber with beta=1.0.
- Classification loss: weighted BCEWithLogits over tau15, tau16, tau17 labels.
- Pairwise ranking loss: softplus pairwise loss on exceedance/non-exceedance score ordering.
- Loss weights:
  - w_reg = 1.0
  - w_bce = 0.6
  - w_rank = 1.4
- Ranking emphasis:
  - primary: tau16 score ranking;
  - secondary: tau15 contribution weighted by 0.3 multiplier inside ranking component.
- Optimizer: AdamW.
- Learning rate: 0.0008.
- Weight decay: 0.001.
- Batch size: 64.
- Epoch cap: 260.
- Patience: 25.
- Gradient clipping: max norm 1.0.

## 7. Validation Criterion and Checkpoint Rule
- Validation criterion for checkpoint selection: maximize Recall@5% at tau=16 on validation only.
- Diagnostic-only metric: validation MAE.
- Tie rule: if validation Recall@5% at tau=16 ties within tolerance, keep earliest epoch.
- Selected checkpoint is used directly for outer-test prediction (no post-selection refit on full outer-training rows in submitted v1.1 behavior).

## 8. Random Seed and Determinism Rule
- Seed: 42 for Python, NumPy, and PyTorch.
- Deterministic execution rule for corrected canonical run: use deterministic PyTorch algorithms and stop if unsupported deterministic operation is encountered.

## 9. Difference from Conventional Regression-Only TCN
- BCR-TCN v1.1 keeps a standard causal dilated residual TCN backbone.
- It differs from regression-only TCN through:
  - dual heads (regression + threshold classification logits),
  - composite regression-classification-ranking objective,
  - threshold-specific exceedance outputs (tau15/tau16/tau17 risk scores),
  - budget-aligned validation checkpoint selection rule.

## 10. Comparison Table

| Dimension | Standard regression TCN | BCR-TCN v1.1 | Evidence source | Architectural difference? | Objective/training difference? |
|---|---|---|---|---|---|
| Temporal backbone | Causal dilated Conv1d residual stack | Same causal dilated residual stack | src/train_bcr_tcn_v11.py; revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_standard_tcn_comparison.csv | No | No |
| Readout | Often pooled or single regression representation | Last-timestep representation fed to dual heads | src/train_bcr_tcn_v11.py | Minor head-interface change | No |
| Output heads | Single regression head | Regression head + 3-threshold classification head | src/train_bcr_tcn_v11.py | Yes (head design) | Yes |
| Objective | Single regression loss common | Huber + weighted BCE + pairwise ranking | src/train_bcr_tcn_v11.py | No backbone change | Yes |
| Decision alignment | Validation loss/min-error checkpointing common | Validation Recall@5% at tau16 checkpointing | src/train_bcr_tcn_v11.py; results/metrics/bcr_tcn_v11_H5_v2_meta.json | No | Yes |
| Budget handling | Usually not explicit | Budget appears in checkpoint selection at r=0.05 | src/train_bcr_tcn_v11.py; revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_identity_audit.md | No | Yes |

## 11. Provenance Sources
- Source implementation inspected read-only: /home/alrezshams/acs_tnout_ulsan/src/train_bcr_tcn_v11.py.
- Submitted-style metadata: results/metrics/bcr_tcn_v11_H5_v2_meta.json.
- Submitted-style predictions: results/predictions/bcr_tcn_v11_H5_v2_preds.csv.
- Identity and architecture audit: revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_identity_audit.md.
- Standard-vs-BCR comparison matrix: revision_2026/00_provenance/bcr_tcn_identity_audit/bcr_tcn_standard_tcn_comparison.csv.
