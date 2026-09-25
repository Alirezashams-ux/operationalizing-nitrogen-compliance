# Fixed Rank Ensemble Policy Definition (H5)

Policy name: HybridRank_fixed_documented
Submitted alias: hybrid_paper_fixed

Components and fixed documented weights:
- BCR-TCN v1.1 = 0.50
- ElasticNet = 0.25
- Persistence = 0.25
- HGBR = 0.00

Required policy properties:
- weights are identical for all folds;
- weights are identical for all thresholds;
- the score itself is independent of alarm budget;
- no corrected validation outcome was used to select weights;
- no outer-test outcome was used to select weights;
- no model was retrained;
- HGBR remains in the documented component registry but contributes zero weight;
- BCR-TCN contributes a threshold-specific probability;
- regression components contribute point predictions;
- score generation is retrospective because component ranks are normalized within the complete held-out outer-test fold;
- this score is not a deployable sequential score;
- a separate sequential policy will be created later.

Preferred manuscript-facing term: fixed rank ensemble

Equation for threshold tau:

S_tau(t) =
0.50 * RN(p_BCR-TCN,tau(t))
+ 0.25 * RN(yhat_ElasticNet(t))
+ 0.25 * RN(yhat_Persistence(t))
+ 0.00 * RN(yhat_HGBR(t))

