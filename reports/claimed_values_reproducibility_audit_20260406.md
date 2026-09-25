# Claimed Values Reproducibility Audit

- PASS: 28
- WARN: 0
- FAIL: 0

## Detailed Results
- [PASS] Table1 H=1: claim mae/rmse/mase1/mase7=1.5373/1.9287/1.2295/0.8116 vs recomputed 1.5373/1.9287/1.2295/0.8116
- [PASS] Table1 H=3: claim mae/rmse/mase1/mase7=1.7191/2.1631/1.3744/0.907 vs recomputed 1.7191/2.1631/1.3744/0.907
- [PASS] Table1 H=5: claim mae/rmse/mase1/mase7=1.7944/2.2665/1.4486/0.9522 vs recomputed 1.7944/2.2665/1.4486/0.9522
- [PASS] Table2 H=1 tau=15: claim n/events/prevalence=762/141/0.185039370079 vs recomputed 762/141/0.185039370079
- [PASS] Table2 H=1 tau=16: claim n/events/prevalence=762/74/0.097112860892 vs recomputed 762/74/0.097112860892
- [PASS] Table2 H=1 tau=17: claim n/events/prevalence=762/30/0.039370078740 vs recomputed 762/30/0.039370078740
- [PASS] Table2 H=3 tau=15: claim n/events/prevalence=759/140/0.184453227931 vs recomputed 759/140/0.184453227931
- [PASS] Table2 H=3 tau=16: claim n/events/prevalence=759/73/0.096179183136 vs recomputed 759/73/0.096179183136
- [PASS] Table2 H=3 tau=17: claim n/events/prevalence=759/29/0.038208168643 vs recomputed 759/29/0.038208168643
- [PASS] Table2 H=5 tau=15: claim n/events/prevalence=759/140/0.184453227931 vs recomputed 759/140/0.184453227931
- [PASS] Table2 H=5 tau=16: claim n/events/prevalence=759/73/0.096179183136 vs recomputed 759/73/0.096179183136
- [PASS] Table2 H=5 tau=17: claim n/events/prevalence=759/29/0.038208168643 vs recomputed 759/29/0.038208168643
- [PASS] Table3 H=1 tau=15 r=0.05 persistence: claim k/events/tp/prec/rec=39/141/30/0.7692/0.2128 vs recomputed 39/141/30/0.7692/0.2128
- [PASS] Table3 H=1 tau=15 r=0.05 enet_a0.1_l0.5: claim k/events/tp/prec/rec=39/141/27/0.6923/0.1915 vs recomputed 39/141/27/0.6923/0.1915
- [PASS] Table3 H=1 tau=15 r=0.1 persistence: claim k/events/tp/prec/rec=77/141/50/0.6494/0.3546 vs recomputed 77/141/50/0.6494/0.3546
- [PASS] Table3 H=1 tau=15 r=0.1 enet_a0.1_l0.5: claim k/events/tp/prec/rec=77/141/44/0.5714/0.3121 vs recomputed 77/141/44/0.5714/0.3121
- [PASS] Table3 H=3 tau=16 r=0.05 persistence: claim k/events/tp/prec/rec=38/73/11/0.2895/0.1507 vs recomputed 38/73/11/0.2895/0.1507
- [PASS] Table3 H=3 tau=16 r=0.05 enet_a1.0_l0.2: claim k/events/tp/prec/rec=38/73/16/0.4211/0.2192 vs recomputed 38/73/16/0.4211/0.2192
- [PASS] Table3 H=3 tau=16 r=0.1 persistence: claim k/events/tp/prec/rec=76/73/21/0.2763/0.2877 vs recomputed 76/73/21/0.2763/0.2877
- [PASS] Table3 H=3 tau=16 r=0.1 enet_a0.01_l0.2: claim k/events/tp/prec/rec=76/73/23/0.3026/0.3151 vs recomputed 76/73/23/0.3026/0.3151
- [PASS] Table3 H=5 tau=16 r=0.05 persistence: claim k/events/tp/prec/rec=38/71/15/0.3947/0.2113 vs recomputed 38/71/15/0.3947/0.2113
- [PASS] Table3 H=5 tau=16 r=0.05 enet: claim k/events/tp/prec/rec=38/71/8/0.2105/0.1127 vs recomputed 38/71/8/0.2105/0.1127
- [PASS] Table3 H=5 tau=16 r=0.05 tcn: claim k/events/tp/prec/rec=38/71/15/0.3947/0.2113 vs recomputed 38/71/15/0.3947/0.2113
- [PASS] Table3 H=5 tau=16 r=0.05 hybrid_rank_v2: claim k/events/tp/prec/rec=38/71/17/0.4474/0.2394 vs guarded-metrics 38/71/17/0.4474/0.2394
- [PASS] Table3 H=5 tau=16 r=0.1 persistence: claim k/events/tp/prec/rec=75/71/23/0.3067/0.3239 vs recomputed 75/71/23/0.3067/0.3239
- [PASS] Table3 H=5 tau=16 r=0.1 enet: claim k/events/tp/prec/rec=75/71/19/0.2533/0.2676 vs recomputed 75/71/19/0.2533/0.2676
- [PASS] Table3 H=5 tau=16 r=0.1 hybrid_paper_fixed: claim k/events/tp/prec/rec=75/71/21/0.28/0.2958 vs recomputed 75/71/21/0.28/0.2958
- [PASS] Table3 H=5 tau=16 r=0.1 hybrid_rank_v2: claim k/events/tp/prec/rec=75/71/26/0.3467/0.3662 vs guarded-metrics 75/71/26/0.3467/0.3662

## Conclusion
All audited claimed values are reproducible from repository source artifacts.