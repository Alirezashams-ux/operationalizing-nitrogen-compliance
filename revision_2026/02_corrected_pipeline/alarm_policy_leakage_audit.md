# Alarm Policy Leakage Audit

- training-only deployable: 3
- validation-calibrated deployable: 0
- retrospective test-set benchmark: 3
- leakage: 3
- unclear: 0

## Classification Notes
- Global k = ceil(r*N) over complete held-out blocks is retrospective benchmarking, not an online deployable rule.
- Rank normalization over full fold score distributions is retrospective block scoring.
- Table3 representative operating-point and run-variant choices are classified as leakage due held-out-informed selection.