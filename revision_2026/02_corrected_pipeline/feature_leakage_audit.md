# Feature Leakage Audit

- PASS-like findings: 16
- Non-pass findings: 0

## Key Findings
- TNout lag and rolling features explicitly apply shift(1) before rolling; current-day TNout is not used in those windows.
- No forward-fill/interpolation operations were detected in primary feature-builder scripts.
- Several process/weather rolling features include current-day (t) measurements; this is not future leakage but requires clear forecast issue-time assumptions.
- Feature windows are computed globally before split, but all audited rolling formulas are backward-looking only.