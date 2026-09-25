# Quota-Enforced Sequential H5 Alarm Policy

## 1. Purpose
- Construct a cumulative-capacity-constrained sequential H5 alarm simulation over the frozen past-only policy scores.
- Preserve existing candidate-alarm logic and add only a hard quota gate.

## 2. Input Verification
- verification_pass: True
- source_sequential_completion_decision: A
- git_branch: controlled-reruns-v1
- git_commit: e98a21d77033d746e7b77f348af8dc34be0768fd

## 3. Candidate Alarm Logic
- candidate_alarm = policy_score > past_only_cutoff (strict inequality).
- Equality with cutoff remains no-alarm.
- candidate_reproduction_vs_frozen_baseline_pass: True

## 4. Quota Gate
- capacity_j = ceil(r * j) on eligible-date prefixes only.
- alarm_flag = candidate_alarm AND alarms_before_current_date < capacity_j.
- capacity_audit_pass: True
- suppression_audit_pass: True

## 5. Capacity Expectations
- eligible_N per fold: 219
- r=0.05 final per-fold cap=11; pooled cap=33.
- r=0.10 final per-fold cap=22; pooled cap=66.
- Unused capacity is allowed when candidate alarms are insufficient.

## 6. Comparison with Unconstrained Sequential Simulation
- The unconstrained policy is causally valid but may exceed capacity.
- The quota-enforced policy guarantees the cumulative alarm cap.
- Both remain historical simulations rather than prospective field validation.

## 7. Information Isolation
- information_isolation_pass: True
- source_preservation_pass: True

## 8. Determinism and Tests
- deterministic_result: pass
- test_result: pass_25_of_25
- run_id: quota_enforced_h5_policy_e98a21d77033_6ed147cd_f83baf57

FINAL DECISION

A. Quota-enforced sequential H5 policy passed; H1/H3 canonical reconciliation may begin.

TERMINAL SUMMARY

1. Input verification result: True
2. Candidate reproduction to frozen baseline: True
3. Capacity audit result: True
4. Suppression audit result: True
5. Information-isolation result: True
6. Source preservation result: True
7. Deterministic status: pass
8. Final decision: A
