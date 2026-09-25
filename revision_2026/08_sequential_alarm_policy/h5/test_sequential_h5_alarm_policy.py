from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_sequential_h5_alarm_policy.py"
TEST_SCRIPT = Path(__file__).resolve()

EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_MODELS = [
    "Persistence",
    "Ridge",
    "ElasticNet",
    "HGBR",
    "BCR-TCN v1.1",
    "HybridRank_fixed_documented",
]
EXPECTED_THRESHOLDS = [15, 16, 17]
EXPECTED_BUDGETS = [0.05, 0.10]
EXPECTED_SCORE_CONTEXT = "historical_sequential_past_only"
STARTUP_DAYS = 30

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"
FIXED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fixed_policy"
RETRO_DIR = ROOT / "revision_2026" / "07_retrospective_alarm_budget" / "h5"

PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"
FIXED_CHECKSUMS_PATH = FIXED_DIR / "fixed_hybridrank_h5_checksums.sha256"
RETRO_CHECKSUMS_PATH = RETRO_DIR / "retrospective_h5_alarm_budget_checksums.sha256"

CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
FIXED_REPORT_PATH = FIXED_DIR / "fixed_hybridrank_h5_completion_report.md"
RETRO_REPORT_PATH = RETRO_DIR / "retrospective_h5_alarm_budget_completion_report.md"
FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"
FIXED_WEIGHTS_PATH = FIXED_DIR / "fixed_hybridrank_weights.csv"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
STARTUP_REGISTRY_PATH = OUT_DIR / "sequential_h5_startup_registry.csv"
CAUSAL_COMPONENT_RANKS_PATH = OUT_DIR / "sequential_h5_causal_component_ranks.csv"
CAUSAL_FIXED_ENSEMBLE_PATH = OUT_DIR / "sequential_h5_causal_fixed_ensemble_scores.csv"
SEQUENTIAL_DECISIONS_PATH = OUT_DIR / "sequential_h5_alarm_decisions.csv"
CUTOFF_AUDIT_PATH = OUT_DIR / "sequential_h5_cutoff_audit.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "sequential_h5_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "sequential_h5_metrics_pooled.csv"
FULL_FOLD_SECONDARY_PATH = OUT_DIR / "sequential_h5_full_fold_secondary_metrics.csv"
BURDEN_AUDIT_PATH = OUT_DIR / "sequential_h5_alarm_burden_audit.csv"
SEQ_VS_RETRO_PATH = OUT_DIR / "sequential_vs_retrospective_definition.md"
INFO_ISOLATION_PATH = OUT_DIR / "sequential_h5_information_isolation_audit.json"
DATE_COVERAGE_AUDIT_PATH = OUT_DIR / "sequential_h5_date_coverage_audit.csv"
MANIFEST_PATH = OUT_DIR / "sequential_h5_alarm_policy_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "sequential_h5_alarm_policy_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "sequential_h5_alarm_policy_checksums.sha256"


class AssertionCollector:
    def __init__(self) -> None:
        self.total = 0
        self.failed = 0
        self.failures: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.total += 1
        if not bool(condition):
            self.failed += 1
            self.failures.append(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise RuntimeError("Malformed checksum line in {}: {}".format(path, line))
        out[parts[-1]] = parts[0]
    return out


def verify_checksum_manifest(path: Path, base_dir: Path) -> bool:
    ok = True
    for expected, rel_name in [(v, k) for k, v in parse_checksum_manifest(path).items()]:
        target = base_dir / rel_name
        if (not target.exists()) or (sha256_file(target) != expected):
            ok = False
            break
    return ok


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"FINAL DECISION\s*(?:\n|\r\n)+(?:<!--.*?-->\s*)?([ABCD])\.\s", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    for ln in txt.splitlines():
        s = ln.strip()
        if re.match(r"^[ABCD]\.\s", s):
            return s[0].upper()
    raise RuntimeError("Could not infer decision from {}".format(path))


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def quantile_higher(values, probability):
    try:
        return float(
            np.quantile(values, probability, method="higher")
        )
    except TypeError:
        return float(
            np.quantile(
                values,
                probability,
                interpolation="higher",
            )
        )


def run_build_script() -> None:
    subprocess.check_call([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT)


def add_canonical_row_id(wide_df: pd.DataFrame) -> pd.DataFrame:
    keys = wide_df[["horizon", "outer_fold", "feature_date", "target_date"]].drop_duplicates()
    keys = keys.sort_values(["outer_fold", "target_date", "feature_date"]).reset_index(drop=True)
    keys["canonical_row_id"] = np.arange(1, len(keys) + 1, dtype=int)
    out = wide_df.merge(keys, on=["horizon", "outer_fold", "feature_date", "target_date"], how="left", validate="one_to_one")
    return out.sort_values(["outer_fold", "target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)


def read_outputs() -> Dict[str, Any]:
    out = {
        "input_verification": json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8")),
        "startup": pd.read_csv(STARTUP_REGISTRY_PATH),
        "component": pd.read_csv(CAUSAL_COMPONENT_RANKS_PATH),
        "ensemble": pd.read_csv(CAUSAL_FIXED_ENSEMBLE_PATH),
        "decisions": pd.read_csv(SEQUENTIAL_DECISIONS_PATH),
        "cutoff_audit": pd.read_csv(CUTOFF_AUDIT_PATH),
        "fold_metrics": pd.read_csv(METRICS_BY_FOLD_PATH),
        "pooled_metrics": pd.read_csv(METRICS_POOLED_PATH),
        "secondary": pd.read_csv(FULL_FOLD_SECONDARY_PATH),
        "burden": pd.read_csv(BURDEN_AUDIT_PATH),
        "definition_md": SEQ_VS_RETRO_PATH.read_text(encoding="utf-8"),
        "info_isolation": json.loads(INFO_ISOLATION_PATH.read_text(encoding="utf-8")),
        "coverage": pd.read_csv(DATE_COVERAGE_AUDIT_PATH),
        "manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        "report": COMPLETION_REPORT_PATH.read_text(encoding="utf-8"),
        "checksums": parse_checksum_manifest(CHECKSUMS_PATH),
    }

    for c in ["feature_date", "target_date", "cutoff_history_start", "cutoff_history_end"]:
        if c in out["decisions"].columns:
            out["decisions"][c] = normalize_date_col(out["decisions"][c])

    for c in ["feature_date", "target_date", "rank_history_start", "rank_history_end"]:
        if c in out["component"].columns:
            out["component"][c] = normalize_date_col(out["component"][c])

    for c in ["feature_date", "target_date"]:
        out["ensemble"][c] = normalize_date_col(out["ensemble"][c])

    return out


def deterministic_snapshot(paths: Sequence[Path]) -> Dict[str, str]:
    return {p.name: sha256_file(p) for p in paths}


def update_manifest_and_report(
    manifest: Dict[str, Any],
    report_text: str,
    test_result: str,
    deterministic_result: str,
    final_decision: str,
    assertions_total: int,
    assertions_passed: int,
) -> None:
    manifest["test_result"] = test_result
    manifest["deterministic_result"] = deterministic_result
    manifest["final_decision"] = final_decision
    manifest["assertions_total"] = int(assertions_total)
    manifest["assertions_passed"] = int(assertions_passed)
    manifest["assertions_failed"] = int(assertions_total - assertions_passed)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = report_text.splitlines()

    # deterministic line
    det_done = False
    for i, ln in enumerate(lines):
        if "deterministic_result:" in ln:
            lines[i] = "- deterministic_result: {}".format(deterministic_result)
            det_done = True
    if not det_done:
        lines.append("- deterministic_result: {}".format(deterministic_result))

    # test line
    test_done = False
    for i, ln in enumerate(lines):
        if "test_result:" in ln:
            lines[i] = "- test_result: {}".format(test_result)
            test_done = True
    if not test_done:
        lines.append("- test_result: {}".format(test_result))

    # final decision
    decision_line = "A. Corrected sequential H5 alarm-policy simulation passed; H1/H3 canonical reconciliation may begin."
    if final_decision == "B":
        decision_line = "B. Sequential simulation completed, but alarm-burden or calibration behavior requires investigation."
    elif final_decision == "C":
        decision_line = "C. Past-only score calibration could not be implemented consistently for all authorized models."
    elif final_decision == "D":
        decision_line = "D. Temporal causality, information isolation, source preservation, or deterministic tests failed."

    final_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "FINAL DECISION":
            final_idx = i
            break

    if final_idx is not None:
        replaced = False
        for j in range(final_idx + 1, min(final_idx + 8, len(lines))):
            if re.match(r"^[ABCD]\.\s", lines[j].strip()):
                lines[j] = decision_line
                replaced = True
                break
        if not replaced:
            lines.insert(final_idx + 2, decision_line)
    else:
        lines.extend(["", "FINAL DECISION", "", decision_line])

    COMPLETION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_checksums() -> None:
    lines: List[str] = []
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.name == CHECKSUMS_PATH.name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        lines.append("{}  {}".format(sha256_file(p), p.name))
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from workspace root: {}".format(ROOT))

    run_build_script()

    ac = AssertionCollector()
    out = read_outputs()

    # Reference reads
    wide = pd.read_csv(CANONICAL_WIDE_PATH)
    wide["feature_date"] = normalize_date_col(wide["feature_date"])
    wide["target_date"] = normalize_date_col(wide["target_date"])
    wide = add_canonical_row_id(wide)

    decisions = out["decisions"].copy()
    component = out["component"].copy()
    ensemble = out["ensemble"].copy()
    startup = out["startup"].copy()
    cutoff_audit = out["cutoff_audit"].copy()
    fold_metrics = out["fold_metrics"].copy()
    pooled_metrics = out["pooled_metrics"].copy()
    secondary = out["secondary"].copy()
    info = out["info_isolation"]
    coverage = out["coverage"].copy()
    manifest = out["manifest"]
    report = out["report"]

    # 1-8
    ac.check(Path.cwd().resolve() == ROOT.resolve(), "1. working directory mismatch")
    ac.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "2. branch mismatch")
    ac.check(sys.version_info.major == 3 and sys.version_info.minor == 8, "3. Python 3.8 compatibility failed")

    checks_ok = (
        verify_checksum_manifest(PROTOCOL_SHA_PATH, PROTOCOL_DIR)
        and verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)
        and verify_checksum_manifest(FIXED_CHECKSUMS_PATH, FIXED_DIR)
        and verify_checksum_manifest(RETRO_CHECKSUMS_PATH, RETRO_DIR)
    )
    # Also verify recorded manifest input checksums.
    manifest_inputs_ok = True
    for p, sha in manifest.get("input_checksums", {}).items():
        pp = Path(str(p))
        if (not pp.exists()) or (sha256_file(pp) != str(sha)):
            manifest_inputs_ok = False
            break
    ac.check(checks_ok and manifest_inputs_ok, "4. input checksums failed")

    ac.check(infer_decision_from_report(CANONICAL_REPORT_PATH) == "A", "5. canonical decision is not A")
    ac.check(infer_decision_from_report(FIXED_REPORT_PATH) == "A", "6. fixed decision is not A")
    ac.check(infer_decision_from_report(RETRO_REPORT_PATH) == "A", "7. retrospective decision is not A")
    ac.check(infer_decision_from_report(FULLY_NESTED_REPORT_PATH) == "C", "8. fully nested tuned decision is not C")

    # 9-20
    ac.check(sorted(decisions["horizon"].astype(int).unique().tolist()) == [5], "9. not H5-only")
    ac.check(int(len(wide)) == 747, "10. canonical N != 747")
    ac.check(bool((wide.groupby("outer_fold").size().astype(int) == 249).all()), "11. fold N != 249")
    ac.check(sorted(decisions["model"].astype(str).unique().tolist()) == sorted(EXPECTED_MODELS), "12. unauthorized models present")
    ac.check(not decisions["model"].astype(str).str.contains("point", case=False).any(), "13. point blend detected")
    ac.check(not decisions["model"].astype(str).str.contains("tuned", case=False).any(), "14. tuned hybrid detected")
    ac.check(sorted(decisions["threshold"].astype(int).unique().tolist()) == EXPECTED_THRESHOLDS, "15. thresholds mismatch")
    ac.check(sorted(np.round(decisions["nominal_budget_r"].astype(float).unique(), 10).tolist()) == EXPECTED_BUDGETS, "16. budgets mismatch")
    ac.check(int(startup["startup_N"].astype(int).unique()[0]) == STARTUP_DAYS, "17. startup length is not 30")
    ac.check(bool((startup["eligible_N"].astype(int) == 219).all()), "18. eligible N not 219 per fold")
    ac.check(bool((fold_metrics["eligible_N"].astype(int) == 219).all()) and int(pooled_metrics["pooled_eligible_N"].astype(int).unique()[0]) == 657, "19. pooled eligible N != 657")

    chronological_ok = True
    for _, g in decisions.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        gs = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        if not pd.to_datetime(gs["target_date"], errors="coerce").is_monotonic_increasing:
            chronological_ok = False
            break
    ac.check(chronological_ok, "20. dates are not chronological")

    # 21-24
    eligible = decisions[decisions["evaluation_eligible"].astype(bool)].copy()
    cutoff_end = pd.to_datetime(eligible["cutoff_history_end"], errors="coerce")
    target_dates = pd.to_datetime(eligible["target_date"], errors="coerce")
    ac.check(bool((cutoff_end < target_dates).all()), "21. current score appears in cutoff history")
    ac.check(
        bool((decisions["future_scores_used"].astype(str).str.lower() == "false").all())
        and bool((cutoff_audit["future_score_access"].astype(str).str.lower() == "false").all()),
        "22. future score access flag violation",
    )
    ac.check(
        bool((decisions["labels_used_for_score"].astype(str).str.lower() == "false").all())
        and bool((component["y_true_used_for_rank"].astype(str).str.lower() == "false").all())
        and bool((component["event_labels_used_for_rank"].astype(str).str.lower() == "false").all()),
        "23. labels used in score construction",
    )
    ac.check(
        bool((decisions["labels_used_for_cutoff"].astype(str).str.lower() == "false").all())
        and bool((cutoff_audit["labels_used"].astype(str).str.lower() == "false").all()),
        "24. labels used in cutoff calibration",
    )

    # 25-27 mapping and weights
    mapping_ok = True
    bcr_mapping_ok = True
    base_map = {
        "Persistence": "persistence_y_pred",
        "Ridge": "ridge_y_pred",
        "ElasticNet": "elasticnet_y_pred",
        "HGBR": "hgbr_y_pred",
    }

    for model_name, col in base_map.items():
        dsub = decisions[
            (decisions["model"].astype(str) == model_name)
            & np.isclose(decisions["nominal_budget_r"].astype(float), 0.05)
        ].copy()
        m = dsub.merge(
            wide[["outer_fold", "feature_date", "target_date", "canonical_row_id", col]],
            on=["outer_fold", "feature_date", "target_date", "canonical_row_id"],
            how="left",
            validate="many_to_one",
        )
        if m[col].isna().any() or (not np.allclose(m["raw_score"].astype(float).to_numpy(), m[col].astype(float).to_numpy(), atol=1e-12, rtol=0.0)):
            mapping_ok = False
            break
    ac.check(mapping_ok, "25. base-model score mapping incorrect")

    for tau in EXPECTED_THRESHOLDS:
        dsub = decisions[
            (decisions["model"].astype(str) == "BCR-TCN v1.1")
            & (decisions["threshold"].astype(int) == int(tau))
            & np.isclose(decisions["nominal_budget_r"].astype(float), 0.05)
        ].copy()
        col = "bcr_tcn_v11_p_tau{}".format(int(tau))
        m = dsub.merge(
            wide[["outer_fold", "feature_date", "target_date", "canonical_row_id", col]],
            on=["outer_fold", "feature_date", "target_date", "canonical_row_id"],
            how="left",
            validate="many_to_one",
        )
        if m[col].isna().any() or (not np.allclose(m["raw_score"].astype(float).to_numpy(), m[col].astype(float).to_numpy(), atol=1e-12, rtol=0.0)):
            bcr_mapping_ok = False
            break
    ac.check(bcr_mapping_ok, "26. BCR-TCN threshold mapping incorrect")

    w_ok = True
    expected_weights = {
        "BCR-TCN v1.1": 0.50,
        "ElasticNet": 0.25,
        "Persistence": 0.25,
        "HGBR": 0.00,
    }
    for comp_name, w in expected_weights.items():
        vals = component[component["component"].astype(str) == comp_name]["component_weight"].astype(float).unique()
        if len(vals) != 1 or not np.isclose(float(vals[0]), float(w), atol=1e-12, rtol=0.0):
            w_ok = False
            break
    ac.check(w_ok, "27. causal ensemble weights incorrect")

    # 28-31 causal rank integrity
    prior_only_ok = True
    tie_formula_ok = True
    first_rank_ok = True

    for _, g in component.groupby(["component", "outer_fold", "threshold"], sort=True):
        gs = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        vals = gs["component_raw_score"].astype(float).to_numpy()
        ranks = gs["causal_rank"].astype(float).to_numpy()
        past_counts = gs["past_score_count"].astype(int).to_numpy()

        if int(past_counts[0]) != 0:
            prior_only_ok = False
        if not np.isclose(float(ranks[0]), 0.5, atol=1e-12, rtol=0.0):
            first_rank_ok = False

        for i in range(len(gs)):
            if int(past_counts[i]) != int(i):
                prior_only_ok = False
                break
            if i > 0:
                expected_end = pd.Timestamp(gs.loc[i - 1, "target_date"])
                observed_end = pd.Timestamp(gs.loc[i, "rank_history_end"])
                if observed_end != expected_end:
                    prior_only_ok = False
                    break
            expected_rank = 0.5
            if i > 0:
                hist = vals[:i]
                expected_rank = (float((hist < vals[i]).sum()) + 0.5 * float((hist == vals[i]).sum())) / float(i)
            if not np.isclose(float(ranks[i]), float(expected_rank), atol=1e-12, rtol=0.0):
                tie_formula_ok = False
                break

    ac.check(prior_only_ok, "28. causal ranks are not strictly prior-score based")
    ac.check(tie_formula_ok, "29. causal tie formula mismatch")
    ac.check(first_rank_ok, "30. first causal rank is not 0.5")
    ac.check(
        bool(np.isfinite(component["causal_rank"].astype(float)).all())
        and bool(((component["causal_rank"].astype(float) >= 0.0) & (component["causal_rank"].astype(float) <= 1.0)).all())
        and bool(np.isfinite(ensemble["policy_score"].astype(float)).all())
        and bool(((ensemble["policy_score"].astype(float) >= 0.0) & (ensemble["policy_score"].astype(float) <= 1.0)).all()),
        "31. causal ranks/scores not finite in [0,1]",
    )

    # 32-36 cutoff and alarm rules
    cutoff_rule_ok = True
    for _, g in decisions.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        gs = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        history: List[float] = []
        r = float(gs.loc[0, "nominal_budget_r"])
        for i, row in gs.iterrows():
            score = float(row["policy_score"])
            if bool(row["evaluation_eligible"]):
                q = float(1.0 - r)
                expected_cutoff = quantile_higher(
                    np.asarray(history, dtype=float),
                    q,
                )
                if not np.isclose(float(row["cutoff_score"]), expected_cutoff, atol=1e-12, rtol=0.0):
                    cutoff_rule_ok = False
                    break
            history.append(score)
        if not cutoff_rule_ok:
            break
    ac.check(cutoff_rule_ok, "32. higher-quantile cutoff rule failed")

    elig = decisions[decisions["evaluation_eligible"].astype(bool)].copy()
    ac.check(
        np.array_equal(
            elig["alarm_flag"].astype(int).to_numpy(),
            (elig["policy_score"].astype(float).to_numpy() > elig["cutoff_score"].astype(float).to_numpy()).astype(int),
        ),
        "33. alarm rule is not score>cutoff",
    )
    eq_rows = decisions[decisions["equal_to_cutoff"].astype(bool)].copy()
    ac.check(bool((eq_rows["alarm_flag"].astype(int) == 0).all()), "34. equality to cutoff triggered alarms")
    ac.check(int(decisions[decisions["startup_calibration"].astype(bool)]["alarm_flag"].astype(int).sum()) == 0, "35. startup rows have alarms")
    ac.check(bool(np.isfinite(elig["cutoff_score"].astype(float)).all()), "36. missing or non-finite cutoff on eligible rows")

    # 37-46 metrics and baseline constraints
    grouped_alarms = fold_metrics[["model", "outer_fold", "threshold", "nominal_budget_r", "alarms_issued", "eligible_N"]].copy()
    grouped_alarms["nominal_k_like"] = np.ceil(grouped_alarms["nominal_budget_r"].astype(float) * grouped_alarms["eligible_N"].astype(float)).astype(int)
    ac.check(bool((grouped_alarms["alarms_issued"].astype(int) != grouped_alarms["nominal_k_like"].astype(int)).any()), "37. alarm count appears forced to nominal k")

    realized_ok = np.allclose(
        fold_metrics["realized_alarm_fraction"].astype(float).to_numpy(),
        (fold_metrics["alarms_issued"].astype(float).to_numpy() / fold_metrics["eligible_N"].astype(float).to_numpy()),
        atol=1e-12,
        rtol=0.0,
    )
    ac.check(realized_ok, "38. realized alarm fraction formula mismatch")

    counts_sum_ok = np.array_equal(
        (fold_metrics["TP"].astype(int) + fold_metrics["FP"].astype(int) + fold_metrics["FN"].astype(int) + fold_metrics["TN"].astype(int)).to_numpy(),
        fold_metrics["eligible_N"].astype(int).to_numpy(),
    )
    ac.check(counts_sum_ok, "39. confusion counts do not sum to eligible N")

    precision_ok = True
    recall_ok = True
    undefined_ok = True
    for row in fold_metrics.itertuples(index=False):
        alarms = int(row.alarms_issued)
        events = int(row.eligible_events)
        tp = int(row.TP)

        expected_prec = float(tp) / float(alarms) if alarms > 0 else float("nan")
        expected_rec = float(tp) / float(events) if events > 0 else float("nan")

        obs_prec = float(row.precision) if not pd.isna(row.precision) else float("nan")
        obs_rec = float(row.recall) if not pd.isna(row.recall) else float("nan")

        if alarms > 0 and (not np.isclose(obs_prec, expected_prec, atol=1e-12, rtol=0.0)):
            precision_ok = False
        if alarms == 0 and (not pd.isna(row.precision)):
            undefined_ok = False

        if events > 0 and (not np.isclose(obs_rec, expected_rec, atol=1e-12, rtol=0.0)):
            recall_ok = False
        if events == 0 and (not pd.isna(row.recall)):
            undefined_ok = False

    ac.check(precision_ok, "40. precision denominator mismatch")
    ac.check(recall_ok, "41. recall denominator mismatch")
    ac.check(undefined_ok, "42. undefined metrics are not explicitly undefined")

    pooled_ok = True
    merged = pooled_metrics.merge(
        fold_metrics.groupby(["model", "horizon", "threshold", "nominal_budget_r"], as_index=False).agg(
            pooled_eligible_N=("eligible_N", "sum"),
            pooled_eligible_events=("eligible_events", "sum"),
            pooled_alarms=("alarms_issued", "sum"),
            pooled_TP=("TP", "sum"),
            pooled_FP=("FP", "sum"),
            pooled_FN=("FN", "sum"),
            pooled_TN=("TN", "sum"),
        ),
        on=["model", "horizon", "threshold", "nominal_budget_r"],
        how="inner",
        suffixes=("", "_re"),
    )
    if int(len(merged)) != 36:
        pooled_ok = False
    else:
        for col in ["pooled_eligible_N", "pooled_eligible_events", "pooled_alarms", "pooled_TP", "pooled_FP", "pooled_FN", "pooled_TN"]:
            if not np.array_equal(merged[col].astype(int).to_numpy(), merged[col + "_re"].astype(int).to_numpy()):
                pooled_ok = False
                break
    ac.check(pooled_ok, "43. pooled metrics are not based on pooled counts")

    ac.check(
        bool((fold_metrics["startup_N"].astype(int) == 30).all()) and bool((fold_metrics["eligible_N"].astype(int) == 219).all()),
        "44. startup and eligible metrics are not properly separated",
    )
    ac.check(
        bool((secondary["metric_scope"].astype(str) == "secondary_full_fold_with_startup_no_alarms").all())
        and bool((secondary["primary_reference"].astype(str) == "primary_post_startup").all()),
        "45. full-fold secondary metrics are not labeled separately",
    )
    ac.check(
        ("random_expected_recall" not in fold_metrics.columns)
        and ("random_expected_recall" not in pooled_metrics.columns)
        and ("K/N random baseline" not in report),
        "46. retrospective random baseline appears applied to sequential metrics",
    )

    # 47-50 information isolation
    ac.check(bool(info.get("label_invariance_pass", False)), "47. label invariance failed")
    ac.check(bool(info.get("future_score_invariance_pass", False)), "48. future-score invariance failed")
    ac.check(bool(info.get("prefix_causality_pass", False)), "49. prefix causality failed")
    ac.check(float(info.get("maximum_earlier_decision_difference", 1.0)) == 0.0, "50. future perturbation changed earlier decision")

    # 51-57 no training/tuning/winner selection
    build_text = BUILD_SCRIPT.read_text(encoding="utf-8")

    has_fit_call = bool(re.search(r"\bfit\s*\(", build_text))
    has_partial_fit_call = bool(re.search(r"\bpartial_fit\s*\(", build_text))
    ac.check((not has_fit_call) and (not has_partial_fit_call), "51. model training call detected")
    ac.check((not has_fit_call) and (not has_partial_fit_call), "52. estimator fit/partial_fit detected")
    ac.check("backward" not in build_text.lower() and "optimizer" not in build_text.lower(), "53. optimizer/backward keyword detected")
    ac.check("optuna" not in build_text.lower(), "54. optuna keyword detected")
    ac.check("FIXED_WEIGHTS" in build_text and "weight_search" not in build_text.lower(), "55. weight search behavior detected")
    ac.check(bool((decisions["labels_used_for_cutoff"].astype(str).str.lower() == "false").all()), "56. cutoff appears tuned from test outcomes")
    ac.check("No model or policy winner was automatically selected." in report, "57. winner-selection guard statement missing")

    # 58-63 source, rows, labels, directory
    source_unchanged_ok = True
    for p, sha in manifest.get("input_checksums", {}).items():
        pp = Path(str(p))
        if (not pp.exists()) or (sha256_file(pp) != str(sha)):
            source_unchanged_ok = False
            break
    ac.check(source_unchanged_ok, "58. source package checksum drift detected")

    ac.check(int(len(decisions)) == 26892, "59. decision row count is not 26,892")
    ac.check(bool((decisions["score_context"].astype(str) == EXPECTED_SCORE_CONTEXT).all()), "60. score_context label mismatch")
    ac.check(not decisions["score_context"].astype(str).str.contains("prospective field validation", case=False).any(), "61. prospective field validation label detected")
    ac.check(not decisions["score_context"].astype(str).str.contains("real-time deployment", case=False).any(), "62. real-time deployment label detected")

    outputs_in_dir_ok = True
    for name in manifest.get("output_files", []):
        p = OUT_DIR / str(name)
        if not p.exists() or p.resolve().parent != OUT_DIR.resolve():
            outputs_in_dir_ok = False
            break
    ac.check(outputs_in_dir_ok, "63. output files are not confined to authorized directory")

    # 64-65 deterministic regeneration + stable checksums
    stable_files = [
        STARTUP_REGISTRY_PATH,
        CAUSAL_COMPONENT_RANKS_PATH,
        CAUSAL_FIXED_ENSEMBLE_PATH,
        SEQUENTIAL_DECISIONS_PATH,
        CUTOFF_AUDIT_PATH,
        METRICS_BY_FOLD_PATH,
        METRICS_POOLED_PATH,
        FULL_FOLD_SECONDARY_PATH,
        BURDEN_AUDIT_PATH,
        SEQ_VS_RETRO_PATH,
        INFO_ISOLATION_PATH,
        DATE_COVERAGE_AUDIT_PATH,
        BUILD_SCRIPT,
        TEST_SCRIPT,
    ]

    snap1 = deterministic_snapshot(stable_files)
    checks1 = parse_checksum_manifest(CHECKSUMS_PATH)
    stable_names = set([p.name for p in stable_files])
    stable_checks1 = {k: v for k, v in checks1.items() if k in stable_names}

    run_build_script()

    snap2 = deterministic_snapshot(stable_files)
    checks2 = parse_checksum_manifest(CHECKSUMS_PATH)
    stable_checks2 = {k: v for k, v in checks2.items() if k in stable_names}

    deterministic_pass = bool(snap1 == snap2)
    stable_checks_pass = bool(stable_checks1 == stable_checks2)

    ac.check(deterministic_pass, "64. repeated generation is not deterministic for stable outputs")
    ac.check(stable_checks_pass, "65. stable output checksums do not match across reruns")

    assertions_total = ac.total
    assertions_passed = ac.total - ac.failed

    test_result = "pass_65_of_65" if ac.failed == 0 else "fail_{}_of_65".format(ac.failed)
    deterministic_result = "pass" if deterministic_pass and stable_checks_pass else "fail"
    final_decision = "A" if (ac.failed == 0 and deterministic_result == "pass") else "D"

    latest_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    latest_report = COMPLETION_REPORT_PATH.read_text(encoding="utf-8")
    update_manifest_and_report(
        manifest=latest_manifest,
        report_text=latest_report,
        test_result=test_result,
        deterministic_result=deterministic_result,
        final_decision=final_decision,
        assertions_total=assertions_total,
        assertions_passed=assertions_passed,
    )

    # Refresh manifest output checksums and checksums registry.
    latest_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output_checksums: Dict[str, str] = {}
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        if p.name == CHECKSUMS_PATH.name:
            continue
        output_checksums[p.name] = sha256_file(p)
    latest_manifest["output_files"] = sorted(output_checksums.keys())
    latest_manifest["output_checksums"] = output_checksums
    MANIFEST_PATH.write_text(json.dumps(latest_manifest, indent=2), encoding="utf-8")

    rewrite_checksums()

    print("Total assertions: {}".format(assertions_total))
    print("Passed assertions: {}".format(assertions_passed))
    print("Failed assertions: {}".format(ac.failed))
    print("Deterministic result: {}".format(deterministic_result))
    print("Final decision: {}".format(final_decision))

    if ac.failed > 0:
        print("Failures:")
        for f in ac.failures:
            print("- {}".format(f))

    if ac.failed > 0 or deterministic_result != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
