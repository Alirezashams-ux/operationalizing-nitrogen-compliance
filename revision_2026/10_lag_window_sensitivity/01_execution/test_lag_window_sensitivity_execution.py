from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "run_lag_window_sensitivity_execution.py"

EXPECTED_REPOSITORY = "/home/alrezshams/acs_tnout_ulsan_revision"
EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_HEAD = "dee06bcfeca0c3c224897a2a92261c13f4fea26e"
EXPECTED_DESIGN_TAG = "lag-window-sensitivity-design-v1"
EXPECTED_DESIGN_COMMIT = EXPECTED_HEAD
EXPECTED_PYTHON = "3.8.10"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions"
HYBRID_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank"
RETRO_DIR = ROOT / "revision_2026" / "07_retrospective_alarm_budget"
SEQUENTIAL_DIR = ROOT / "revision_2026" / "08_sequential_alarm_policy"
STAGE1_DIR = ROOT / "revision_2026" / "09_h1_h3_reconciliation"
DESIGN_DIR = ROOT / "revision_2026" / "10_lag_window_sensitivity" / "00_design"

PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
STAGE1_CHECKSUMS_PATH = STAGE1_DIR / "canonical_h1_h3_reconciliation_checksums.sha256"
DESIGN_CHECKSUMS_PATH = DESIGN_DIR / "lag_window_sensitivity_design_checksums.sha256"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
SOURCE_REGISTRY_PATH = OUT_DIR / "execution_source_registry.csv"
SOFTWARE_ENV_PATH = OUT_DIR / "software_environment.json"
FIT_REGISTRY_PATH = OUT_DIR / "model_fit_registry.csv"
PRED_LONG_PATH = OUT_DIR / "lag_window_predictions_long.csv"
PRED_WIDE_PATH = OUT_DIR / "lag_window_predictions_wide.csv"
REF_AUDIT_PATH = OUT_DIR / "reference_reproduction_audit.csv"
REF_SUMMARY_PATH = OUT_DIR / "reference_reproduction_summary.json"
POINT_BY_FOLD_PATH = OUT_DIR / "point_metrics_by_fold.csv"
POINT_POOLED_PATH = OUT_DIR / "point_metrics_pooled.csv"
STABILITY_BY_FOLD_PATH = OUT_DIR / "prediction_stability_by_fold.csv"
STABILITY_POOLED_PATH = OUT_DIR / "prediction_stability_pooled.csv"
RECALL_BY_FOLD_PATH = OUT_DIR / "event_ranking_recall5_by_fold.csv"
RECALL_POOLED_PATH = OUT_DIR / "event_ranking_recall5_pooled.csv"
RETENTION_REALIZED_PATH = OUT_DIR / "realized_training_retention.csv"
FEATURE_REG_REALIZED_PATH = OUT_DIR / "realized_feature_registry.csv"
SPLIT_EMBARGO_AUDIT_PATH = OUT_DIR / "split_and_embargo_audit.csv"
SUMMARY_CSV_PATH = OUT_DIR / "lag_window_sensitivity_summary.csv"
SUMMARY_JSON_PATH = OUT_DIR / "lag_window_sensitivity_summary.json"
SUMMARY_MD_PATH = OUT_DIR / "lag_window_sensitivity_summary.md"
MANIFEST_PATH = OUT_DIR / "lag_window_sensitivity_execution_manifest.json"
REPORT_PATH = OUT_DIR / "lag_window_sensitivity_execution_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "lag_window_sensitivity_execution_checksums.sha256"

IMMUTABLE_DIRS = [
    ROOT / "revision_2026" / "03_corrected_protocol",
    ROOT / "revision_2026" / "04_controlled_reruns",
    ROOT / "revision_2026" / "05_canonical_predictions",
    ROOT / "revision_2026" / "06_corrected_hybridrank",
    ROOT / "revision_2026" / "07_retrospective_alarm_budget",
    ROOT / "revision_2026" / "08_sequential_alarm_policy",
    ROOT / "revision_2026" / "09_h1_h3_reconciliation",
    ROOT / "revision_2026" / "10_lag_window_sensitivity" / "00_design",
    ROOT / "src",
    ROOT / "features",
    ROOT / "results",
]

FIT_MODELS = ["Ridge", "HGBR"]
HORIZONS = ["H1", "H3", "H5"]
CONFIGS = ["SHORT", "REFERENCE", "LONG"]
FINAL_REF_MAP = {
    "Ridge_H1": "REFERENCE",
    "Ridge_H3": "REFERENCE",
    "Ridge_H5": "REFERENCE",
    "HGBR_H1": "REFERENCE",
    "HGBR_H3": "REFERENCE",
    "HGBR_H5": "LONG",
}


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


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise RuntimeError("malformed checksum line in {}".format(path))
        out[parts[-1]] = parts[0]
    return out


def verify_checksum_manifest(path: Path, base_dir: Path) -> bool:
    for rel, expected in parse_checksum_manifest(path).items():
        target = base_dir / rel
        if not target.exists():
            return False
        if sha256_file(target) != expected:
            return False
    return True


def snapshot_tree_checksums(dirs: Sequence[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            raise RuntimeError("immutable directory missing: {}".format(d))
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def deterministic_snapshot(out_dir: Path) -> Dict[str, str]:
    checks: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        if p.name == CHECKSUMS_PATH.name:
            continue
        checks[p.name] = sha256_file(p)
    return checks


def rewrite_checksums() -> None:
    lines: List[str] = []
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        if p.name == CHECKSUMS_PATH.name:
            continue
        lines.append("{}  {}".format(sha256_file(p), p.name))
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def checksum_coverage_ok() -> Dict[str, Any]:
    listed = parse_checksum_manifest(CHECKSUMS_PATH)
    listed_files = sorted(list(listed.keys()))
    actual_files = sorted(
        [
            p.name
            for p in OUT_DIR.iterdir()
            if p.is_file() and p.name != CHECKSUMS_PATH.name and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
        ]
    )
    missing = sorted(set(actual_files).difference(set(listed_files)))
    extra = sorted(set(listed_files).difference(set(actual_files)))

    registry_ok = True
    for rel, expected in listed.items():
        target = OUT_DIR / rel
        if not target.exists() or sha256_file(target) != expected:
            registry_ok = False
            break

    return {
        "registry_pass": bool(registry_ok),
        "coverage_pass": (len(missing) == 0 and len(extra) == 0),
        "coverage_missing": missing,
        "coverage_extra": extra,
    }


def run_sha256_check() -> Dict[str, Any]:
    proc = subprocess.run(
        ["sha256sum", "-c", CHECKSUMS_PATH.name],
        cwd=OUT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run_builder() -> None:
    subprocess.check_call([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT)


def read_outputs() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "input_verification": json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8")),
        "source_registry": pd.read_csv(SOURCE_REGISTRY_PATH),
        "software_env": json.loads(SOFTWARE_ENV_PATH.read_text(encoding="utf-8")),
        "fit_registry": pd.read_csv(FIT_REGISTRY_PATH),
        "pred_long": pd.read_csv(PRED_LONG_PATH),
        "pred_wide": pd.read_csv(PRED_WIDE_PATH),
        "ref_audit": pd.read_csv(REF_AUDIT_PATH),
        "ref_summary": json.loads(REF_SUMMARY_PATH.read_text(encoding="utf-8")),
        "point_fold": pd.read_csv(POINT_BY_FOLD_PATH),
        "point_pooled": pd.read_csv(POINT_POOLED_PATH),
        "stab_fold": pd.read_csv(STABILITY_BY_FOLD_PATH),
        "stab_pooled": pd.read_csv(STABILITY_POOLED_PATH),
        "recall_fold": pd.read_csv(RECALL_BY_FOLD_PATH),
        "recall_pooled": pd.read_csv(RECALL_POOLED_PATH),
        "retention": pd.read_csv(RETENTION_REALIZED_PATH),
        "feature_registry": pd.read_csv(FEATURE_REG_REALIZED_PATH),
        "split_embargo": pd.read_csv(SPLIT_EMBARGO_AUDIT_PATH),
        "summary_csv": pd.read_csv(SUMMARY_CSV_PATH),
        "summary_json": json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8")),
        "summary_md": SUMMARY_MD_PATH.read_text(encoding="utf-8"),
        "manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        "report": REPORT_PATH.read_text(encoding="utf-8"),
    }

    for c in ["feature_date", "target_date"]:
        out["pred_long"][c] = normalize_date_col(out["pred_long"][c])
        out["pred_wide"][c] = normalize_date_col(out["pred_wide"][c])

    return out


def write_manifest_and_report(
    manifest: Dict[str, Any],
    final_decision: str,
    test_result: str,
    deterministic_result: str,
) -> None:
    manifest["test_result"] = test_result
    manifest["deterministic_result"] = deterministic_result
    manifest["final_decision"] = final_decision
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gate = {
        "A": "A. Controlled lag-window sensitivity execution passed; results are valid for final computational integration.",
        "B": "B. Execution completed, but material sensitivity or reference-reproduction questions require investigation before integration.",
        "C": "C. Canonical dates, embargoes, frozen hyperparameters, or reference lineage could not be preserved safely.",
        "D": "D. Source preservation, deterministic regeneration, tests, or checksum validation failed.",
    }

    lines = [
        "# Stage 2 Lag-Window Sensitivity Execution Completion Report",
        "",
        "## Gate Summary",
        "- preflight_pass: {}".format(bool(manifest.get("input_checksum_result", {}).get("corrected_protocol_checksum_pass", False))),
        "- source_preservation_pass: {}".format(bool(manifest.get("source_preservation_result", {}).get("pass", False))),
        "- input_checksum_pass: {}".format(bool(manifest.get("input_checksum_result", {}).get("corrected_protocol_checksum_pass", False))),
        "- design_checksum_pass: {}".format(bool(manifest.get("design_checksum_result", {}).get("design_checksum_pass", False))),
        "- common_date_pass: {}".format(bool(manifest.get("common_date_result", {}).get("pass", False))),
        "- embargo_pass: {}".format(bool(manifest.get("embargo_result", {}).get("pass", False))),
        "- training_retention_pass: {}".format(bool(manifest.get("training_retention_result", {}).get("pass", False))),
        "- reference_reproduction_pass: {}".format(bool(manifest.get("reference_reproduction_result", {}).get("overall_pass", False))),
        "- test_result: {}".format(test_result),
        "- deterministic_result: {}".format(deterministic_result),
        "",
        "## Scope Labels",
        "- retrospective",
        "- offline",
        "- fixed-budget",
        "- sensitivity evidence only",
        "",
        "FINAL DECISION",
        "",
        gate[final_decision],
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("run this script from workspace root: {}".format(ROOT))

    ac = AssertionCollector()

    pre_snapshot = snapshot_tree_checksums(IMMUTABLE_DIRS)

    run_builder()
    first_hash = deterministic_snapshot(OUT_DIR)

    run_builder()
    second_hash = deterministic_snapshot(OUT_DIR)

    det_files = sorted(set(first_hash.keys()).union(set(second_hash.keys())))
    differences = [f for f in det_files if first_hash.get(f) != second_hash.get(f)]
    deterministic_pass = len(differences) == 0

    post_snapshot = snapshot_tree_checksums(IMMUTABLE_DIRS)
    source_preservation_pass = pre_snapshot == post_snapshot

    out = read_outputs()

    pred = out["pred_long"].copy()
    point_fold = out["point_fold"].copy()
    point_pooled = out["point_pooled"].copy()
    stab_fold = out["stab_fold"].copy()
    stab_pooled = out["stab_pooled"].copy()
    recall_fold = out["recall_fold"].copy()
    recall_pooled = out["recall_pooled"].copy()
    retention = out["retention"].copy()
    fit_registry = out["fit_registry"].copy()
    feat_reg = out["feature_registry"].copy()
    ref_audit = out["ref_audit"].copy()
    split_embargo = out["split_embargo"].copy()
    manifest = out["manifest"]

    split = pd.read_csv(SPLIT_PATH)
    split["horizon"] = split["horizon"].astype(int)
    split["outer_fold"] = split["outer_fold"].astype(int)
    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])

    pred["horizon_int"] = pred["horizon"].astype(str).str.replace("H", "", regex=False).astype(int)

    # 1-10
    ac.check(out["input_verification"].get("repository") == EXPECTED_REPOSITORY, "1. repository root mismatch")
    ac.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "2. branch identity mismatch")
    ac.check(git_output(["git", "rev-parse", "HEAD"]) == EXPECTED_HEAD, "3. starting commit mismatch")
    ac.check(out["input_verification"].get("design_tag_commit") == EXPECTED_DESIGN_COMMIT, "3b. design tag commit mismatch")
    ac.check(sys.version.split()[0] == EXPECTED_PYTHON, "4. Python version mismatch")
    ac.check(bool(out["input_verification"].get("checks", {}).get("clean_initial_source_state_pass", False)), "5. clean initial source state gate failed")
    ac.check(bool(out["input_verification"].get("checks", {}).get("corrected_protocol_checksum_pass", False)), "6. corrected protocol checksums failed")
    ac.check(bool(out["input_verification"].get("checks", {}).get("stage1_checksum_pass", False)), "7. Stage 1 checksums failed")
    ac.check(bool(out["input_verification"].get("checks", {}).get("design_checksum_pass", False)), "8. design checksums failed")
    ac.check(bool(manifest.get("source_preservation_result", {}).get("pass", False)), "9. source preservation result failed in manifest")
    ac.check(source_preservation_pass, "10. immutable tree changed outside authorized execution output")

    # 11-20
    ac.check(sorted(pred["model"].astype(str).unique().tolist()) == sorted(FIT_MODELS), "11. exact model set mismatch")
    ac.check(sorted(pred["horizon"].astype(str).unique().tolist()) == sorted(HORIZONS), "12. exact horizon set mismatch")
    ac.check(sorted(pred["configuration_id"].astype(str).unique().tolist()) == sorted(CONFIGS), "13. exact configuration set mismatch")
    ac.check(int(pred[["model", "horizon", "configuration_id"]].drop_duplicates().shape[0]) == 18, "14. fitted combinations are not exactly 18")
    ac.check(not pred["model"].astype(str).str.contains("ElasticNet", case=False).any(), "15. ElasticNet fitting detected")
    ac.check(not pred["model"].astype(str).str.contains("Persistence", case=False).any(), "16. Persistence fitting detected")
    ac.check(not pred["model"].astype(str).str.contains("BCR", case=False).any(), "17. BCR-TCN fitting detected")
    ac.check(not pred["model"].astype(str).str.contains("Hybrid", case=False).any(), "18. HybridRank fitting detected")
    ac.check(not pred["model"].astype(str).str.contains("blend", case=False).any(), "19. point-blend fitting detected")
    ac.check(
        fit_registry["hyperparameter_source_file"].astype(str).isin(
            [
                "revision_2026/04_controlled_reruns/ridge/ridge_selected_configurations.csv",
                "revision_2026/04_controlled_reruns/hgbr/hgbr_selected_configurations.csv",
            ]
        ).all(),
        "20. hyperparameter source provenance mismatch",
    )

    # 21-28
    ac.check(
        fit_registry["hyperparameter_source_file"].astype(str).str.contains("selected_configurations").all(),
        "21. frozen hyperparameter provenance is not selected configurations",
    )
    run_src = BUILD_SCRIPT.read_text(encoding="utf-8")
    ac.check("shift(3)" in run_src and "shift(5)" in run_src and "shift(7)" in run_src, "22. strictly prior TNout lag construction not found")
    ac.check("shift(1).rolling(7)" in run_src and "shift(1).rolling(14)" in run_src and "shift(1).rolling(30)" in run_src, "23. strictly prior TNout rolling construction not found")
    ac.check("tnout_roll_includes_day_t" in feat_reg.columns and (feat_reg["tnout_roll_includes_day_t"].astype(str).str.lower() == "false").all(), "24. day-t exclusion metadata failed")

    non_tn_variability_ok = True
    for (m, h), grp in feat_reg.groupby(["model", "horizon"], sort=True):
        if grp["non_tnout_features"].astype(str).nunique() != 1:
            non_tn_variability_ok = False
            break
    ac.check(non_tn_variability_ok, "25. fixed non-TN feature set rule failed")

    h5_hgbr = feat_reg[(feat_reg["model"] == "HGBR") & (feat_reg["horizon"] == 5)]
    ac.check(h5_hgbr["non_tnout_features"].astype(str).str.contains("Inflow_roll14").all(), "26. HGBR-H5 v2 non-TN lineage missing roll14 features")

    h13_hgbr = feat_reg[(feat_reg["model"] == "HGBR") & (feat_reg["horizon"].isin([1, 3]))]
    ac.check(~h13_hgbr["non_tnout_features"].astype(str).str.contains("Inflow_roll14").any(), "27. HGBR-H1/H3 lineage is not ordinary")

    ridge_all = feat_reg[feat_reg["model"] == "Ridge"]
    ac.check(~ridge_all["non_tnout_features"].astype(str).str.contains("Inflow_roll14").any(), "28. Ridge lineage is not ordinary")

    # 29-36
    split_keys = (
        split[split["outer_role"].astype(str) == "outer_test"][
            ["horizon", "outer_fold", "feature_date", "target_date"]
        ]
        .drop_duplicates()
        .copy()
    )
    pred_keys = pred[["horizon_int", "outer_fold", "feature_date", "target_date"]].drop_duplicates().rename(
        columns={"horizon_int": "horizon"}
    )
    merged_check = split_keys.merge(pred_keys, on=["horizon", "outer_fold", "feature_date", "target_date"], how="left", indicator=True)
    ac.check((merged_check["_merge"] == "both").all(), "29. canonical split usage mismatch")
    ac.check(split_embargo["outer_boundary_pass"].astype(bool).all(), "30. outer embargo preservation failed")
    ac.check(split_embargo["inner_boundary_pass"].astype(bool).all(), "31. inner-boundary preservation failed")

    h1_counts_ok = (
        pred[pred["horizon"] == "H1"].groupby(["model", "configuration_id", "outer_fold"]).size().eq(254).all()
    )
    h3_counts_ok = (
        pred[pred["horizon"] == "H3"].groupby(["model", "configuration_id", "outer_fold"]).size().eq(249).all()
    )
    h5_counts_ok = (
        pred[pred["horizon"] == "H5"].groupby(["model", "configuration_id", "outer_fold"]).size().eq(249).all()
    )
    ac.check(bool(h1_counts_ok), "32. H1 fold test counts mismatch")
    ac.check(bool(h3_counts_ok), "33. H3 fold test counts mismatch")
    ac.check(bool(h5_counts_ok), "34. H5 fold test counts mismatch")

    ac.check(bool(manifest.get("common_date_result", {}).get("pass", False)), "35. canonical test-date loss detected")

    common_dates_ok = True
    for (m, h, f), grp in pred.groupby(["model", "horizon", "outer_fold"], sort=True):
        sets = []
        for cfg in CONFIGS:
            d = grp[grp["configuration_id"] == cfg][["feature_date", "target_date"]]
            sets.append(set(tuple(r) for r in d.itertuples(index=False, name=None)))
        if not (sets[0] == sets[1] == sets[2]):
            common_dates_ok = False
            break
    ac.check(common_dates_ok, "36. common-date requirement across configurations failed")

    # 37-45
    ac.check(int(len(pred)) == 13536, "37. expected total prediction rows mismatch")
    by_h = pred.groupby("horizon").size().to_dict()
    ac.check(by_h == {"H1": 4572, "H3": 4482, "H5": 4482}, "38. prediction rows by horizon mismatch")

    dup = pred.duplicated(subset=["model", "horizon", "configuration_id", "outer_fold", "feature_date", "target_date"])
    ac.check((~dup).all(), "39. prediction keys are not unique")

    y_true_identical = True
    for (m, h, f, fd, td), grp in pred.groupby(["model", "horizon", "outer_fold", "feature_date", "target_date"], sort=False):
        if grp["y_true"].astype(float).nunique() != 1:
            y_true_identical = False
            break
    ac.check(y_true_identical, "40. y_true is not identical across configurations")

    ac.check(not pred[["y_true", "y_pred"]].isna().any().any(), "41. prediction completeness failed")
    ac.check(np.isfinite(pred[["y_true", "y_pred"]].to_numpy(dtype=float)).all(), "42. non-finite predictions detected")
    ac.check(ref_audit["keys_match"].astype(bool).all(), "43. reference reproduction key check failed")
    ac.check(ref_audit["y_true_match"].astype(bool).all(), "44. reference reproduction y_true check failed")
    ac.check((ref_audit["number_outside_tolerance"].astype(int) == 0).all(), "45. reference prediction tolerance check failed")

    # 46-54
    ac.check(retention["reconciliation_pass"].astype(bool).all(), "46. training-retention reconciliation failed")

    warmup_ok = True
    for model in FIT_MODELS:
        s = retention[(retention["model"] == model) & (retention["horizon"] == 1) & (retention["configuration_id"] == "LONG")]
        if sorted(s["warmup_losses_observed"].astype(int).tolist()) != [16, 16, 16]:
            warmup_ok = False
            break
        n = retention[(retention["model"] == model) & (retention["horizon"].isin([3, 5]))]
        if (n["warmup_losses_observed"].astype(int) != 0).any():
            warmup_ok = False
            break
    ac.check(warmup_ok, "47. warm-up loss accounting failed")

    feat_count_ok = True
    for r in feat_reg.itertuples(index=False):
        tn = [t.strip() for t in str(r.tnout_features).split(";") if t.strip()]
        non = [t.strip() for t in str(r.non_tnout_features).split(";") if t.strip()]
        if int(r.feature_count) != len(tn) + len(non):
            feat_count_ok = False
            break
    ac.check(feat_count_ok, "48. feature-count reconciliation failed")

    feat_name_ok = feat_reg["feature_names"].astype(str).str.len().gt(0).all()
    ac.check(feat_name_ok, "49. feature-name reconciliation failed")

    ac.check(int(len(point_fold)) == 54, "50. fold-level point metric row count mismatch")
    ac.check(int(len(point_pooled)) == 18, "51. pooled point metric row count mismatch")

    pooled_recompute_ok = True
    for (m, h, c), grp in pred.groupby(["model", "horizon", "configuration_id"], sort=True):
        p = point_pooled[(point_pooled["model"] == m) & (point_pooled["horizon"] == int(h.replace("H", ""))) & (point_pooled["configuration_id"] == c)]
        if p.empty:
            pooled_recompute_ok = False
            break
        prow = p.iloc[0]
        mae = float(grp["abs_error"].mean())
        mse = float(grp["squared_error"].mean())
        rmse = float(np.sqrt(mse))
        if not (abs(mae - float(prow["MAE"])) <= 1e-12 and abs(mse - float(prow["MSE"])) <= 1e-12 and abs(rmse - float(prow["RMSE"])) <= 1e-12):
            pooled_recompute_ok = False
            break
    ac.check(pooled_recompute_ok, "52. pooled metric recomputation mismatch")

    ac.check((point_fold["mase_denominator_source"].astype(str) == "outer_train_y_true_abs_first_difference_mean_m1").all(), "53. MASE denominator convention mismatch")
    ac.check((point_pooled["mase_method"].astype(str) == "mean(abs_error/(row_fold_denominator+epsilon))").all(), "54. pooled MASE row convention mismatch")

    # 55-63
    ac.check(int(len(stab_fold)) == 54 and int(len(stab_pooled)) == 18, "55. stability row count mismatch")

    self_rows_fold = stab_fold[stab_fold["configuration_id"] == stab_fold["final_reference_configuration"]]
    self_rows_pool = stab_pooled[stab_pooled["configuration_id"] == stab_pooled["final_reference_configuration"]]
    ac.check(np.allclose(self_rows_fold["pearson_prediction_correlation"].to_numpy(dtype=float), 1.0, atol=1e-12), "56. reference self Pearson != 1")
    ac.check(np.allclose(self_rows_fold["spearman_prediction_rank_correlation"].to_numpy(dtype=float), 1.0, atol=1e-12), "56b. reference self Spearman != 1")

    ac.check((self_rows_fold["maximum_absolute_prediction_difference"].astype(float).abs() <= 1e-12).all(), "57. reference zero-difference max failed")
    ac.check((self_rows_fold["mean_absolute_prediction_difference"].astype(float).abs() <= 1e-12).all(), "57b. reference zero-difference mean failed")

    ac.check((recall_fold["tau"].astype(float) == 16.0).all(), "58. event threshold definition mismatch")
    k_ok = (recall_fold["k"].astype(int) == np.ceil(recall_fold["budget_fraction"].astype(float) * recall_fold["N"].astype(float)).astype(int)).all()
    ac.check(bool(k_ok), "59. fold-local k calculation mismatch")
    ac.check(
        (recall_fold["tie_break_rule"].astype(str) == "y_pred_desc_then_feature_date_asc_then_target_date_asc_then_canonical_test_row_asc").all(),
        "60. deterministic top-k tie-breaking rule mismatch",
    )

    pooled_topk_ok = True
    for (m, h, c), grp in recall_fold.groupby(["model", "horizon", "configuration_id"], sort=True):
        p = recall_pooled[(recall_pooled["model"] == m) & (recall_pooled["horizon"] == int(h)) & (recall_pooled["configuration_id"] == c)]
        if p.empty:
            pooled_topk_ok = False
            break
        prow = p.iloc[0]
        if int(prow["N"]) != int(grp["N"].sum()) or int(prow["k"]) != int(grp["k"].sum()):
            pooled_topk_ok = False
            break
    ac.check(pooled_topk_ok, "61. pooled top-k aggregation mismatch")

    recall_recompute_ok = True
    for r in recall_pooled.itertuples(index=False):
        ev = int(r.event_count)
        tp = int(r.true_positives)
        k = int(r.k)
        rec = float(tp / ev) if ev > 0 else float("nan")
        prec = float(tp / k) if k > 0 else float("nan")
        if (not (math.isnan(rec) and math.isnan(float(r.recall)))) and abs(rec - float(r.recall)) > 1e-12:
            recall_recompute_ok = False
            break
        if (not (math.isnan(prec) and math.isnan(float(r.precision)))) and abs(prec - float(r.precision)) > 1e-12:
            recall_recompute_ok = False
            break
    ac.check(recall_recompute_ok, "62. Recall@5% recomputation mismatch")
    ac.check((recall_fold["retrospective_only"].astype(str).str.lower() == "true").all(), "63. retrospective-only labels failed")

    # 64-70+
    bad_terms = [
        "best configuration",
        "optimal lag window",
        "globally superior",
        "selected as best",
        "rank configurations",
    ]
    lowered = (out["summary_md"] + "\n" + out["report"]).lower()
    no_bad_language = True
    for t in bad_terms:
        if t in lowered:
            no_bad_language = False
            break
    ac.check(no_bad_language, "64. optimization language detected in structured outputs")

    map_ok = manifest.get("final_reference_mapping", {}) == FINAL_REF_MAP
    ac.check(map_ok, "65. final reference mapping changed post-hoc")

    src_hash_ok = True
    for r in out["source_registry"].itertuples(index=False):
        p = ROOT / str(r.input_path)
        if (not p.exists()) or (sha256_file(p) != str(r.sha256)):
            src_hash_ok = False
            break
    ac.check(src_hash_ok, "66. input source hash registry mismatch")

    software = out["software_env"]
    req_keys = {"python_version", "numpy_version", "pandas_version", "scikit_learn_version", "scipy_version"}
    ac.check(req_keys.issubset(set(software.keys())), "67. software environment registry incomplete")

    ac.check(deterministic_pass, "68. deterministic regeneration failed")

    rewrite_checksums()
    cov = checksum_coverage_ok()
    sha_cmd = run_sha256_check()
    checksum_pass = bool(cov["registry_pass"] and cov["coverage_pass"] and sha_cmd["exit_code"] == 0)
    ac.check(checksum_pass, "69. checksum registry coverage or sha256sum -c failed")

    ac.check(source_preservation_pass, "70. final source checksum preservation failed")

    # Extra cross-checks.
    ac.check(int(len(recall_fold)) == 54 and int(len(recall_pooled)) == 18, "71. Recall@5% row counts mismatch")
    ac.check((pred["reference_reproduction_status"].astype(str) != "pending").all(), "72. pending reference status remains in prediction rows")

    test_result = "pass_{}_of_{}".format(ac.total - ac.failed, ac.total) if ac.failed == 0 else "fail_{}_of_{}".format(ac.total - ac.failed, ac.total)
    deterministic_result = "pass" if deterministic_pass else "fail"

    # Final decision policy.
    manifest = out["manifest"]
    manifest["source_preservation_result"] = {
        "pass": bool(source_preservation_pass),
        "files_before": int(len(pre_snapshot)),
        "files_after": int(len(post_snapshot)),
        "changed_paths": sorted(
            [p for p in set(pre_snapshot.keys()).union(set(post_snapshot.keys())) if pre_snapshot.get(p) != post_snapshot.get(p)]
        ),
    }

    manifest["deterministic_details"] = {
        "first_build_hash": first_hash,
        "second_build_hash": second_hash,
        "files_compared": sorted(list(det_files)),
        "differences_found": differences,
    }

    manifest["checksum_result"] = {
        "registry_pass": bool(cov["registry_pass"]),
        "coverage_pass": bool(cov["coverage_pass"]),
        "coverage_missing": cov["coverage_missing"],
        "coverage_extra": cov["coverage_extra"],
        "sha256sum_exit_code": int(sha_cmd["exit_code"]),
    }

    manifest["assertions_total"] = int(ac.total)
    manifest["assertions_passed"] = int(ac.total - ac.failed)
    manifest["assertions_failed"] = int(ac.failed)

    if (ac.failed > 0) or (not deterministic_pass) or (not checksum_pass) or (not source_preservation_pass):
        final_decision = "D"
    elif (
        not bool(manifest.get("common_date_result", {}).get("pass", False))
        or not bool(manifest.get("embargo_result", {}).get("pass", False))
        or not bool(manifest.get("training_retention_result", {}).get("pass", False))
        or bool(manifest.get("reference_reproduction_result", {}).get("material_failure", False))
    ):
        final_decision = "C"
    elif not bool(manifest.get("reference_reproduction_result", {}).get("overall_pass", False)):
        final_decision = "B"
    else:
        final_decision = "A"

    write_manifest_and_report(manifest, final_decision, test_result, deterministic_result)

    # Recompute checksums after manifest/report update.
    rewrite_checksums()
    final_sha = run_sha256_check()
    if final_sha["exit_code"] != 0:
        raise RuntimeError("final sha256sum -c failed:\n{}\n{}".format(final_sha["stdout"], final_sha["stderr"]))

    if ac.failed:
        details = "\n".join(ac.failures)
        raise AssertionError(
            "Lag-window execution test failed ({} failed of {}):\n{}".format(ac.failed, ac.total, details)
        )

    print("Ran {} checks; failures=0".format(ac.total))


if __name__ == "__main__":
    main()
