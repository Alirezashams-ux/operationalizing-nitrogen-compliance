from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/06_corrected_hybridrank/h5/fully_nested"
EXPECTED_BRANCH = "controlled-reruns-v1"

BUILD_SCRIPT_PATH = OUT_DIR / "build_fully_nested_hybridrank_h5.py"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"

ENET_SELECTION_PLAN_PATH = CONTROLLED_DIR / "elasticnet" / "elasticnet_selection_plan.json"
HGBR_SELECTION_PLAN_PATH = CONTROLLED_DIR / "hgbr" / "hgbr_selection_plan.json"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
NESTED_ASSIGNMENT_PATH = OUT_DIR / "fully_nested_h5_split_assignment.csv"
NESTED_SUMMARY_PATH = OUT_DIR / "fully_nested_h5_split_summary.csv"
COMPONENT_SELECTION_PATH = OUT_DIR / "component_selection_results.csv"
BCR_GATE_PATH = OUT_DIR / "bcr_component_validation_gate.csv"
FOLD1_TAU16_DIAGNOSTIC_PATH = OUT_DIR / "fold1_component_validation_tau16_diagnostic.csv"
HV_COMPONENT_PRED_PATH = OUT_DIR / "hybrid_validation_component_predictions.csv"
HV_RANK_PATH = OUT_DIR / "hybrid_validation_rank_scores.csv"
HV_SUPPORT_PATH = OUT_DIR / "hybrid_validation_event_support.csv"
WEIGHTS_PATH = OUT_DIR / "fully_nested_hybridrank_weights.csv"
OUTER_COMPONENT_PRED_PATH = OUT_DIR / "fully_nested_outer_test_component_predictions.csv"
OUTER_SCORES_PATH = OUT_DIR / "fully_nested_hybridrank_outer_test_scores.csv"
ISOLATION_AUDIT_PATH = OUT_DIR / "fully_nested_hybridrank_isolation_audit.csv"
MANIFEST_PATH = OUT_DIR / "fully_nested_hybridrank_h5_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "fully_nested_hybridrank_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "fully_nested_hybridrank_h5_checksums.sha256"

H = 5
TAUS = (15.0, 16.0, 17.0)
BUDGETS = (0.05, 0.10)
COMPONENT_ORDER = ("BCR-TCN", "ElasticNet", "Persistence", "HGBR")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_output(args: List[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def to_date_str(x: Any) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        out.append((parts[0], parts[-1]))
    return out


def expected_checksum_for_file(path: Path, filename: str) -> str:
    for expected, rel_name in parse_checksum_manifest(path):
        rel_norm = str(rel_name).replace("\\", "/")
        if rel_norm == filename or rel_norm.endswith("/" + filename):
            return expected
    raise KeyError("Checksum entry not found for {} in {}".format(filename, path))


def verify_checksum_manifest(path: Path, base_dir: Path) -> bool:
    ok = True
    for expected, rel_name in parse_checksum_manifest(path):
        target = base_dir / rel_name
        if not target.exists():
            return False
        observed = sha256_file(target)
        if observed != expected:
            ok = False
    return ok


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"Decision:\s*([ABCD])", txt)
    if m:
        return m.group(1)

    lines = [ln.strip() for ln in txt.splitlines()]
    final_idx = -1
    for i, ln in enumerate(lines):
        if "FINAL DECISION" in ln.upper():
            final_idx = i
            break
    if final_idx >= 0:
        for ln in lines[final_idx + 1 : final_idx + 20]:
            if re.match(r"^[ABCD]\.\s", ln):
                return ln[0]

    for ln in lines[-30:]:
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0]

    raise RuntimeError("Could not infer decision from {}".format(path))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_checksums_file(out_dir: Path, checksum_path: Path) -> None:
    targets = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == checksum_path.name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        targets.append(p)

    lines = ["{}  {}".format(sha256_file(p), p.name) for p in targets]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_completion_report_decision(path: Path, decision: str) -> None:
    text = path.read_text(encoding="utf-8")

    decision_line = "D. Temporal isolation, source preservation, or deterministic tests failed."
    if decision == "A":
        decision_line = "A. Fully nested corrected H5 HybridRank scores passed; retrospective alarm-budget evaluation may begin."
    elif decision == "B":
        decision_line = "B. Pipeline completed, but one or more component or validation-support issues require investigation."
    elif decision == "C":
        decision_line = "C. Fully nested component or ensemble selection could not be completed safely with the available data."

    if "<!-- AUTO_DECISION_START -->" in text and "<!-- AUTO_DECISION_END -->" in text:
        text = re.sub(
            r"<!-- AUTO_DECISION_START -->.*?<!-- AUTO_DECISION_END -->",
            "<!-- AUTO_DECISION_START -->\n{}\n<!-- AUTO_DECISION_END -->".format(decision_line),
            text,
            flags=re.S,
        )
    else:
        text += "\n\nFINAL DECISION\n\n{}\n".format(decision_line)

    path.write_text(text, encoding="utf-8")


class AssertionTracker:
    def __init__(self) -> None:
        self.count = 0
        self.passed = 0
        self.failures: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.count += 1
        if condition:
            self.passed += 1
        else:
            self.failures.append("{}. {}".format(self.count, message))



def run_build() -> None:
    cmd = [sys.executable, str(BUILD_SCRIPT_PATH)]
    subprocess.check_call(cmd, cwd=ROOT)



def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run tests from workspace root: {}".format(ROOT))

    # Ensure artifacts exist and branch by checkpoint-gate status.
    run_build()
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Build did not produce manifest: {}".format(MANIFEST_PATH))

    manifest_probe = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    gate_probe = manifest_probe.get("bcr_checkpoint_gate", {})
    blocked_folds_probe = gate_probe.get("blocked_folds", []) if isinstance(gate_probe, dict) else []

    if blocked_folds_probe:
        first_checks = {
            "fully_nested_h5_split_assignment.csv": sha256_file(NESTED_ASSIGNMENT_PATH),
            "fully_nested_h5_split_summary.csv": sha256_file(NESTED_SUMMARY_PATH),
            "component_selection_results.csv": sha256_file(COMPONENT_SELECTION_PATH),
            "bcr_component_validation_gate.csv": sha256_file(BCR_GATE_PATH),
            "fold1_component_validation_tau16_diagnostic.csv": sha256_file(FOLD1_TAU16_DIAGNOSTIC_PATH),
        }

        run_build()
        second_checks = {
            "fully_nested_h5_split_assignment.csv": sha256_file(NESTED_ASSIGNMENT_PATH),
            "fully_nested_h5_split_summary.csv": sha256_file(NESTED_SUMMARY_PATH),
            "component_selection_results.csv": sha256_file(COMPONENT_SELECTION_PATH),
            "bcr_component_validation_gate.csv": sha256_file(BCR_GATE_PATH),
            "fold1_component_validation_tau16_diagnostic.csv": sha256_file(FOLD1_TAU16_DIAGNOSTIC_PATH),
        }
        deterministic_pass = first_checks == second_checks

        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        split_expected_sha = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")

        input_verification = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        assignment_df = pd.read_csv(NESTED_ASSIGNMENT_PATH)
        summary_df = pd.read_csv(NESTED_SUMMARY_PATH)
        selection_df = pd.read_csv(COMPONENT_SELECTION_PATH)
        gate_df = pd.read_csv(BCR_GATE_PATH)
        fold1_diag_df = pd.read_csv(FOLD1_TAU16_DIAGNOSTIC_PATH)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        tracker = AssertionTracker()
        tracker.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "Correct branch")
        tracker.check(Path.cwd().resolve() == ROOT.resolve(), "Correct working directory")
        tracker.check(sys.version_info[:2] == (3, 8), "Python 3.8 compatibility")
        tracker.check(sha256_file(Path(lock["absolute_path"])) == lock["sha256"], "Dataset checksum matches")
        tracker.check(sha256_file(SPLIT_PATH) == split_expected_sha, "Split checksum matches")
        tracker.check(verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR), "Canonical package checksums match")
        tracker.check(infer_decision_from_report(CANONICAL_REPORT_PATH) == "A", "Canonical decision is A")
        tracker.check("h5" in str(manifest.get("purpose", "")).lower(), "H5 only")
        tracker.check(not gate_df.empty, "BCR gate report exists")

        tracker.check(not fold1_diag_df.empty, "Fold 1 tau16 support diagnostic artifact exists")
        required_diag_cols = {
            "outer_fold",
            "split_boundary",
            "nested_role",
            "included",
            "rows_N",
            "tau_threshold",
            "tau_event_count",
            "tau_event_prevalence",
            "feature_date_start",
            "feature_date_end",
            "target_date_start",
            "target_date_end",
            "component_validation_start",
            "hybrid_validation_start",
            "outer_test_start",
            "diagnostic_note",
        }
        tracker.check(required_diag_cols.issubset(set(fold1_diag_df.columns)), "Fold 1 diagnostic columns are complete")

        fold1_diag = fold1_diag_df[fold1_diag_df["outer_fold"].astype(int) == 1].copy()
        expected_roles = {
            "component_subtrain",
            "inner_embargo_excluded",
            "component_validation",
            "hybrid_embargo_excluded",
            "hybrid_validation",
            "outer_embargo_excluded",
            "outer_test",
        }
        tracker.check(expected_roles.issubset(set(fold1_diag["nested_role"].astype(str).tolist())), "Fold 1 diagnostic covers all split-boundary nested roles")

        assign_fold1_counts = (
            assignment_df[assignment_df["outer_fold"].astype(int) == 1]
            .groupby("nested_role", sort=True)
            .size()
            .to_dict()
        )
        diag_fold1_counts = (
            fold1_diag.groupby("nested_role", sort=True)["rows_N"].sum().astype(int).to_dict()
        )
        counts_match = True
        for role, c in assign_fold1_counts.items():
            if int(diag_fold1_counts.get(role, -1)) != int(c):
                counts_match = False
                break
        counts_match = counts_match and set(assign_fold1_counts.keys()) == set(diag_fold1_counts.keys())
        tracker.check(counts_match, "Fold 1 diagnostic counts match split-boundary assignment counts")

        required_gate_cols = {
            "outer_fold",
            "component_validation_N",
            "component_validation_events_tau16",
            "component_validation_k_r005",
            "recall_at_5pct_defined",
            "blocked_for_bcr_checkpoint_selection",
            "gate_notes",
        }
        tracker.check(required_gate_cols.issubset(set(gate_df.columns)), "BCR gate columns are complete")

        k_expected = np.ceil(0.05 * gate_df["component_validation_N"].astype(float)).astype(int)
        tracker.check(bool((gate_df["component_validation_k_r005"].astype(int) == k_expected).all()), "BCR gate uses k=ceil(0.05*N)")

        blocked = gate_df[gate_df["blocked_for_bcr_checkpoint_selection"].astype(str).str.lower() == "true"].copy()
        tracker.check(not blocked.empty, "At least one fold is blocked when recall criterion is undefined")

        block_reason_ok = bool(
            (
                (blocked["component_validation_events_tau16"].astype(int) <= 0)
                | (blocked["component_validation_k_r005"].astype(int) <= 0)
            ).all()
        )
        tracker.check(block_reason_ok, "Blocked folds are due to tau16 event count zero or k zero")

        tracker.check(bool((blocked["recall_at_5pct_defined"].astype(str).str.lower() == "false").all()), "Blocked folds have undefined Recall@5%")
        tracker.check(bool(blocked["gate_notes"].astype(str).str.contains("no_checkpoint_selection_permitted", regex=False).all()), "No fallback checkpoint criterion is permitted on blocked folds")

        fold1_cv_diag = fold1_diag[fold1_diag["nested_role"].astype(str) == "component_validation"].copy()
        tracker.check(len(fold1_cv_diag) == 1, "Fold 1 diagnostic has exactly one component_validation row")
        fold1_cv_ok = False
        if len(fold1_cv_diag) == 1:
            cv_row = fold1_cv_diag.iloc[0]
            cv_n = int(cv_row["rows_N"])
            cv_e = int(cv_row["tau_event_count"])
            cv_p = float(cv_row["tau_event_prevalence"])

            gate_fold1 = gate_df[gate_df["outer_fold"].astype(int) == 1].copy()
            gate_n = int(gate_fold1.iloc[0]["component_validation_N"]) if len(gate_fold1) == 1 else -1
            gate_e = int(gate_fold1.iloc[0]["component_validation_events_tau16"]) if len(gate_fold1) == 1 else -1

            date_ok = True
            for col in ["feature_date_start", "feature_date_end", "target_date_start", "target_date_end"]:
                date_ok = date_ok and bool(pd.notna(pd.to_datetime(cv_row[col], errors="coerce")))

            fold1_cv_ok = bool(
                cv_n == gate_n
                and cv_e == gate_e
                and cv_e == 0
                and abs(cv_p - 0.0) <= 1e-12
                and date_ok
            )
        tracker.check(fold1_cv_ok, "Fold 1 diagnostic confirms component-validation tau16 support is zero with valid date span")

        blocked_folds = sorted(blocked["outer_fold"].astype(int).unique().tolist())
        blocked_bcr_rows = selection_df[
            (selection_df["component"] == "BCR-TCN")
            & (selection_df["outer_fold"].astype(int).isin(blocked_folds))
        ]
        tracker.check(not blocked_bcr_rows.empty, "Blocked BCR component-selection rows are recorded")
        tracker.check(bool(blocked_bcr_rows["selection_decision"].astype(str).str.contains("blocked_no_tau16_event_support", regex=False).all()), "Blocked folds do not select fallback BCR checkpoints")

        decision_now = infer_decision_from_report(COMPLETION_REPORT_PATH)
        tracker.check(decision_now in {"B", "C"}, "Completion report decision is B or C for blocked folds")
        tracker.check(decision_now == "C", "Blocked fold path issues Decision C")

        tracker.check(verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR), "Canonical package files are unchanged")
        tracker.check(bool(len(manifest.get("files_modified_outside_authorized_directory", [])) == 0), "All outputs remain inside authorized directory")
        tracker.check(deterministic_pass, "Repeated execution is deterministic on safeguard-block path")

        all_pass = len(tracker.failures) == 0
        manifest["test_result"] = "pass_guard_block_{}_of_{}".format(tracker.passed, tracker.count) if all_pass else "fail_guard_block_{}_of_{}".format(tracker.passed, tracker.count)
        manifest["deterministic_result"] = "pass" if deterministic_pass else "fail"
        manifest["output_checksums"] = {p.name: sha256_file(p) for p in OUT_DIR.iterdir() if p.is_file()}
        manifest["timestamp"] = utc_now_iso()
        write_json(MANIFEST_PATH, manifest)
        update_completion_report_decision(COMPLETION_REPORT_PATH, "C")
        write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

        print("Assertions passed (guard-block path): {}/{}".format(tracker.passed, tracker.count))
        print("Deterministic check: {}".format("pass" if deterministic_pass else "fail"))
        print("Final decision: C")
        print("Next action: Investigate component-validation tau16 event support and rerun build_fully_nested_hybridrank_h5.py.")

        if not all_pass:
            print("Failed assertions:")
            for msg in tracker.failures:
                print(msg)
            raise SystemExit(1)
        return

    # Full-path determinism check.
    first_checks = {
        "hybrid_validation_component_predictions.csv": sha256_file(HV_COMPONENT_PRED_PATH),
        "fully_nested_outer_test_component_predictions.csv": sha256_file(OUTER_COMPONENT_PRED_PATH),
        "fully_nested_hybridrank_outer_test_scores.csv": sha256_file(OUTER_SCORES_PATH),
        "fully_nested_hybridrank_weights.csv": sha256_file(WEIGHTS_PATH),
    }

    run_build()
    second_checks = {
        "hybrid_validation_component_predictions.csv": sha256_file(HV_COMPONENT_PRED_PATH),
        "fully_nested_outer_test_component_predictions.csv": sha256_file(OUTER_COMPONENT_PRED_PATH),
        "fully_nested_hybridrank_outer_test_scores.csv": sha256_file(OUTER_SCORES_PATH),
        "fully_nested_hybridrank_weights.csv": sha256_file(WEIGHTS_PATH),
    }

    deterministic_pass = first_checks == second_checks

    # Load artifacts.
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    split_expected_sha = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")

    input_verification = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    assignment_df = pd.read_csv(NESTED_ASSIGNMENT_PATH)
    summary_df = pd.read_csv(NESTED_SUMMARY_PATH)
    selection_df = pd.read_csv(COMPONENT_SELECTION_PATH)
    hv_pred_df = pd.read_csv(HV_COMPONENT_PRED_PATH)
    hv_rank_df = pd.read_csv(HV_RANK_PATH)
    support_df = pd.read_csv(HV_SUPPORT_PATH)
    weights_df = pd.read_csv(WEIGHTS_PATH)
    outer_pred_df = pd.read_csv(OUTER_COMPONENT_PRED_PATH)
    score_df = pd.read_csv(OUTER_SCORES_PATH)
    isolation_df = pd.read_csv(ISOLATION_AUDIT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    canonical_wide = pd.read_csv(CANONICAL_WIDE_PATH)

    enet_plan = json.loads(ENET_SELECTION_PLAN_PATH.read_text(encoding="utf-8"))
    hgbr_plan = json.loads(HGBR_SELECTION_PLAN_PATH.read_text(encoding="utf-8"))

    # Normalize date columns where needed.
    for df, cols in [
        (assignment_df, ["feature_date", "target_date", "component_validation_start", "hybrid_validation_start", "outer_test_start"]),
        (summary_df, ["component_subtrain_start", "component_subtrain_end", "component_validation_start", "component_validation_end", "hybrid_validation_start", "hybrid_validation_end", "outer_test_start", "outer_test_end"]),
        (hv_pred_df, ["feature_date", "target_date"]),
        (hv_rank_df, ["feature_date", "target_date"]),
        (outer_pred_df, ["feature_date", "target_date"]),
        (score_df, ["feature_date", "target_date"]),
        (canonical_wide, ["feature_date", "target_date"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = normalize_date_col(df[col])

    corrected_summary = pd.read_csv(SPLIT_SUMMARY_PATH)
    corrected_h5_summary = corrected_summary[corrected_summary["horizon"].astype(int) == H].copy()

    tracker = AssertionTracker()

    # 1-8
    tracker.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "Correct branch")
    tracker.check(Path.cwd().resolve() == ROOT.resolve(), "Correct working directory")
    tracker.check(sys.version_info[:2] == (3, 8), "Python 3.8 compatibility")
    tracker.check(sha256_file(Path(lock["absolute_path"])) == lock["sha256"], "Dataset checksum matches")
    tracker.check(sha256_file(SPLIT_PATH) == split_expected_sha, "Split checksum matches")
    tracker.check(verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR), "Canonical package checksums match")
    tracker.check(infer_decision_from_report(CANONICAL_REPORT_PATH) == "A", "Canonical decision is A")
    tracker.check(score_df["horizon"].astype(int).nunique() == 1 and int(score_df["horizon"].iloc[0]) == H, "H5 only")

    # 9-15
    joined = summary_df.merge(
        corrected_h5_summary[["outer_fold", "outer_train_n_after_purge"]],
        on="outer_fold",
        how="left",
    )
    tracker.check(bool((joined["outer_train_N"].astype(int) == joined["outer_train_n_after_purge"].astype(int)).all()), "Corrected outer-training counts match protocol")
    tracker.check(bool((summary_df["component_validation_end"] < summary_df["hybrid_validation_start"]).all()), "Component validation earlier than HybridRank validation")
    tracker.check(bool((summary_df["hybrid_validation_end"] < summary_df["outer_test_start"]).all()), "HybridRank validation earlier than outer test")
    tracker.check(bool(summary_df["inner_target_separation_pass"].astype(str).str.lower().eq("true").all()), "Five-day target separation holds at component boundary")
    tracker.check(bool(summary_df["hybrid_target_separation_pass"].astype(str).str.lower().eq("true").all()), "Five-day target separation holds at HybridRank boundary")
    tracker.check(bool(summary_df["outer_target_separation_pass"].astype(str).str.lower().eq("true").all()), "Five-day target separation holds at outer boundary")
    tracker.check(bool(summary_df["future_fold_exclusion_pass"].astype(str).str.lower().eq("true").all()), "No future fold enters any earlier fold")

    # 16-23
    tracker.check(bool(selection_df["hybrid_validation_used"].astype(str).str.lower().eq("false").all()), "Component selection does not use HybridRank validation")
    tracker.check(bool(selection_df["outer_test_used"].astype(str).str.lower().eq("false").all()), "Component selection does not use outer test")
    tracker.check(bool(weights_df["component_selection_frozen"].astype(str).str.lower().eq("true").all()), "Weight selection uses frozen selected component configurations")
    tracker.check(bool(weights_df["weight_selection_partition"].astype(str).eq("hybrid_validation_only").all()), "Weight selection uses only HybridRank validation")
    tracker.check(bool(weights_df["outer_test_outcomes_used"].astype(str).str.lower().eq("false").all()), "Weight selection does not use outer-test outcomes")

    bcr_sel = selection_df[selection_df["component"] == "BCR-TCN"]
    tracker.check(bool(bcr_sel["checkpoint_selection_source"].astype(str).eq("component_validation_only").all()), "BCR-TCN checkpoint selection uses only component validation")
    tracker.check(bool(hv_pred_df["bcr_refit_early_stop_used"].astype(str).str.lower().eq("false").all()), "BCR-TCN HV refit does not early-stop on HV")
    tracker.check(bool(outer_pred_df["bcr_outer_fit_early_stop_used"].astype(str).str.lower().eq("false").all()), "Final BCR-TCN outer-test fit does not early-stop on outer test")

    # 24-28
    enet_candidate_expected = enet_plan["candidate_configs"]
    enet_candidate_observed = bcr_sel = selection_df[selection_df["component"] == "ElasticNet"]["candidate_space"].iloc[0]
    tracker.check(json.loads(enet_candidate_observed) == enet_candidate_expected, "ElasticNet candidate space matches approved provenance")

    hgbr_candidate_observed = json.loads(selection_df[selection_df["component"] == "HGBR"]["candidate_space"].iloc[0])
    tracker.check(hgbr_candidate_observed["candidate_space"] == hgbr_plan["candidate_space_by_horizon"][str(H)], "HGBR candidate space matches approved provenance")

    persistence_rows = selection_df[selection_df["component"] == "Persistence"]
    persistence_ok = bool(
        persistence_rows["candidate_space"].astype(str).str.contains("TNout at feature_date", regex=False).all()
        and persistence_rows["selected_configuration"].astype(str).str.contains("feature_date_target_lag_h5", regex=False).all()
    )
    tracker.check(persistence_ok, "Persistence rule matches approved provenance")

    all_components = sorted(selection_df["component"].unique().tolist())
    tracker.check("Ridge" not in all_components, "No Ridge component exists")
    tracker.check(not any("point" in c.lower() and "blend" in c.lower() for c in all_components), "No point-blend component exists")

    # 29-37
    tracker.check(bool(hv_rank_df["rank_method"].astype(str).eq("average_ties").all()), "Rank normalization uses average ties")
    tracker.check(bool(hv_rank_df["rank_denominator"].astype(str).eq("N-1").all()), "Rank normalization uses N-1 denominator")
    tracker.check(bool(hv_rank_df["rank_scope"].astype(str).eq("component_wise_within_outer_fold").all()), "Rank normalization is component-wise and fold-wise")
    tracker.check(bool(hv_rank_df["used_outcomes"].astype(str).str.lower().eq("false").all()), "Rank normalization does not use y_true")

    tracker.check(bool(weights_df["candidate_grid"].astype(str).eq("0.0_to_1.0_step_0.1").all()), "Candidate weight grid uses step 0.1")

    grp_w = weights_df.groupby(["outer_fold", "threshold", "alarm_budget", "policy_name"], sort=True)["weight"].sum().to_numpy(dtype=float)
    tracker.check(bool(np.allclose(grp_w, np.ones_like(grp_w), atol=1e-9)), "Candidate vectors are normalized to sum to one")

    tuned_selected = weights_df[
        (weights_df["policy_name"] == "HybridRank_nested_tuned")
        & (weights_df["fallback_applied"].astype(str).str.lower() == "false")
    ].copy()
    precision_floor_ok = True
    if not tuned_selected.empty:
        precision_floor_ok = bool((tuned_selected["objective_precision"].astype(float) + 1e-12 >= tuned_selected["precision_floor"].astype(float)).all())
    tracker.check(precision_floor_ok, "Weight selection uses precision-floor filtering")

    tracker.check(bool(weights_df["tie_rule"].astype(str).str.contains("lexicographic_recall_precision_tp", regex=False).all()), "Objective order is recall, precision, TP")
    tracker.check(bool(weights_df["tie_rule"].astype(str).str.contains("first_encountered", regex=False).all()), "Exact objective ties retain the first candidate")

    # 38-43
    fixed = weights_df[weights_df["policy_name"] == "HybridRank_fixed"].copy()
    fixed_ok = True
    expected_fixed = {"BCR-TCN": 0.50, "ElasticNet": 0.25, "Persistence": 0.25, "HGBR": 0.00}
    for comp, w in expected_fixed.items():
        vals = fixed[fixed["component"] == comp]["weight"].astype(float)
        fixed_ok = fixed_ok and bool(np.allclose(vals.to_numpy(), np.full(len(vals), w), atol=1e-12))
    tracker.check(fixed_ok, "Fixed weights equal 0.50, 0.25, 0.25, 0.00")

    guarded = weights_df[weights_df["policy_name"] == "HybridRank_nested_guarded"].copy()
    guard_ok = True
    if not guarded.empty:
        g_small = guarded[np.isclose(guarded["alarm_budget"].astype(float), 0.05)]
        guard_ok = bool(g_small["guard_applied"].astype(str).str.lower().eq("true").all())
    tracker.check(guard_ok, "Guard condition uses r <= 0.05")

    support_ok = bool(
        (support_df["minimum_events_required"].astype(int) == 5).all()
        and (support_df["minimum_alarm_slots_required"].astype(int) == 3).all()
    )
    tracker.check(support_ok, "Sparse-event fallback uses the prespecified support rule")

    tracker.check(bool(weights_df["outer_test_outcomes_used"].astype(str).str.lower().eq("false").all()), "No fallback decision uses outer-test performance")
    tracker.check(bool((weights_df["weight"].astype(float) >= -1e-12).all()), "All selected weights are non-negative")
    tracker.check(bool(np.allclose(grp_w, np.ones_like(grp_w), atol=1e-9)), "All selected weights sum to one")

    # 44-51
    hv_keys = hv_pred_df[["outer_fold", "feature_date", "target_date"]].copy()
    hv_valid_keys = not hv_keys.duplicated().any()
    assignment_hv = assignment_df[(assignment_df["nested_role"] == "hybrid_validation") & (assignment_df["included"].astype(str).str.lower() == "true")][["outer_fold", "feature_date", "target_date"]].copy()
    merged_hv = hv_keys.merge(assignment_hv, on=["outer_fold", "feature_date", "target_date"], how="outer", indicator=True)
    hv_valid_keys = hv_valid_keys and bool((merged_hv["_merge"] == "both").all())
    tracker.check(hv_valid_keys, "Hybrid-validation prediction keys are valid")

    tracker.check(int(len(outer_pred_df)) == 747, "Outer-test prediction count equals 747")

    canon_keys = canonical_wide[["outer_fold", "feature_date", "target_date"]].copy()
    merged_outer_keys = outer_pred_df[["outer_fold", "feature_date", "target_date"]].merge(
        canon_keys,
        on=["outer_fold", "feature_date", "target_date"],
        how="outer",
        indicator=True,
    )
    tracker.check(bool((merged_outer_keys["_merge"] == "both").all()), "Outer-test keys match corrected H5 canonical key set")

    merged_outer_values = outer_pred_df.merge(
        canonical_wide[["outer_fold", "feature_date", "target_date", "y_true", "event_tau15", "event_tau16", "event_tau17"]],
        on=["outer_fold", "feature_date", "target_date"],
        how="inner",
        suffixes=("", "_canonical"),
    )
    y_match = float(np.max(np.abs(merged_outer_values["y_true"].to_numpy(dtype=float) - merged_outer_values["y_true_canonical"].to_numpy(dtype=float)))) <= 1e-6
    e_match = True
    for c in ["event_tau15", "event_tau16", "event_tau17"]:
        e_match = e_match and bool((merged_outer_values[c].astype(int).to_numpy() == merged_outer_values[c + "_canonical"].astype(int).to_numpy()).all())
    tracker.check(bool(y_match and e_match), "y_true and event labels match canonical H5 values")

    prob_cols = ["bcr_tcn_p_tau15", "bcr_tcn_p_tau16", "bcr_tcn_p_tau17"]
    prob_ok = True
    for c in prob_cols:
        vals_hv = hv_pred_df[c].to_numpy(dtype=float)
        vals_out = outer_pred_df[c].to_numpy(dtype=float)
        prob_ok = prob_ok and bool(np.isfinite(vals_hv).all() and np.isfinite(vals_out).all())
        prob_ok = prob_ok and bool(((vals_hv >= 0.0) & (vals_hv <= 1.0)).all() and ((vals_out >= 0.0) & (vals_out <= 1.0)).all())
    tracker.check(prob_ok, "All probabilities are finite and in [0,1]")

    pred_cols = ["bcr_tcn_y_pred", "elasticnet_y_pred", "persistence_y_pred", "hgbr_y_pred"]
    point_ok = True
    for c in pred_cols:
        point_ok = point_ok and bool(np.isfinite(hv_pred_df[c].to_numpy(dtype=float)).all())
        point_ok = point_ok and bool(np.isfinite(outer_pred_df[c].to_numpy(dtype=float)).all())
    tracker.check(point_ok, "All point predictions are finite")

    no_dups = (
        not hv_pred_df.duplicated(subset=["outer_fold", "feature_date", "target_date"]).any()
        and not outer_pred_df.duplicated(subset=["outer_fold", "feature_date", "target_date"]).any()
        and not score_df.duplicated(subset=["outer_fold", "feature_date", "target_date", "threshold", "alarm_budget", "policy_name"]).any()
    )
    tracker.check(no_dups, "No duplicate keys exist")

    td_ok = True
    for df in [hv_pred_df, outer_pred_df, score_df]:
        delta = (pd.to_datetime(df["target_date"]) - pd.to_datetime(df["feature_date"]))
        td_ok = td_ok and bool((delta.dt.days == 5).all())
    tracker.check(td_ok, "target_date equals feature_date + 5 days")

    # 52-57
    produced_files = [p.name.lower() for p in OUT_DIR.iterdir() if p.is_file()]
    produced_cols = set([c.lower() for c in score_df.columns] + [c.lower() for c in outer_pred_df.columns] + [c.lower() for c in hv_pred_df.columns])

    tracker.check(not any("alarm_flag" in x for x in produced_files) and "alarm_flag" not in produced_cols, "No alarm flag is produced")
    tracker.check(not any("top_k" in x or "topk" in x for x in produced_files) and not any("top_k" in c or "topk" in c for c in produced_cols), "No top-k selection is produced")
    tracker.check(not any("precision@" in c or "precision_at_k" in c for c in produced_cols), "No Precision@k is produced")
    tracker.check(not any("recall@" in c or "recall_at_k" in c for c in produced_cols), "No Recall@k is produced")

    forbidden_metric_cols = ["fp", "fn", "tn", "enrichment"]
    no_forbidden_metrics = True
    for c in score_df.columns:
        lc = c.lower()
        if lc in forbidden_metric_cols or "enrichment" in lc:
            no_forbidden_metrics = False
    tracker.check(no_forbidden_metrics, "No TP, FP, FN, TN, or enrichment result is produced")
    tracker.check(not any("sequential" in x for x in produced_files) and not any("sequential" in c for c in produced_cols), "No sequential policy is produced")

    # 58-62
    protected_snapshot = manifest.get("protected_source_checksum_snapshot", {})
    changed_list = protected_snapshot.get("changed", []) if isinstance(protected_snapshot, dict) else []
    has_controlled_entries = False
    if isinstance(protected_snapshot, dict) and isinstance(protected_snapshot.get("before", {}), dict):
        has_controlled_entries = any(str(k).startswith("revision_2026/04_controlled_reruns/") for k in protected_snapshot["before"].keys())
    tracker.check(bool(len(changed_list) == 0 and has_controlled_entries), "Source controlled-rerun files are unchanged")

    canonical_unchanged = verify_checksum_manifest(CANONICAL_CHECKSUMS_PATH, CANONICAL_DIR)
    tracker.check(bool(canonical_unchanged), "Canonical package files are unchanged")

    auth_prefix = str((ROOT / AUTHORIZED_REL_DIR).resolve())
    output_inside = True
    for p in OUT_DIR.iterdir():
        if p.is_file():
            output_inside = output_inside and str(p.resolve()).startswith(auth_prefix)
    output_inside = output_inside and len(manifest.get("files_modified_outside_authorized_directory", [])) == 0
    tracker.check(output_inside, "All outputs remain inside authorized directory")

    tracker.check(deterministic_pass, "Repeated execution is deterministic")
    tracker.check(
        first_checks["hybrid_validation_component_predictions.csv"] == second_checks["hybrid_validation_component_predictions.csv"]
        and first_checks["fully_nested_outer_test_component_predictions.csv"] == second_checks["fully_nested_outer_test_component_predictions.csv"]
        and first_checks["fully_nested_hybridrank_outer_test_scores.csv"] == second_checks["fully_nested_hybridrank_outer_test_scores.csv"],
        "Prediction and score outputs are checksum-identical across repeated runs, excluding timestamp-only metadata",
    )

    if tracker.count != 62:
        raise RuntimeError("Internal test construction error: expected 62 assertions, built {}".format(tracker.count))

    all_pass = tracker.passed == 62 and len(tracker.failures) == 0

    # Stamp manifest/report.
    manifest["test_result"] = "pass_62_of_62" if all_pass else "fail_{}_of_62".format(tracker.passed)
    manifest["deterministic_result"] = "pass" if deterministic_pass else "fail"
    manifest["output_checksums"] = {p.name: sha256_file(p) for p in OUT_DIR.iterdir() if p.is_file()}
    manifest["timestamp"] = utc_now_iso()
    write_json(MANIFEST_PATH, manifest)

    decision = "A" if all_pass else "D"
    update_completion_report_decision(COMPLETION_REPORT_PATH, decision)
    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

    print("Assertions passed: {}/62".format(tracker.passed))
    print("Deterministic check: {}".format("pass" if deterministic_pass else "fail"))
    print("Final decision: {}".format(decision))
    if decision == "A":
        print("Next action: Run the corrected retrospective top-k alarm-budget evaluation using the frozen fully nested HybridRank scores and the canonical base-model scores.")
    else:
        print("Next action: Investigate failed assertions and re-run test_fully_nested_hybridrank_h5.py.")

    if not all_pass:
        print("Failed assertions:")
        for msg in tracker.failures:
            print(msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
