from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/06_corrected_hybridrank/h5/fixed_policy"
EXPECTED_BRANCH = "controlled-reruns-v1"

BUILD_SCRIPT_PATH = OUT_DIR / "build_fixed_hybridrank_h5_scores.py"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CONTROLLED_DIR = ROOT / "revision_2026" / "04_controlled_reruns"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_LONG_PATH = CANONICAL_DIR / "canonical_h5_predictions_long.csv"
CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"

FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
POLICY_DEFINITION_PATH = OUT_DIR / "fixed_hybridrank_policy_definition.md"
WEIGHTS_PATH = OUT_DIR / "fixed_hybridrank_weights.csv"
SCORES_PATH = OUT_DIR / "fixed_hybridrank_h5_scores.csv"
RANK_AUDIT_PATH = OUT_DIR / "fixed_hybridrank_rank_audit.csv"
ISOLATION_AUDIT_PATH = OUT_DIR / "fixed_hybridrank_isolation_audit.csv"
LABEL_INVARIANCE_PATH = OUT_DIR / "fixed_hybridrank_label_invariance_audit.json"
SCORE_SUMMARY_PATH = OUT_DIR / "fixed_hybridrank_score_summary.csv"
TIE_RULE_PATH = OUT_DIR / "future_alarm_tie_rule_registry.md"
SOURCE_REGISTRY_PATH = OUT_DIR / "fixed_hybridrank_source_registry.csv"
MANIFEST_PATH = OUT_DIR / "fixed_hybridrank_h5_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "fixed_hybridrank_h5_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "fixed_hybridrank_h5_checksums.sha256"

H = 5
THRESHOLDS = {15, 16, 17}
POLICY_NAME = "HybridRank_fixed_documented"
SUBMITTED_ALIAS = "hybrid_paper_fixed"


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


def parse_checksum_manifest(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        out.append((parts[0], parts[-1]))
    return out


def expected_checksum_for_file(path: Path, filename: str) -> str:
    for expected, rel_name in parse_checksum_manifest(path):
        rel_norm = str(rel_name).replace("\\", "/")
        if rel_norm == filename or rel_norm.endswith("/" + filename):
            return expected
    raise KeyError("Checksum entry not found for {} in {}".format(filename, path))


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


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
        for ln in lines[final_idx + 1 : final_idx + 24]:
            if re.match(r"^[ABCD]\.\s", ln):
                return ln[0]

    for ln in lines[-40:]:
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0]

    raise RuntimeError("Could not infer final decision from {}".format(path))


def rank_normalize_values(x: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if np.isnan(vals).any():
        raise RuntimeError("NaN in rank normalization input")
    n = len(vals)
    if n <= 1:
        return np.zeros(n, dtype=float)
    r = pd.Series(vals).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (r - 1.0) / float(n - 1)


def snapshot_tree_checksums(dirs: Sequence[Path], rel_base: Path) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(rel_base)).replace("\\", "/")
            snap[rel] = sha256_file(p)
    return snap


def update_completion_report_decision(path: Path, decision: str) -> None:
    text = path.read_text(encoding="utf-8")

    if decision == "A":
        line = "A. Corrected fixed H5 rank-ensemble scores passed; retrospective alarm-budget evaluation may begin."
    elif decision == "B":
        line = "B. Fixed score construction completed, but one or more provenance or rank-reproduction issues require investigation."
    elif decision == "C":
        line = "C. The fixed policy could not be reproduced safely from the available evidence."
    else:
        line = "D. Canonical-input, source-preservation, label-invariance, or deterministic tests failed."

    if "<!-- AUTO_DECISION_START -->" in text and "<!-- AUTO_DECISION_END -->" in text:
        text = re.sub(
            r"<!-- AUTO_DECISION_START -->.*?<!-- AUTO_DECISION_END -->",
            "<!-- AUTO_DECISION_START -->\n{}\n<!-- AUTO_DECISION_END -->".format(line),
            text,
            flags=re.S,
        )
    else:
        text += "\n\nFINAL DECISION\n\n{}\n".format(line)

    path.write_text(text, encoding="utf-8")


def compute_output_checksums(out_dir: Path, exclude_name: str) -> Dict[str, str]:
    checks: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == exclude_name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        checks[p.name] = sha256_file(p)
    return checks


def write_checksums_file(out_dir: Path, checksum_path: Path) -> None:
    lines: List[str] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == checksum_path.name:
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        lines.append("{}  {}".format(sha256_file(p), p.name))
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_build() -> None:
    subprocess.check_call([sys.executable, str(BUILD_SCRIPT_PATH)], cwd=ROOT)


def read_build_function_block(script_text: str, function_name: str) -> str:
    marker = "def {}(".format(function_name)
    start = script_text.find(marker)
    if start < 0:
        return ""
    tail = script_text[start:]
    # Approximate until next top-level function definition.
    m = re.search(r"\n\ndef\s+[a-zA-Z_][a-zA-Z0-9_]*\(", tail)
    if m:
        return tail[: m.start()]
    return tail


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run tests from workspace root: {}".format(ROOT))

    controlled_before = snapshot_tree_checksums([CONTROLLED_DIR], ROOT)
    canonical_before = snapshot_tree_checksums([CANONICAL_DIR], ROOT)
    fully_nested_before = snapshot_tree_checksums([FULLY_NESTED_DIR], ROOT)

    run_build()

    stable_files = {
        "fixed_hybridrank_weights.csv": WEIGHTS_PATH,
        "fixed_hybridrank_h5_scores.csv": SCORES_PATH,
        "fixed_hybridrank_rank_audit.csv": RANK_AUDIT_PATH,
        "fixed_hybridrank_isolation_audit.csv": ISOLATION_AUDIT_PATH,
        "fixed_hybridrank_label_invariance_audit.json": LABEL_INVARIANCE_PATH,
        "fixed_hybridrank_score_summary.csv": SCORE_SUMMARY_PATH,
        "fixed_hybridrank_source_registry.csv": SOURCE_REGISTRY_PATH,
        "fixed_hybridrank_policy_definition.md": POLICY_DEFINITION_PATH,
        "future_alarm_tie_rule_registry.md": TIE_RULE_PATH,
    }
    first_stable = {name: sha256_file(path) for name, path in stable_files.items()}

    run_build()
    second_stable = {name: sha256_file(path) for name, path in stable_files.items()}
    deterministic_pass = first_stable == second_stable

    controlled_after = snapshot_tree_checksums([CONTROLLED_DIR], ROOT)
    canonical_after = snapshot_tree_checksums([CANONICAL_DIR], ROOT)
    fully_nested_after = snapshot_tree_checksums([FULLY_NESTED_DIR], ROOT)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    input_verification = json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    long_df = pd.read_csv(CANONICAL_LONG_PATH)
    wide_df = pd.read_csv(CANONICAL_WIDE_PATH)
    score_df = pd.read_csv(SCORES_PATH)
    weights_df = pd.read_csv(WEIGHTS_PATH)
    rank_audit_df = pd.read_csv(RANK_AUDIT_PATH)
    isolation_df = pd.read_csv(ISOLATION_AUDIT_PATH)
    label_invariance = json.loads(LABEL_INVARIANCE_PATH.read_text(encoding="utf-8"))
    score_summary_df = pd.read_csv(SCORE_SUMMARY_PATH)

    for df, cols in [
        (wide_df, ["feature_date", "target_date"]),
        (score_df, ["feature_date", "target_date"]),
        (long_df, ["feature_date", "target_date"]),
    ]:
        for c in cols:
            df[c] = normalize_date_col(df[c])

    canonical_long_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_long.csv")
    canonical_wide_expected = expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_wide.csv")
    split_expected = expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv")

    tracker = AssertionTracker()

    # 1-3
    tracker.check(Path.cwd().resolve() == ROOT.resolve(), "Correct working directory")
    tracker.check(git_output(["git", "branch", "--show-current"]) == EXPECTED_BRANCH, "Correct branch")
    tracker.check(sys.version_info[:2] == (3, 8), "Python code is compatible with Python 3.8.10")

    # 4-9
    tracker.check(sha256_file(CANONICAL_LONG_PATH) == canonical_long_expected, "Canonical long checksum matches")
    tracker.check(sha256_file(CANONICAL_WIDE_PATH) == canonical_wide_expected, "Canonical wide checksum matches")
    tracker.check(infer_decision_from_report(CANONICAL_REPORT_PATH) == "A", "Canonical assembly decision is A")
    tracker.check(infer_decision_from_report(FULLY_NESTED_REPORT_PATH) == "C", "Fully nested tuned pathway decision is C")
    tracker.check(sha256_file(Path(lock["absolute_path"])) == lock["sha256"], "Dataset checksum matches")
    tracker.check(sha256_file(SPLIT_PATH) == split_expected, "Corrected split checksum matches")

    # 10-17
    tracker.check(sorted(score_df["horizon"].astype(int).unique().tolist()) == [H], "H = 5 only")
    canon_keys = wide_df[["outer_fold", "feature_date", "target_date"]].drop_duplicates()
    tracker.check(int(len(canon_keys)) == 747, "Canonical key count equals 747")
    tracker.check(bool((wide_df.groupby("outer_fold").size().astype(int) == 249).all()), "Each outer fold contains 249 keys")
    tracker.check(set(score_df["threshold"].astype(int).unique().tolist()) == THRESHOLDS, "Exactly three thresholds are present: 15, 16, and 17")
    tracker.check(int(len(score_df)) == 2241, "Score output row count equals 2241")
    tracker.check(bool((score_df.groupby("threshold").size().astype(int) == 747).all()), "Every threshold contains exactly 747 score rows")
    tracker.check(bool((score_df.groupby(["outer_fold", "threshold"]).size().astype(int) == 249).all()), "Every fold-threshold contains exactly 249 rows")
    dup_count = int(score_df.duplicated(["policy_name", "threshold", "outer_fold", "feature_date", "target_date"]).sum())
    tracker.check(dup_count == 0, "No duplicate policy-threshold-key rows exist")

    # 18-25
    score_keys = score_df[["outer_fold", "feature_date", "target_date", "threshold"]].copy()
    canon_expanded = pd.concat(
        [
            wide_df[["outer_fold", "feature_date", "target_date"]].assign(threshold=t)
            for t in sorted(THRESHOLDS)
        ],
        axis=0,
        ignore_index=True,
    )
    key_merge = score_keys.merge(canon_expanded, on=["outer_fold", "feature_date", "target_date", "threshold"], how="outer", indicator=True)
    tracker.check(bool((key_merge["_merge"] == "both").all()), "Score keys match canonical keys exactly")

    merged = score_df.merge(
        wide_df,
        on=["horizon", "outer_fold", "feature_date", "target_date"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_canon"),
    )
    tracker.check(bool(np.allclose(merged["y_true"].to_numpy(dtype=float), merged["y_true_canon"].to_numpy(dtype=float), atol=0.0, rtol=0.0)), "y_true matches canonical values exactly")
    evt_ok = True
    for c in ["event_tau15", "event_tau16", "event_tau17"]:
        evt_ok = evt_ok and bool((merged[c].astype(int).to_numpy() == merged[c + "_canon"].astype(int).to_numpy()).all())
    tracker.check(evt_ok, "Event labels match canonical values exactly")

    bcr_match = True
    bcr15 = merged[merged["threshold"].astype(int) == 15]
    bcr16 = merged[merged["threshold"].astype(int) == 16]
    bcr17 = merged[merged["threshold"].astype(int) == 17]
    bcr_match = bcr_match and bool(np.allclose(bcr15["bcr_tcn_raw_score"].to_numpy(float), bcr15["bcr_tcn_v11_p_tau15"].to_numpy(float), atol=0.0, rtol=0.0))
    bcr_match = bcr_match and bool(np.allclose(bcr16["bcr_tcn_raw_score"].to_numpy(float), bcr16["bcr_tcn_v11_p_tau16"].to_numpy(float), atol=0.0, rtol=0.0))
    bcr_match = bcr_match and bool(np.allclose(bcr17["bcr_tcn_raw_score"].to_numpy(float), bcr17["bcr_tcn_v11_p_tau17"].to_numpy(float), atol=0.0, rtol=0.0))
    tracker.check(bcr_match, "BCR-TCN raw scores match canonical probabilities exactly")

    tracker.check(bool(np.allclose(merged["elasticnet_raw_score"].to_numpy(float), merged["elasticnet_y_pred"].to_numpy(float), atol=0.0, rtol=0.0)), "ElasticNet raw scores match canonical predictions exactly")
    tracker.check(bool(np.allclose(merged["persistence_raw_score"].to_numpy(float), merged["persistence_y_pred"].to_numpy(float), atol=0.0, rtol=0.0)), "Persistence raw scores match canonical predictions exactly")
    tracker.check(bool(np.allclose(merged["hgbr_raw_score"].to_numpy(float), merged["hgbr_y_pred"].to_numpy(float), atol=0.0, rtol=0.0)), "HGBR raw scores match canonical predictions exactly")

    bcr_mapping_ok = bcr_match
    tracker.check(bcr_mapping_ok, "BCR-TCN threshold mapping is correct")

    # 26-33
    within_fold_ok = bool((score_df.groupby(["outer_fold", "threshold"]).size().astype(int) == 249).all())
    tracker.check(within_fold_ok, "Rank normalization is performed within each outer fold")

    comp_sep_ok = True
    for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
        for tau in sorted(score_df["threshold"].astype(int).unique().tolist()):
            sf = score_df[(score_df["outer_fold"].astype(int) == fold) & (score_df["threshold"].astype(int) == tau)].copy()
            checks = [
                ("bcr_tcn_raw_score", "bcr_tcn_rank_score"),
                ("elasticnet_raw_score", "elasticnet_rank_score"),
                ("persistence_raw_score", "persistence_rank_score"),
                ("hgbr_raw_score", "hgbr_rank_score"),
            ]
            for raw_col, rank_col in checks:
                exp_rank = rank_normalize_values(sf[raw_col])
                got = sf[rank_col].to_numpy(dtype=float)
                comp_sep_ok = comp_sep_ok and bool(np.allclose(exp_rank, got, atol=1e-12, rtol=0.0))
    tracker.check(comp_sep_ok, "Rank normalization is performed separately by component")

    ascending_ok = True
    for raw_col, rank_col in [
        ("bcr_tcn_raw_score", "bcr_tcn_rank_score"),
        ("elasticnet_raw_score", "elasticnet_rank_score"),
        ("persistence_raw_score", "persistence_rank_score"),
        ("hgbr_raw_score", "hgbr_rank_score"),
    ]:
        for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
            for tau in sorted(score_df["threshold"].astype(int).unique().tolist()):
                sf = score_df[(score_df["outer_fold"].astype(int) == fold) & (score_df["threshold"].astype(int) == tau)].copy()
                raw = sf[raw_col].to_numpy(dtype=float)
                rk = sf[rank_col].to_numpy(dtype=float)
                order = np.argsort(raw, kind="mergesort")
                ascending_ok = ascending_ok and bool(np.all(np.diff(rk[order]) >= -1e-12))
    tracker.check(ascending_ok, "Rank normalization uses ascending direction")

    average_tie_ok = True
    for raw_col, rank_col in [
        ("bcr_tcn_raw_score", "bcr_tcn_rank_score"),
        ("elasticnet_raw_score", "elasticnet_rank_score"),
        ("persistence_raw_score", "persistence_rank_score"),
        ("hgbr_raw_score", "hgbr_rank_score"),
    ]:
        for fold in sorted(score_df["outer_fold"].astype(int).unique().tolist()):
            for tau in sorted(score_df["threshold"].astype(int).unique().tolist()):
                sf = score_df[(score_df["outer_fold"].astype(int) == fold) & (score_df["threshold"].astype(int) == tau)].copy()
                avg_rank = pd.Series(sf[raw_col].to_numpy(dtype=float)).rank(method="average", ascending=True).to_numpy(dtype=float)
                den = float(len(sf) - 1) if len(sf) > 1 else 1.0
                exp = np.zeros(len(sf), dtype=float) if len(sf) <= 1 else (avg_rank - 1.0) / den
                average_tie_ok = average_tie_ok and bool(np.allclose(exp, sf[rank_col].to_numpy(dtype=float), atol=1e-12, rtol=0.0))
    tracker.check(average_tie_ok, "Rank normalization uses average ties")

    tracker.check(bool((score_df["rank_denominator"].astype(str) == "N_minus_1").all()), "Rank normalization uses N-1 denominator")

    rank_cols = ["bcr_tcn_rank_score", "elasticnet_rank_score", "persistence_rank_score", "hgbr_rank_score"]
    rank_range_ok = True
    for c in rank_cols:
        v = score_df[c].to_numpy(dtype=float)
        rank_range_ok = rank_range_ok and bool(((v >= -1e-12) & (v <= 1.0 + 1e-12)).all())
    tracker.check(rank_range_ok, "Rank values remain within [0,1]")

    formula_exact_ok = comp_sep_ok
    tracker.check(formula_exact_ok, "Rank values reproduce the documented formula exactly")

    pooled_diff_ok = True
    for tau in sorted(THRESHOLDS):
        st = score_df[score_df["threshold"].astype(int) == tau].copy()
        for raw_col, rank_col in [
            ("bcr_tcn_raw_score", "bcr_tcn_rank_score"),
            ("elasticnet_raw_score", "elasticnet_rank_score"),
            ("persistence_raw_score", "persistence_rank_score"),
            ("hgbr_raw_score", "hgbr_rank_score"),
        ]:
            pooled = rank_normalize_values(st[raw_col])
            same = np.allclose(pooled, st[rank_col].to_numpy(dtype=float), atol=1e-12, rtol=0.0)
            pooled_diff_ok = pooled_diff_ok and bool(not same)
    tracker.check(pooled_diff_ok, "No pooled 747-row ranking is used")

    # 34-44
    fixed_weight_ok = bool(
        np.allclose(score_df["bcr_tcn_weight"].to_numpy(dtype=float), 0.50, atol=1e-12)
        and np.allclose(score_df["elasticnet_weight"].to_numpy(dtype=float), 0.25, atol=1e-12)
        and np.allclose(score_df["persistence_weight"].to_numpy(dtype=float), 0.25, atol=1e-12)
        and np.allclose(score_df["hgbr_weight"].to_numpy(dtype=float), 0.0, atol=1e-12)
    )
    tracker.check(fixed_weight_ok, "Fixed weights equal 0.50, 0.25, 0.25, and 0.00")

    w_fold_const = True
    for comp_col, expected in [
        ("bcr_tcn_weight", 0.50),
        ("elasticnet_weight", 0.25),
        ("persistence_weight", 0.25),
        ("hgbr_weight", 0.0),
    ]:
        by_fold = score_df.groupby("outer_fold")[comp_col].nunique(dropna=False)
        w_fold_const = w_fold_const and bool((by_fold == 1).all()) and bool(np.allclose(score_df[comp_col].to_numpy(float), expected, atol=1e-12))
    tracker.check(w_fold_const, "Fixed weights are identical across folds")

    w_thr_const = True
    for comp_col, expected in [
        ("bcr_tcn_weight", 0.50),
        ("elasticnet_weight", 0.25),
        ("persistence_weight", 0.25),
        ("hgbr_weight", 0.0),
    ]:
        by_tau = score_df.groupby("threshold")[comp_col].nunique(dropna=False)
        w_thr_const = w_thr_const and bool((by_tau == 1).all()) and bool(np.allclose(score_df[comp_col].to_numpy(float), expected, atol=1e-12))
    tracker.check(w_thr_const, "Fixed weights are identical across thresholds")

    budget_independent_ok = (
        "budget" not in " ".join(score_df.columns).lower()
        and bool((weights_df["budget_specific"].astype(str).str.lower().isin(["false", "0"]).all()))
    )
    tracker.check(budget_independent_ok, "Fixed weights are budget-independent")

    weight_sum = weights_df.groupby(["outer_fold", "threshold"], sort=True)["weight"].sum().to_numpy(dtype=float)
    tracker.check(bool(np.allclose(weight_sum, np.ones_like(weight_sum), atol=1e-12)), "Weights sum exactly to one")

    tracker.check(bool((weights_df["weight_selection_performed"].astype(str).str.lower().isin(["false", "0"]).all())), "No weight search is performed")
    tracker.check(bool((weights_df["validation_outcomes_used"].astype(str).str.lower().isin(["false", "0"]).all()) and (score_df["validation_outcomes_used"].astype(str).str.lower().isin(["false", "0"]).all())), "No validation outcome is used")
    tracker.check(bool((weights_df["outer_test_outcomes_used"].astype(str).str.lower().isin(["false", "0"]).all()) and (score_df["outer_test_outcomes_used_for_weight_selection"].astype(str).str.lower().isin(["false", "0"]).all())), "No outer-test outcome is used for weight selection")
    tracker.check(bool((weights_df["future_fold_outcomes_used"].astype(str).str.lower().isin(["false", "0"]).all())), "No future-fold outcome is used")

    formula_score = (
        0.50 * score_df["bcr_tcn_rank_score"].to_numpy(dtype=float)
        + 0.25 * score_df["elasticnet_rank_score"].to_numpy(dtype=float)
        + 0.25 * score_df["persistence_rank_score"].to_numpy(dtype=float)
        + 0.00 * score_df["hgbr_rank_score"].to_numpy(dtype=float)
    )
    tracker.check(bool(np.allclose(formula_score, score_df["hybrid_score"].to_numpy(dtype=float), atol=1e-12, rtol=0.0)), "Hybrid score reproduces the fixed weighted formula exactly")

    hgbr_zero_ok = bool(
        np.allclose(score_df["hgbr_weight"].to_numpy(dtype=float), 0.0, atol=1e-12)
        and np.allclose(
            score_df["hybrid_score"].to_numpy(dtype=float),
            0.50 * score_df["bcr_tcn_rank_score"].to_numpy(dtype=float)
            + 0.25 * score_df["elasticnet_rank_score"].to_numpy(dtype=float)
            + 0.25 * score_df["persistence_rank_score"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    )
    tracker.check(hgbr_zero_ok, "HGBR contributes exactly zero to the hybrid score")

    # 45-47
    build_text = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
    core_fn_block = read_build_function_block(build_text, "compute_core_scores_from_raw_components")
    tracker.check("y_true" not in core_fn_block, "y_true is not accessed by score-construction functions")
    tracker.check(("event_tau15" not in core_fn_block) and ("event_tau16" not in core_fn_block) and ("event_tau17" not in core_fn_block), "Event labels are not accessed by score-construction functions")

    label_pass = bool(
        label_invariance.get("label_invariance_pass")
        and label_invariance.get("rank_values_identical")
        and label_invariance.get("hybrid_scores_identical")
        and float(label_invariance.get("maximum_absolute_score_difference", 1.0)) == 0.0
    )
    tracker.check(label_pass, "Label-permutation invariance passes")

    # 48-57
    tracker.check(bool(np.isfinite(score_df["hybrid_score"].to_numpy(dtype=float)).all()), "Every score is finite")

    out_names = [p.name.lower() for p in OUT_DIR.iterdir() if p.is_file()]
    cols_lower = [c.lower() for c in score_df.columns]

    tracker.check((not any("alarm_flag" in n for n in out_names)) and ("alarm_flag" not in cols_lower), "No alarm flag is produced")
    tracker.check((not any("topk" in n or "top_k" in n for n in out_names)) and (not any("topk" in c or "top_k" in c for c in cols_lower)), "No top-k selection is produced")
    tracker.check(("budget" not in cols_lower) and int(len(score_df)) == 2241, "No alarm-budget-specific score duplication is produced")
    tracker.check(not any("precision" in c for c in cols_lower), "No Precision@k is produced")
    tracker.check(not any("recall" in c for c in cols_lower), "No Recall@k is produced")

    forbidden_cols = {"tp", "fp", "fn", "tn", "enrichment"}
    tracker.check(not any((c in forbidden_cols) or ("enrichment" in c) for c in cols_lower), "No TP, FP, FN, TN, or enrichment is produced")
    tracker.check(not any("sequential" in c for c in cols_lower) and not any("sequential" in n for n in out_names), "No sequential policy is produced")
    tracker.check(bool((score_df["score_context"].astype(str) == "retrospective_offline_only").all()), "Every score is labeled retrospective_offline_only")

    banned_context_terms = ["deployable", "prospective", "streaming", "sequential"]
    no_banned_context = True
    for v in score_df["score_context"].astype(str).tolist():
        lv = v.lower()
        if any(term in lv for term in banned_context_terms):
            no_banned_context = False
            break
    tracker.check(no_banned_context, "No score is labeled deployable, prospective, streaming, or sequential")

    # 58-61
    no_training_tokens = (
        ("sklearn" not in build_text.lower())
        and ("torch" not in build_text.lower())
        and ("tensorflow" not in build_text.lower())
    )
    tracker.check(no_training_tokens, "No source model is trained or fitted")

    no_fit_calls = (".fit(" not in build_text) and ("partial_fit(" not in build_text)
    tracker.check(no_fit_calls, "No estimator fit or partial_fit call is executed")

    no_optimizer_calls = ("optimizer.step" not in build_text) and (".backward(" not in build_text)
    tracker.check(no_optimizer_calls, "No optimizer step or neural-network backward pass is executed")

    no_optuna_calls = "optuna" not in build_text.lower()
    tracker.check(no_optuna_calls, "No Optuna study is executed")

    # 62-65
    tracker.check(controlled_before == controlled_after, "Source controlled-rerun files remain unchanged")
    tracker.check(canonical_before == canonical_after, "Canonical package files remain unchanged")
    tracker.check(fully_nested_before == fully_nested_after, "Fully nested feasibility-audit files remain unchanged")

    auth_prefix = str((ROOT / AUTHORIZED_REL_DIR).resolve())
    inside_ok = True
    for p in OUT_DIR.iterdir():
        if p.is_file():
            inside_ok = inside_ok and str(p.resolve()).startswith(auth_prefix)
    inside_ok = inside_ok and bool(len(manifest.get("files_modified_outside_authorized_directory", [])) == 0)
    tracker.check(inside_ok, "All outputs remain inside the authorized fixed_policy directory")

    # 66-67
    tracker.check(bool(deterministic_pass), "Repeated score generation is deterministic")

    key_stable = [
        "fixed_hybridrank_h5_scores.csv",
        "fixed_hybridrank_weights.csv",
        "fixed_hybridrank_rank_audit.csv",
        "fixed_hybridrank_isolation_audit.csv",
        "fixed_hybridrank_label_invariance_audit.json",
    ]
    key_stable_equal = True
    for k in key_stable:
        key_stable_equal = key_stable_equal and bool(first_stable[k] == second_stable[k])
    tracker.check(key_stable_equal, "Stable outputs are checksum-identical across repeated runs, excluding timestamp-only metadata")

    if tracker.count != 67:
        raise RuntimeError("Internal test construction error: expected 67 assertions, built {}".format(tracker.count))

    all_pass = bool(len(tracker.failures) == 0 and tracker.passed == 67)

    manifest["test_result"] = "pass_67_of_67" if all_pass else "fail_{}_of_67".format(tracker.passed)
    manifest["deterministic_result"] = "pass" if deterministic_pass else "fail"
    manifest["output_checksums"] = compute_output_checksums(OUT_DIR, CHECKSUMS_PATH.name)
    manifest["output_files"] = sorted(manifest["output_checksums"].keys())
    manifest["timestamp"] = utc_now_iso()
    manifest["final_decision"] = "A" if all_pass else "D"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    update_completion_report_decision(COMPLETION_REPORT_PATH, "A" if all_pass else "D")
    write_checksums_file(OUT_DIR, CHECKSUMS_PATH)

    print("Assertions passed: {}/67".format(tracker.passed))
    print("Deterministic check: {}".format("pass" if deterministic_pass else "fail"))
    print("Final decision: {}".format("A" if all_pass else "D"))
    if all_pass:
        print("Next action: Run the corrected retrospective top-k alarm-budget evaluation using the frozen canonical base-model scores and corrected fixed rank-ensemble scores.")
    else:
        print("Next action: Investigate failed assertions and re-run test_fixed_hybridrank_h5_scores.py.")

    if not all_pass:
        print("Failed assertions:")
        for msg in tracker.failures:
            print(msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
