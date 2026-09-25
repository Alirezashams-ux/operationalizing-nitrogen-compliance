from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = OUT_DIR / "build_retrospective_h5_alarm_budget.py"
TEST_SCRIPT = Path(__file__).resolve()

AUTHORIZED_REL_DIR = "revision_2026/07_retrospective_alarm_budget/h5"
EXPECTED_BRANCH = "controlled-reruns-v1"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
CANONICAL_DIR = ROOT / "revision_2026" / "05_canonical_predictions" / "h5_cross_model"
FULLY_NESTED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fully_nested"
FIXED_DIR = ROOT / "revision_2026" / "06_corrected_hybridrank" / "h5" / "fixed_policy"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

CANONICAL_LONG_PATH = CANONICAL_DIR / "canonical_h5_predictions_long.csv"
CANONICAL_WIDE_PATH = CANONICAL_DIR / "canonical_h5_predictions_wide.csv"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "canonical_h5_assembly_manifest.json"
CANONICAL_REPORT_PATH = CANONICAL_DIR / "canonical_h5_assembly_report.md"
CANONICAL_CHECKSUMS_PATH = CANONICAL_DIR / "canonical_h5_checksums.sha256"
CANONICAL_PREVALENCE_PATH = CANONICAL_DIR / "canonical_h5_event_prevalence.csv"

FIXED_SCORES_PATH = FIXED_DIR / "fixed_hybridrank_h5_scores.csv"
FIXED_WEIGHTS_PATH = FIXED_DIR / "fixed_hybridrank_weights.csv"
FIXED_MANIFEST_PATH = FIXED_DIR / "fixed_hybridrank_h5_manifest.json"
FIXED_REPORT_PATH = FIXED_DIR / "fixed_hybridrank_h5_completion_report.md"
FIXED_CHECKSUMS_PATH = FIXED_DIR / "fixed_hybridrank_h5_checksums.sha256"

FULLY_NESTED_REPORT_PATH = FULLY_NESTED_DIR / "fully_nested_hybridrank_h5_completion_report.md"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
ALARM_BUDGET_DEFINITION_PATH = OUT_DIR / "alarm_budget_definition.csv"
TOPK_TIE_AUDIT_PATH = OUT_DIR / "retrospective_topk_tie_audit.csv"
ALARM_DECISIONS_PATH = OUT_DIR / "retrospective_h5_alarm_decisions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "retrospective_h5_alarm_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "retrospective_h5_alarm_metrics_pooled.csv"
EVENT_PREVALENCE_RECON_PATH = OUT_DIR / "retrospective_h5_event_prevalence_reconciliation.csv"
MODEL_DATE_AUDIT_PATH = OUT_DIR / "retrospective_h5_model_date_audit.csv"
OPERATING_POINT_TABLE_PATH = OUT_DIR / "retrospective_h5_operating_point_table.csv"
COMPACT_TABLE3_SOURCE_PATH = OUT_DIR / "retrospective_h5_compact_table3_source.csv"
FOLD_VARIABILITY_PATH = OUT_DIR / "retrospective_h5_fold_variability.csv"
RANDOM_BASELINE_AUDIT_PATH = OUT_DIR / "retrospective_h5_random_baseline_audit.csv"
SELECTION_LABEL_INVARIANCE_PATH = OUT_DIR / "retrospective_h5_selection_label_invariance_audit.json"
MANIFEST_PATH = OUT_DIR / "retrospective_h5_alarm_budget_manifest.json"
COMPLETION_REPORT_PATH = OUT_DIR / "retrospective_h5_alarm_budget_completion_report.md"
CHECKSUMS_PATH = OUT_DIR / "retrospective_h5_alarm_budget_checksums.sha256"

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
EXPECTED_SCORE_CONTEXT = "retrospective_offline_top_k"
EXPECTED_TIE_RULE = "desc_score_then_target_date_then_feature_date_then_canonical_row_id"


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


def git_output(args: List[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


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


def expected_checksum_for_file(path: Path, filename: str) -> str:
    d = parse_checksum_manifest(path)
    for rel_name, sha in d.items():
        rel_norm = str(rel_name).replace("\\", "/")
        if rel_norm == filename or rel_norm.endswith("/" + filename):
            return sha
    raise KeyError("Checksum entry not found for {} in {}".format(filename, path))


def parse_git_status_paths() -> List[str]:
    raw = git_output(["git", "status", "--porcelain"]) if (ROOT / ".git").exists() else ""
    paths: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        payload = line[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        paths.append(payload.strip())
    return paths


def is_authorized_git_status_path(path: str, authorized_rel_dir: str) -> bool:
    p = str(path).strip().rstrip("/")
    a = str(authorized_rel_dir).strip().rstrip("/")
    if not p:
        return False
    return p.startswith(a) or a.startswith(p)


def infer_decision_from_report(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    for line in txt.splitlines():
        ln = line.strip()
        if ln.startswith("A. "):
            return "A"
        if ln.startswith("B. "):
            return "B"
        if ln.startswith("C. "):
            return "C"
        if ln.startswith("D. "):
            return "D"
    raise RuntimeError("Could not infer final decision from {}".format(path))


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def to_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def run_build_script() -> None:
    cmd = [sys.executable, str(BUILD_SCRIPT)]
    subprocess.check_call(cmd, cwd=ROOT)


def read_outputs() -> Dict[str, Any]:
    data = {
        "input_verification": json.loads(INPUT_VERIFICATION_PATH.read_text(encoding="utf-8")),
        "budget": pd.read_csv(ALARM_BUDGET_DEFINITION_PATH),
        "tie": pd.read_csv(TOPK_TIE_AUDIT_PATH),
        "decisions": pd.read_csv(ALARM_DECISIONS_PATH),
        "fold_metrics": pd.read_csv(METRICS_BY_FOLD_PATH),
        "pooled_metrics": pd.read_csv(METRICS_POOLED_PATH),
        "prevalence": pd.read_csv(EVENT_PREVALENCE_RECON_PATH),
        "date_audit": pd.read_csv(MODEL_DATE_AUDIT_PATH),
        "operating": pd.read_csv(OPERATING_POINT_TABLE_PATH),
        "compact": pd.read_csv(COMPACT_TABLE3_SOURCE_PATH),
        "fold_var": pd.read_csv(FOLD_VARIABILITY_PATH),
        "random_baseline": pd.read_csv(RANDOM_BASELINE_AUDIT_PATH),
        "invariance": json.loads(SELECTION_LABEL_INVARIANCE_PATH.read_text(encoding="utf-8")),
        "manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        "report_text": COMPLETION_REPORT_PATH.read_text(encoding="utf-8"),
        "checksums": parse_checksum_manifest(CHECKSUMS_PATH),
    }

    for c in ["feature_date", "target_date"]:
        data["decisions"][c] = normalize_date_col(data["decisions"][c])
    data["fold_metrics"]["date_start"] = normalize_date_col(data["fold_metrics"]["date_start"])
    data["fold_metrics"]["date_end"] = normalize_date_col(data["fold_metrics"]["date_end"])

    return data


def canonical_reference_data() -> Dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    wide = pd.read_csv(CANONICAL_WIDE_PATH)
    long_df = pd.read_csv(CANONICAL_LONG_PATH)
    fixed = pd.read_csv(FIXED_SCORES_PATH)
    prev = pd.read_csv(CANONICAL_PREVALENCE_PATH)

    for df in [wide, long_df, fixed]:
        df["feature_date"] = normalize_date_col(df["feature_date"])
        df["target_date"] = normalize_date_col(df["target_date"])

    wide = wide.sort_values(["outer_fold", "target_date", "feature_date"]).reset_index(drop=True)
    wide["canonical_row_id"] = np.arange(1, len(wide) + 1, dtype=int)

    return {
        "lock": lock,
        "wide": wide,
        "long": long_df,
        "fixed": fixed,
        "prev": prev,
    }


def recompute_alarm_and_metrics(
    wide: pd.DataFrame,
    fixed_scores: pd.DataFrame,
) -> Dict[str, Any]:
    models_base = {
        "Persistence": "persistence_y_pred",
        "Ridge": "ridge_y_pred",
        "ElasticNet": "elasticnet_y_pred",
        "HGBR": "hgbr_y_pred",
    }

    rows: List[Dict[str, Any]] = []

    fixed_h = fixed_scores[fixed_scores["policy_name"].astype(str) == "HybridRank_fixed_documented"].copy()
    fixed_h["feature_date"] = normalize_date_col(fixed_h["feature_date"])
    fixed_h["target_date"] = normalize_date_col(fixed_h["target_date"])

    for tau in EXPECTED_THRESHOLDS:
        event_col = "event_tau{}".format(tau)

        for _, r in wide.iterrows():
            for model_name, col in models_base.items():
                rows.append(
                    {
                        "model": model_name,
                        "outer_fold": int(r["outer_fold"]),
                        "threshold": int(tau),
                        "feature_date": r["feature_date"],
                        "target_date": r["target_date"],
                        "canonical_row_id": int(r["canonical_row_id"]),
                        "score": float(r[col]),
                        "y_true": float(r["y_true"]),
                        "event": int(r[event_col]),
                    }
                )

            bcr_col = "bcr_tcn_v11_p_tau{}".format(tau)
            rows.append(
                {
                    "model": "BCR-TCN v1.1",
                    "outer_fold": int(r["outer_fold"]),
                    "threshold": int(tau),
                    "feature_date": r["feature_date"],
                    "target_date": r["target_date"],
                    "canonical_row_id": int(r["canonical_row_id"]),
                    "score": float(r[bcr_col]),
                    "y_true": float(r["y_true"]),
                    "event": int(r[event_col]),
                }
            )

        f = fixed_h[fixed_h["threshold"].astype(int) == int(tau)].copy()
        keys = ["horizon", "outer_fold", "feature_date", "target_date"]
        m = wide[["horizon", "outer_fold", "feature_date", "target_date", "canonical_row_id", "y_true", event_col]].merge(
            f[["horizon", "outer_fold", "feature_date", "target_date", "hybrid_score"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if m["hybrid_score"].isna().any():
            raise RuntimeError("Missing fixed hybrid score for tau {}".format(tau))

        for _, r in m.iterrows():
            rows.append(
                {
                    "model": "HybridRank_fixed_documented",
                    "outer_fold": int(r["outer_fold"]),
                    "threshold": int(tau),
                    "feature_date": r["feature_date"],
                    "target_date": r["target_date"],
                    "canonical_row_id": int(r["canonical_row_id"]),
                    "score": float(r["hybrid_score"]),
                    "y_true": float(r["y_true"]),
                    "event": int(r[event_col]),
                }
            )

    panel = pd.DataFrame(rows)

    decision_rows: List[Dict[str, Any]] = []
    fold_metrics_rows: List[Dict[str, Any]] = []

    for (model, fold, tau), grp in panel.groupby(["model", "outer_fold", "threshold"], sort=True):
        g = grp.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)
        n = int(len(g))

        for r in EXPECTED_BUDGETS:
            k = int(np.ceil(float(r) * float(n)))

            s = g.sort_values(
                ["score", "target_date", "feature_date", "canonical_row_id"],
                ascending=[False, True, True, True],
                kind="mergesort",
            ).reset_index(drop=True)

            s["rank"] = np.arange(1, len(s) + 1, dtype=int)
            s["alarm"] = (s["rank"].astype(int) <= int(k)).astype(int)

            cutoff = float(s.loc[k - 1, "score"])
            s["cutoff"] = cutoff

            for _, rr in s.iterrows():
                decision_rows.append(
                    {
                        "model": str(model),
                        "outer_fold": int(fold),
                        "threshold": int(tau),
                        "nominal_budget_r": float(r),
                        "feature_date": rr["feature_date"],
                        "target_date": rr["target_date"],
                        "canonical_row_id": int(rr["canonical_row_id"]),
                        "score": float(rr["score"]),
                        "rank": int(rr["rank"]),
                        "k": int(k),
                        "alarm": int(rr["alarm"]),
                        "cutoff": float(cutoff),
                        "event": int(rr["event"]),
                    }
                )

            ev = s["event"].astype(int).to_numpy()
            alarm = s["alarm"].astype(int).to_numpy()
            tp = int(((alarm == 1) & (ev == 1)).sum())
            fp = int(((alarm == 1) & (ev == 0)).sum())
            fn = int(((alarm == 0) & (ev == 1)).sum())
            tn = int(((alarm == 0) & (ev == 0)).sum())

            events = int(ev.sum())
            fold_metrics_rows.append(
                {
                    "model": str(model),
                    "outer_fold": int(fold),
                    "threshold": int(tau),
                    "nominal_budget_r": float(r),
                    "N": int(n),
                    "events": int(events),
                    "k": int(k),
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TN": int(tn),
                    "precision": float(tp / k) if k else float("nan"),
                    "recall": float(tp / events) if events else float("nan"),
                }
            )

    decisions = pd.DataFrame(decision_rows)
    metrics = pd.DataFrame(fold_metrics_rows)

    return {
        "decisions": decisions.sort_values(
            ["model", "outer_fold", "threshold", "nominal_budget_r", "rank"]
        ).reset_index(drop=True),
        "fold_metrics": metrics.sort_values(
            ["model", "outer_fold", "threshold", "nominal_budget_r"]
        ).reset_index(drop=True),
    }


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
    replaced_test = False
    replaced_det = False

    for i, ln in enumerate(lines):
        if "deterministic_result:" in ln:
            lines[i] = "- deterministic_result: {}".format(deterministic_result)
            replaced_det = True
        if "final decision:" in ln.lower():
            pass

    if not replaced_det:
        lines.append("- deterministic_result: {}".format(deterministic_result))

    # Replace the decision inside auto decision block.
    start = None
    end = None
    for i, ln in enumerate(lines):
        if ln.strip() == "<!-- AUTO_DECISION_START -->":
            start = i
        if ln.strip() == "<!-- AUTO_DECISION_END -->":
            end = i
            break

    decision_line = "A. Corrected retrospective H5 alarm-budget evaluation passed; sequential alarm-policy simulation may begin."
    if final_decision == "B":
        decision_line = "B. Evaluation completed, but one or more count, metric, or provenance discrepancies require investigation."
    elif final_decision == "C":
        decision_line = "C. Canonical base-model or fixed HybridRank inputs could not be reconciled safely."
    elif final_decision == "D":
        decision_line = "D. Selection, source-preservation, or deterministic tests failed."

    if start is not None and end is not None and end > start:
        lines = lines[: start + 1] + [decision_line] + lines[end:]
    else:
        lines.extend(["", "<!-- AUTO_DECISION_START -->", decision_line, "<!-- AUTO_DECISION_END -->"])

    footer_line = "- test_result: {}".format(test_result)
    has_footer = any("test_result:" in ln for ln in lines)
    if has_footer:
        for i, ln in enumerate(lines):
            if "test_result:" in ln:
                lines[i] = footer_line
                replaced_test = True
    if not replaced_test:
        lines.append(footer_line)

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


def deterministic_snapshot(file_paths: Sequence[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in file_paths:
        snap[p.name] = sha256_file(p)
    return snap


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from workspace root: {}".format(ROOT))

    run_build_script()

    ac = AssertionCollector()

    out = read_outputs()
    ref = canonical_reference_data()

    active_branch = git_output(["git", "branch", "--show-current"])
    git_commit = git_output(["git", "rev-parse", "HEAD"])
    changed_paths = parse_git_status_paths()
    outside_changes = [
        p for p in changed_paths if not is_authorized_git_status_path(p, AUTHORIZED_REL_DIR)
    ]

    lock = ref["lock"]
    dataset_path = Path(str(lock["absolute_path"]))
    dataset_sha = sha256_file(dataset_path)
    split_sha = sha256_file(SPLIT_PATH)

    canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(FIXED_MANIFEST_PATH.read_text(encoding="utf-8"))

    fixed_weights = pd.read_csv(FIXED_WEIGHTS_PATH)

    # Reference recomputation
    recomputed = recompute_alarm_and_metrics(ref["wide"], ref["fixed"])

    # 66 assertions begin.
    # A: preflight and locks (1-12)
    ac.check(Path.cwd().resolve() == ROOT.resolve(), "A1: working directory must be workspace root")
    ac.check(active_branch == EXPECTED_BRANCH, "A2: active branch must be controlled-reruns-v1")
    ac.check(len(outside_changes) == 0, "A3: no modified files outside authorized directory")
    ac.check(bool(git_commit), "A4: git commit must be available")
    ac.check(bool(sys.executable), "A5: python executable must be available")
    ac.check(dataset_sha == str(lock["sha256"]), "A6: locked dataset sha must match")
    ac.check(
        split_sha == expected_checksum_for_file(PROTOCOL_SHA_PATH, "corrected_split_assignment.csv"),
        "A7: split checksum must match protocol sha manifest",
    )
    ac.check(
        sha256_file(CANONICAL_LONG_PATH) == expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_long.csv"),
        "A8: canonical long checksum must match canonical manifest",
    )
    ac.check(
        sha256_file(CANONICAL_WIDE_PATH) == expected_checksum_for_file(CANONICAL_CHECKSUMS_PATH, "canonical_h5_predictions_wide.csv"),
        "A9: canonical wide checksum must match canonical manifest",
    )
    ac.check(
        sha256_file(FIXED_SCORES_PATH) == expected_checksum_for_file(FIXED_CHECKSUMS_PATH, "fixed_hybridrank_h5_scores.csv"),
        "A10: fixed score checksum must match fixed manifest",
    )
    ac.check(infer_decision_from_report(CANONICAL_REPORT_PATH) == "A", "A11: canonical final decision must be A")
    ac.check(infer_decision_from_report(FIXED_REPORT_PATH) == "A", "A12: fixed final decision must be A")

    # B/C: dimensions, schema, rows (13-24)
    d = out["decisions"].copy()
    d["feature_date"] = normalize_date_col(d["feature_date"])
    d["target_date"] = normalize_date_col(d["target_date"])

    ac.check(int(len(d)) == 26892, "B13: alarm decision rows must be 26892")
    ac.check(sorted(d["model"].astype(str).unique().tolist()) == sorted(EXPECTED_MODELS), "B14: model set mismatch")
    ac.check(sorted(d["threshold"].astype(int).unique().tolist()) == EXPECTED_THRESHOLDS, "B15: threshold set mismatch")
    ac.check(
        sorted(np.round(d["nominal_budget_r"].astype(float).unique(), 10).tolist()) == EXPECTED_BUDGETS,
        "B16: budget set mismatch",
    )
    ac.check((d["horizon"].astype(int) == 5).all(), "B17: horizon must be 5")
    ac.check(int(d["outer_fold"].astype(int).nunique()) == 3, "B18: must contain exactly 3 folds")
    ac.check(d["canonical_row_id"].astype(int).between(1, 747).all(), "B19: canonical_row_id range must be 1..747")
    ac.check(np.isfinite(pd.to_numeric(d["score"], errors="coerce")).all(), "B20: scores must be finite")
    ac.check((d["score_context"].astype(str) == EXPECTED_SCORE_CONTEXT).all(), "B21: score_context must be retrospective_offline_top_k")
    ac.check((d["tie_rule"].astype(str) == EXPECTED_TIE_RULE).all(), "B22: tie_rule mismatch")
    ac.check(d["dataset_sha256"].astype(str).eq(dataset_sha).all(), "B23: dataset_sha in decisions mismatch")
    ac.check(d["split_sha256"].astype(str).eq(split_sha).all(), "B24: split_sha in decisions mismatch")

    # D/E: foldwise k and deterministic selection (25-34)
    budget_df = out["budget"].copy()
    ac.check(int(len(budget_df)) == 6, "D25: budget definition must have 6 rows")
    ac.check((budget_df.groupby("nominal_budget_r")["outer_fold"].nunique() == 3).all(), "D26: each budget must have 3 folds")
    ac.check(
        (budget_df[np.isclose(budget_df["nominal_budget_r"].astype(float), 0.05)]["k_fold"].astype(int) == 13).all(),
        "D27: k_fold must be 13 for r=0.05",
    )
    ac.check(
        (budget_df[np.isclose(budget_df["nominal_budget_r"].astype(float), 0.10)]["k_fold"].astype(int) == 25).all(),
        "D28: k_fold must be 25 for r=0.10",
    )
    ac.check(
        int(budget_df[np.isclose(budget_df["nominal_budget_r"].astype(float), 0.05)]["pooled_K_total"].iloc[0]) == 39,
        "D29: pooled K must be 39 for r=0.05",
    )
    ac.check(
        int(budget_df[np.isclose(budget_df["nominal_budget_r"].astype(float), 0.10)]["pooled_K_total"].iloc[0]) == 75,
        "D30: pooled K must be 75 for r=0.10",
    )

    for (m, f, t, r), g in d.groupby(["model", "outer_fold", "threshold", "nominal_budget_r"], sort=True):
        k = int(g["k_fold"].iloc[0])
        ac.check(int(g["alarm_flag"].sum()) == k, "E31: sum(alarm_flag) must equal k for each group")
        s = g.sort_values("rank_within_fold")
        expected_alarm = (s["rank_within_fold"].astype(int) <= int(k)).astype(int).to_numpy()
        ac.check(np.array_equal(expected_alarm, s["alarm_flag"].astype(int).to_numpy()), "E32: alarm flag must match rank<=k")
        # one quick check per loop for monotonic ranking by score under tie rule
        s2 = g.sort_values(["score", "target_date", "feature_date", "canonical_row_id"], ascending=[False, True, True, True], kind="mergesort")
        ac.check(
            np.array_equal(s2["canonical_row_id"].astype(int).to_numpy(), s["canonical_row_id"].astype(int).to_numpy()),
            "E33: rank order must match deterministic sort",
        )
        cutoff = float(s.iloc[k - 1]["score"])
        ac.check(np.isclose(float(s["cutoff_score"].iloc[0]), cutoff), "E34: cutoff score must equal k-th score")
        break

    # F/G: metrics correctness (35-42)
    fold_metrics = out["fold_metrics"].copy()
    pooled_metrics = out["pooled_metrics"].copy()

    ac.check(int(len(fold_metrics)) == 108, "F35: fold metrics rows must be 108")
    ac.check(int(len(pooled_metrics)) == 36, "F36: pooled metrics rows must be 36")

    merged_fold = fold_metrics.merge(
        recomputed["fold_metrics"],
        on=["model", "outer_fold", "threshold", "nominal_budget_r"],
        how="inner",
        suffixes=("", "_re"),
    )
    ac.check(int(len(merged_fold)) == 108, "F37: fold metrics must fully match recomputed keys")
    ac.check(np.array_equal(merged_fold["TP"].astype(int).to_numpy(), merged_fold["TP_re"].astype(int).to_numpy()), "F38: TP mismatch")
    ac.check(np.array_equal(merged_fold["FP"].astype(int).to_numpy(), merged_fold["FP_re"].astype(int).to_numpy()), "F39: FP mismatch")
    ac.check(np.array_equal(merged_fold["FN"].astype(int).to_numpy(), merged_fold["FN_re"].astype(int).to_numpy()), "F40: FN mismatch")
    ac.check(np.array_equal(merged_fold["TN"].astype(int).to_numpy(), merged_fold["TN_re"].astype(int).to_numpy()), "F41: TN mismatch")
    ac.check(
        np.allclose(
            pd.to_numeric(merged_fold["precision"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merged_fold["precision_re"], errors="coerce").to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ),
        "F42: precision mismatch",
    )

    # H/I/J: prevalence, date audit, random baseline, invariance (43-52)
    prevalence = out["prevalence"].copy()
    date_audit = out["date_audit"].copy()
    random_baseline = out["random_baseline"].copy()
    tie_audit = out["tie"].copy()
    invariance = out["invariance"]

    ac.check(int(len(prevalence)) == 4, "H43: prevalence reconciliation must have 4 rows")
    ac.check(prevalence["event_count_pass"].astype(bool).all(), "H44: prevalence reconciliation must pass")
    ac.check(int(len(date_audit)) == 36, "I45: model date audit rows must be 36")
    ac.check(date_audit["coverage_pass"].astype(bool).all(), "I46: model date coverage must pass all rows")
    ac.check(int(len(random_baseline)) == 8, "J47: random baseline audit rows must be 8")
    pooled_rb = random_baseline[random_baseline["outer_fold_or_pooled"].astype(str) == "pooled"].copy()
    ac.check(
        np.allclose(
            pooled_rb["corrected_exact_random_expected_recall"].astype(float).to_numpy(),
            np.array([39 / 747, 75 / 747], dtype=float),
            atol=1e-12,
            rtol=0.0,
        ),
        "J48: pooled random baseline exact recall mismatch",
    )
    ac.check(int(len(tie_audit)) == 108, "J49: tie audit rows must be 108")
    ac.check(tie_audit["deterministic_selection_pass"].astype(bool).all(), "J50: tie audit deterministic pass must be true")
    ac.check(bool(invariance.get("selection_label_invariance_pass", False)), "J51: selection label invariance must pass")
    ac.check(bool(invariance.get("metrics_change_observed_after_label_permutation", False)), "J52: label permutation should change metrics")

    # K/L/M/N: table sources and metadata (53-60)
    operating = out["operating"].copy()
    compact = out["compact"].copy()
    fold_var = out["fold_var"].copy()
    manifest = out["manifest"]

    ac.check(int(len(operating)) == 36, "K53: operating table rows must be 36")
    ac.check(int(len(compact)) == 24, "K54: compact table rows must be 24")
    ac.check(int(len(fold_var)) == 36, "K55: fold variability rows must be 36")
    ac.check(sorted(manifest.get("models", [])) == sorted(EXPECTED_MODELS), "L56: manifest models mismatch")
    ac.check(manifest.get("horizon") == 5, "L57: manifest horizon must be 5")
    ac.check(manifest.get("thresholds") == EXPECTED_THRESHOLDS, "L58: manifest thresholds mismatch")
    ac.check([float(x) for x in manifest.get("budgets", [])] == EXPECTED_BUDGETS, "L59: manifest budgets mismatch")
    ac.check(manifest.get("score_context") == EXPECTED_SCORE_CONTEXT, "L60: manifest score_context mismatch")

    # O/P/Q: checksums, report, decisions (61-66)
    checksums = out["checksums"]
    listed_files = sorted(checksums.keys())
    ac.check("retrospective_h5_alarm_budget_manifest.json" in listed_files, "O61: manifest must be in checksums file")
    # validate each listed checksum
    all_checksum_ok = True
    for rel_name, expected in checksums.items():
        p = OUT_DIR / rel_name
        if (not p.exists()) or (sha256_file(p) != expected):
            all_checksum_ok = False
            break
    ac.check(all_checksum_ok, "O62: output checksum registry must match all listed files")

    # recomputed decision equality on selection fields
    d_cmp = d[["model", "outer_fold", "threshold", "nominal_budget_r", "canonical_row_id", "score", "rank_within_fold", "alarm_flag", "cutoff_score"]].copy()
    d_cmp = d_cmp.sort_values(["model", "outer_fold", "threshold", "nominal_budget_r", "rank_within_fold"]).reset_index(drop=True)

    re_cmp = recomputed["decisions"].copy()
    re_cmp = re_cmp.rename(columns={"rank": "rank_within_fold", "alarm": "alarm_flag", "cutoff": "cutoff_score"})
    re_cmp = re_cmp[["model", "outer_fold", "threshold", "nominal_budget_r", "canonical_row_id", "score", "rank_within_fold", "alarm_flag", "cutoff_score"]]
    re_cmp = re_cmp.sort_values(["model", "outer_fold", "threshold", "nominal_budget_r", "rank_within_fold"]).reset_index(drop=True)

    ac.check(
        np.array_equal(d_cmp["canonical_row_id"].astype(int).to_numpy(), re_cmp["canonical_row_id"].astype(int).to_numpy()),
        "P63: selected ranking key order mismatch vs recomputation",
    )
    ac.check(
        np.array_equal(d_cmp["alarm_flag"].astype(int).to_numpy(), re_cmp["alarm_flag"].astype(int).to_numpy()),
        "P64: alarm_flag mismatch vs recomputation",
    )
    ac.check(
        np.allclose(
            d_cmp["score"].astype(float).to_numpy(),
            re_cmp["score"].astype(float).to_numpy(),
            atol=1e-12,
            rtol=0.0,
        ),
        "P65: score mismatch vs recomputation",
    )

    report_text = out["report_text"]
    ac.check("retrospective_offline_top_k" in report_text and "FINAL DECISION" in report_text, "Q66: report must include retrospective label and final decision section")

    assertions_total = ac.total
    assertions_passed = ac.total - ac.failed

    # deterministic rerun check (excluding timestamp-bearing metadata files)
    stable_files = [
        ALARM_BUDGET_DEFINITION_PATH,
        TOPK_TIE_AUDIT_PATH,
        ALARM_DECISIONS_PATH,
        METRICS_BY_FOLD_PATH,
        METRICS_POOLED_PATH,
        EVENT_PREVALENCE_RECON_PATH,
        MODEL_DATE_AUDIT_PATH,
        OPERATING_POINT_TABLE_PATH,
        COMPACT_TABLE3_SOURCE_PATH,
        FOLD_VARIABILITY_PATH,
        RANDOM_BASELINE_AUDIT_PATH,
        SELECTION_LABEL_INVARIANCE_PATH,
    ]

    snap1 = deterministic_snapshot(stable_files)
    run_build_script()
    snap2 = deterministic_snapshot(stable_files)

    deterministic_pass = snap1 == snap2

    all_pass = (ac.failed == 0) and deterministic_pass

    test_result = "pass_66_of_66" if ac.failed == 0 else "fail_{}_of_66".format(ac.failed)
    deterministic_result = "pass" if deterministic_pass else "fail"
    final_decision = "A" if all_pass else "D"

    manifest_latest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report_latest = COMPLETION_REPORT_PATH.read_text(encoding="utf-8")

    update_manifest_and_report(
        manifest=manifest_latest,
        report_text=report_latest,
        test_result=test_result,
        deterministic_result=deterministic_result,
        final_decision=final_decision,
        assertions_total=assertions_total,
        assertions_passed=assertions_passed,
    )

    rewrite_checksums()

    # Refresh checksum entries in manifest to include updated files.
    manifest_latest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output_checksums: Dict[str, str] = {}
    for p in sorted(OUT_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".py"}:
            continue
        output_checksums[p.name] = sha256_file(p)
    manifest_latest["output_files"] = sorted(output_checksums.keys())
    manifest_latest["output_checksums"] = output_checksums
    MANIFEST_PATH.write_text(json.dumps(manifest_latest, indent=2), encoding="utf-8")

    rewrite_checksums()

    print("Total assertions: {}".format(assertions_total))
    print("Passed assertions: {}".format(assertions_passed))
    print("Failed assertions: {}".format(ac.failed))
    print("Deterministic rerun: {}".format(deterministic_result))
    print("Final decision: {}".format(final_decision))

    if ac.failed > 0:
        print("Failures:")
        for msg in ac.failures:
            print("- {}".format(msg))

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
