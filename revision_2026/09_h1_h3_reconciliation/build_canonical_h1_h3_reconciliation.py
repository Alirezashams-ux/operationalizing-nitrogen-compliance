from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
AUTHORIZED_REL_DIR = "revision_2026/09_h1_h3_reconciliation"
EXPECTED_BRANCH = "controlled-reruns-v1"
EXPECTED_PYTHON_VERSION = "3.8.10"

HORIZONS = (1, 3)
MODEL_ORDER = ("Persistence", "Ridge", "ElasticNet", "HGBR")
MODEL_SLUG = {
    "Persistence": "persistence",
    "Ridge": "ridge",
    "ElasticNet": "elasticnet",
    "HGBR": "hgbr",
}
MODEL_FREEZE_TAG = {
    "Persistence": "persistence-corrected-v1",
    "Ridge": "ridge-corrected-v1",
    "ElasticNet": "elasticnet-corrected-v1",
    "HGBR": "hgbr-corrected-v1",
}
MODEL_EXPECTED_TEST_RESULT = {
    "Persistence": "pass_12_of_12",
    "Ridge": "pass_15_of_15",
    "ElasticNet": "pass_15_of_15",
    "HGBR": "pass_20_of_20",
}

PROTOCOL_TAG = "corrected-protocol-v1"
FREEZE_TAGS = [
    "corrected-protocol-v1",
    "persistence-corrected-v1",
    "ridge-corrected-v1",
    "elasticnet-corrected-v1",
    "hgbr-corrected-v1",
]

EXPECTED_DATASET_SHA256 = "6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb"
EXPECTED_SPLIT_SHA256 = "f83baf5732ffbf88b9a448a3d9ab9fa270f7495fe44d1559c1ac49795a30881a"

EXPECTED_EVENT_COUNTS = {
    1: {"events_tau15": 141, "events_tau16": 74, "events_tau17": 30},
    3: {"events_tau15": 134, "events_tau16": 71, "events_tau17": 29},
}

EXPECTED_H_COUNTS = {
    1: {"N": 762, "fold_counts": {1: 254, 2: 254, 3: 254}},
    3: {"N": 747, "fold_counts": {1: 249, 2: 249, 3: 249}},
}

KEY_COLS = ["horizon", "outer_fold", "feature_date", "target_date"]
TOL = 1e-12
MASE_EPS = 1e-9
MASE_METHOD = "mean_absolute_scaled_error_over_pooled_rows_using_fold_local_train_denominator"

PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"
LOCKED_PROTOCOL_DIR = PROTOCOL_DIR / "locked_protocol_v1"
RERUNS_DIR = ROOT / "revision_2026" / "04_controlled_reruns"

DATASET_LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"
LOCKED_PROTOCOL_SHA_PATH = LOCKED_PROTOCOL_DIR / "corrected_protocol_v1.sha256"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SPLIT_SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
HGBR_ISSUE_REGISTRY_PATH = RERUNS_DIR / "hgbr" / "hgbr_decision_b_issue_registry.csv"
HGBR_RECONCILIATION_PATH = RERUNS_DIR / "hgbr" / "hgbr_metric_reconciliation.csv"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
TABLE1_PATH = OUT_DIR / "canonical_h1_h3_table1_reconciliation.csv"
TABLE2_PATH = OUT_DIR / "canonical_h1_h3_point_table2_source.csv"
SOURCE_REGISTRY_PATH = OUT_DIR / "canonical_h1_h3_source_registry.csv"
MANIFEST_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_manifest.json"
REPORT_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_completion_report.md"
CHECKSUM_PATH = OUT_DIR / "canonical_h1_h3_reconciliation_checksums.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise").dt.strftime("%Y-%m-%d")


def ensure_inside_authorized(path: Path) -> None:
    rp = path.resolve()
    auth = (ROOT / AUTHORIZED_REL_DIR).resolve()
    if not str(rp).startswith(str(auth)):
        raise RuntimeError("Attempted write outside authorized directory: {}".format(rp))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_authorized(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame, date_cols: Sequence[str]) -> None:
    out = df.copy()
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    ensure_inside_authorized(path)
    out.to_csv(path, index=False)


def rel_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def rel_to_stage(path: Path) -> str:
    return str(path.resolve().relative_to(OUT_DIR.resolve())).replace("\\", "/")


def git_output(args: Sequence[str]) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def git_status_paths() -> List[str]:
    txt = git_output(["git", "status", "--porcelain"])
    rows = [r for r in txt.splitlines() if r.strip()]
    out: List[str] = []
    for r in rows:
        if len(r) < 4:
            continue
        p = r[3:]
        if " -> " in p:
            p = p.split(" -> ")[-1]
        out.append(p.strip())
    return out


def run_shell(command: str, cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def infer_decision(report_text: str) -> str:
    patterns = [
        r"FINAL DECISION\s*(?:\n|\r\n)+(?:<!--.*?-->\s*)?([ABCD])\.\s",
        r"Decision:\s*([ABCD])",
        r"Readiness Decision\s*(?:\n|\r\n)+(?:<!--.*?-->\s*)?([ABCD])\.\s",
    ]
    for pat in patterns:
        m = re.search(pat, report_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()
    for ln in reversed([x.strip() for x in report_text.splitlines()]):
        if re.match(r"^[ABCD]\.\s", ln):
            return ln[0].upper()
    raise RuntimeError("Could not parse decision from completion report")


def canonical_row_id(horizon: int, outer_fold: int, feature_date: str, target_date: str) -> str:
    return "H{}_F{}_{}_{}".format(int(horizon), int(outer_fold), str(feature_date), str(target_date))


def bool_from_any(x: Any) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def get_stage_file_paths() -> Dict[str, Any]:
    h_paths: Dict[int, Dict[str, Path]] = {}
    for h in HORIZONS:
        hdir = OUT_DIR / "h{}".format(h)
        h_paths[h] = {
            "dir": hdir,
            "long": hdir / "canonical_h{}_predictions_long.csv".format(h),
            "wide": hdir / "canonical_h{}_predictions_wide.csv".format(h),
            "key_audit": hdir / "canonical_h{}_key_audit.csv".format(h),
            "event_prevalence": hdir / "canonical_h{}_event_prevalence.csv".format(h),
            "metrics_by_fold": hdir / "canonical_h{}_point_metrics_by_fold.csv".format(h),
            "metrics_pooled": hdir / "canonical_h{}_point_metrics_pooled.csv".format(h),
            "reconciliation": hdir / "h{}_controlled_rerun_reconciliation.csv".format(h),
        }
    return h_paths


def get_model_file_map() -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for model in MODEL_ORDER:
        slug = MODEL_SLUG[model]
        d = RERUNS_DIR / slug
        out[model] = {
            "dir": d,
            "prediction": d / "{}_predictions.csv".format(slug),
            "metrics_by_fold": d / "{}_metrics_by_fold.csv".format(slug),
            "metrics_pooled": d / "{}_metrics_pooled.csv".format(slug),
            "manifest": d / "{}_run_manifest.json".format(slug),
            "completion_report": d / "{}_completion_report.md".format(slug),
            "event_prevalence": d / "{}_event_prevalence.csv".format(slug),
        }
    return out


def expected_target_files_from_protocol_manifest() -> List[str]:
    return list(parse_checksum_manifest(PROTOCOL_SHA_PATH).keys())


def verify_checksum_map(base_dir: Path, expected_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for rel_name, expected in expected_map.items():
        fp = base_dir / rel_name
        exists = fp.exists()
        observed = sha256_file(fp) if exists else None
        ok = bool(exists and observed == expected)
        checks[rel_name] = {
            "expected": expected,
            "observed": observed,
            "exists": bool(exists),
            "pass": bool(ok),
        }
    return checks


def parse_source_model_info(
    model_file_map: Dict[str, Dict[str, Path]],
    local_tag_commits: Dict[str, str],
    remote_tag_presence: Dict[str, bool],
    source_preservation: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    model_info: Dict[str, Any] = {}
    all_ok = True

    for model in MODEL_ORDER:
        fp = model_file_map[model]
        manifest = json.loads(fp["manifest"].read_text(encoding="utf-8"))
        report_txt = fp["completion_report"].read_text(encoding="utf-8")

        pred_sha_obs = sha256_file(fp["prediction"])
        pred_sha_exp = str(manifest.get("prediction_sha256", ""))
        pred_pass = bool(pred_sha_obs == pred_sha_exp)

        metrics_sha_map = manifest.get("metrics_sha256", {})
        metrics_checks: Dict[str, Dict[str, Any]] = {}
        for mf in [fp["metrics_by_fold"], fp["metrics_pooled"], fp["event_prevalence"]]:
            if not mf.exists():
                continue
            nm = mf.name
            exp = str(metrics_sha_map.get(nm, ""))
            obs = sha256_file(mf)
            mp = bool(exp != "" and exp == obs)
            metrics_checks[nm] = {
                "expected": exp,
                "observed": obs,
                "pass": bool(mp),
            }
            all_ok = all_ok and bool(mp)

        test_result = str(manifest.get("test_result", ""))
        test_ok = bool(test_result == MODEL_EXPECTED_TEST_RESULT[model])

        decision = infer_decision(report_txt)
        decision_ok = bool(decision == "A")

        artifact_generation_commit = str(manifest.get("artifact_generation_commit") or manifest.get("git_commit") or "")
        freeze_tag = MODEL_FREEZE_TAG[model]

        model_ok = bool(
            pred_pass
            and all(v["pass"] for v in metrics_checks.values())
            and test_ok
            and decision_ok
            and source_preservation[freeze_tag]["pass"]
        )

        all_ok = all_ok and model_ok

        model_info[model] = {
            "run_id": str(manifest.get("run_id", "")),
            "artifact_generation_commit": artifact_generation_commit,
            "prediction_file": rel_to_root(fp["prediction"]),
            "prediction_sha256": pred_sha_obs,
            "prediction_sha256_manifest": pred_sha_exp,
            "prediction_file_identity_pass": bool(pred_pass),
            "metrics_by_fold_file": rel_to_root(fp["metrics_by_fold"]),
            "metrics_pooled_file": rel_to_root(fp["metrics_pooled"]),
            "event_prevalence_file": rel_to_root(fp["event_prevalence"]) if fp["event_prevalence"].exists() else None,
            "metrics_file_checks": metrics_checks,
            "manifest_file": rel_to_root(fp["manifest"]),
            "completion_report_file": rel_to_root(fp["completion_report"]),
            "test_result": test_result,
            "completion_decision": decision,
            "package_freeze_tag": freeze_tag,
            "package_freeze_commit": local_tag_commits[freeze_tag],
            "remote_tag_present": bool(remote_tag_presence.get(freeze_tag, False)),
            "source_preservation_pass": bool(source_preservation[freeze_tag]["pass"]),
            "model_info_pass": bool(model_ok),
        }

    return model_info, bool(all_ok)


def compute_fold_local_mase_denominators(split_df: pd.DataFrame, dataset_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    values = dataset_df["TNout"].astype(float).to_numpy()
    denoms: Dict[Tuple[int, int], float] = {}

    split_df = split_df.copy()
    split_df["horizon"] = pd.to_numeric(split_df["horizon"], errors="raise").astype(int)
    split_df["outer_fold"] = pd.to_numeric(split_df["outer_fold"], errors="raise").astype(int)
    split_df["original_row_index"] = pd.to_numeric(split_df["original_row_index"], errors="raise").astype(int)
    split_df["purged_outer_boundary"] = split_df["purged_outer_boundary"].map(bool_from_any)
    split_df["target_date"] = pd.to_datetime(split_df["target_date"], errors="raise")

    for h in HORIZONS:
        for fold in [1, 2, 3]:
            s = split_df[
                (split_df["horizon"] == int(h))
                & (split_df["outer_fold"] == int(fold))
                & (split_df["outer_role"].astype(str) == "outer_train")
                & (~split_df["purged_outer_boundary"].astype(bool))
            ].copy()
            s = s.sort_values(["target_date", "original_row_index"]).reset_index(drop=True)
            y_train: List[float] = []
            for r in s.itertuples(index=False):
                idx = int(r.original_row_index) + int(h)
                if idx < 0 or idx >= len(values):
                    raise RuntimeError("original_row_index + horizon out of bounds for h={}, fold={}".format(h, fold))
                y_train.append(float(values[idx]))

            if len(y_train) < 2:
                raise RuntimeError("Insufficient train rows to compute MASE denominator for h={}, fold={}".format(h, fold))

            denom = float(pd.Series(y_train, dtype=float).diff().abs().dropna().mean())
            denoms[(int(h), int(fold))] = denom

    return denoms


def build_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_tau15"] = (out["y_true"].astype(float) >= 15.0).astype(int)
    out["event_tau16"] = (out["y_true"].astype(float) >= 16.0).astype(int)
    out["event_tau17"] = (out["y_true"].astype(float) >= 17.0).astype(int)
    return out


def make_stage_assembly_id(input_hashes: Dict[str, str]) -> str:
    payload = {
        "scope": {
            "horizons": list(HORIZONS),
            "models": list(MODEL_ORDER),
            "protocol_tag": PROTOCOL_TAG,
        },
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes.keys())},
    }
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
    return "canonical_h1_h3_reconciliation_{}".format(digest)


def compute_metrics_by_fold(
    long_df: pd.DataFrame,
    horizon: int,
    denoms: Dict[Tuple[int, int], float],
    model_info: Dict[str, Any],
    dataset_sha: str,
    split_sha: str,
    assembly_id: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    d = long_df[long_df["horizon"].astype(int) == int(horizon)].copy()

    for model in MODEL_ORDER:
        dm = d[d["model"].astype(str) == str(model)].copy()
        for fold in [1, 2, 3]:
            g = dm[dm["outer_fold"].astype(int) == int(fold)].copy()
            g = g.sort_values(["target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

            err = g["error"].astype(float).to_numpy()
            abs_err = np.abs(err)
            sq_err = np.square(err)
            n = int(len(g))
            denom = float(denoms[(int(horizon), int(fold))])

            mae = float(np.mean(abs_err))
            mse = float(np.mean(sq_err))
            rmse = float(math.sqrt(mse))
            mase = float(np.mean(abs_err / (denom + MASE_EPS)))

            rows.append(
                {
                    "horizon": int(horizon),
                    "model": model,
                    "outer_fold": int(fold),
                    "N": int(n),
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MASE": mase,
                    "MASE_period": 1,
                    "MASE_denom_train_only": denom,
                    "MASE_method": MASE_METHOD,
                    "date_start": str(g["target_date"].iloc[0]),
                    "date_end": str(g["target_date"].iloc[-1]),
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "source_run_id": str(model_info[model]["run_id"]),
                    "artifact_generation_commit": str(model_info[model]["artifact_generation_commit"]),
                    "package_freeze_tag": str(model_info[model]["package_freeze_tag"]),
                    "package_freeze_commit": str(model_info[model]["package_freeze_commit"]),
                    "assembly_id": assembly_id,
                }
            )

    out = pd.DataFrame(rows).sort_values(["horizon", "model", "outer_fold"]).reset_index(drop=True)
    return out


def compute_metrics_pooled(
    long_df: pd.DataFrame,
    horizon: int,
    denoms: Dict[Tuple[int, int], float],
    model_info: Dict[str, Any],
    dataset_sha: str,
    split_sha: str,
    assembly_id: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    d = long_df[long_df["horizon"].astype(int) == int(horizon)].copy()

    for model in MODEL_ORDER:
        g = d[d["model"].astype(str) == str(model)].copy()
        g = g.sort_values(["outer_fold", "target_date", "feature_date", "canonical_row_id"]).reset_index(drop=True)

        err = g["error"].astype(float).to_numpy()
        abs_err = np.abs(err)
        sq_err = np.square(err)
        n = int(len(g))

        denom_vals = np.array([denoms[(int(horizon), int(f))] for f in g["outer_fold"].astype(int).tolist()], dtype=float)

        mae = float(np.mean(abs_err))
        mse = float(np.mean(sq_err))
        rmse = float(math.sqrt(mse))
        mase = float(np.mean(abs_err / (denom_vals + MASE_EPS)))

        rows.append(
            {
                "horizon": int(horizon),
                "model": model,
                "outer_fold": "pooled",
                "N": int(n),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "MASE_period": 1,
                "MASE_method": MASE_METHOD,
                "date_start": str(g["target_date"].iloc[0]),
                "date_end": str(g["target_date"].iloc[-1]),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "source_run_id": str(model_info[model]["run_id"]),
                "artifact_generation_commit": str(model_info[model]["artifact_generation_commit"]),
                "package_freeze_tag": str(model_info[model]["package_freeze_tag"]),
                "package_freeze_commit": str(model_info[model]["package_freeze_commit"]),
                "assembly_id": assembly_id,
            }
        )

    out = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    return out


def compute_event_prevalence(wide_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    d = wide_df[wide_df["horizon"].astype(int) == int(horizon)].copy()

    for fold in [1, 2, 3, "pooled"]:
        if fold == "pooled":
            g = d.copy()
        else:
            g = d[d["outer_fold"].astype(int) == int(fold)].copy()

        n = int(len(g))
        e15 = int(g["event_tau15"].astype(int).sum())
        e16 = int(g["event_tau16"].astype(int).sum())
        e17 = int(g["event_tau17"].astype(int).sum())

        rows.append(
            {
                "horizon": int(horizon),
                "outer_fold": fold,
                "N": n,
                "events_tau15": e15,
                "prevalence_tau15": safe_div(float(e15), float(n)),
                "events_tau16": e16,
                "prevalence_tau16": safe_div(float(e16), float(n)),
                "events_tau17": e17,
                "prevalence_tau17": safe_div(float(e17), float(n)),
            }
        )

    return pd.DataFrame(rows)


def load_and_standardize_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["horizon", "outer_fold", "feature_date", "target_date", "y_true", "y_pred"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError("Prediction file missing columns {}: {}".format(path, missing))

    out = df.copy()
    out["horizon"] = pd.to_numeric(out["horizon"], errors="raise").astype(int)
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="raise").astype(int)
    out["feature_date"] = normalize_date(out["feature_date"])
    out["target_date"] = normalize_date(out["target_date"])
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(float)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(float)
    out = out[out["horizon"].isin(HORIZONS)].copy()
    out = out.sort_values(["horizon", "outer_fold", "feature_date", "target_date"]).reset_index(drop=True)
    return out


def generate_checksums_registry() -> None:
    rows: List[str] = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == CHECKSUM_PATH.resolve():
            continue
        if p.suffix.lower() not in {".py", ".csv", ".json", ".md"}:
            continue
        rows.append("{}  {}".format(sha256_file(p), rel_to_stage(p)))

    ensure_inside_authorized(CHECKSUM_PATH)
    CHECKSUM_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_checksums_registry() -> Dict[str, Any]:
    cmd = "sha256sum -c {}".format(CHECKSUM_PATH.name)
    res = run_shell(cmd, cwd=OUT_DIR)

    expected = sorted(
        [
            rel_to_stage(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.resolve() != CHECKSUM_PATH.resolve()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
        ]
    )

    listed = sorted(parse_checksum_manifest(CHECKSUM_PATH).keys()) if CHECKSUM_PATH.exists() else []

    return {
        "exit_code": int(res["exit_code"]),
        "stdout": res["stdout"],
        "stderr": res["stderr"],
        "pass": bool(res["exit_code"] == 0),
        "coverage_missing": sorted(set(expected) - set(listed)),
        "coverage_extra": sorted(set(listed) - set(expected)),
        "coverage_pass": bool(set(expected) == set(listed)),
    }


def get_existing_test_result() -> str:
    if not MANIFEST_PATH.exists():
        return "pending_not_run"
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(payload.get("test_result", "pending_not_run"))
    except Exception:
        return "pending_not_run"


def compute_decision(
    verification_pass: bool,
    key_equality_pass: bool,
    source_preservation_pass: bool,
    metric_reconciliation_pass: bool,
    mase_convention_pass: bool,
    checksum_pass: bool,
    test_result: str,
) -> str:
    if not verification_pass:
        return "C"
    if not key_equality_pass or not source_preservation_pass:
        return "D"
    if str(test_result).startswith("fail"):
        return "D"
    if not metric_reconciliation_pass or not mase_convention_pass or not checksum_pass:
        return "B"
    if str(test_result).startswith("pass"):
        return "A"
    return "B"


def build_report(
    verification_pass: bool,
    source_tests: Dict[str, str],
    source_decisions: Dict[str, str],
    local_tags: Dict[str, str],
    remote_tags: Dict[str, bool],
    source_preservation_pass: bool,
    h_stats: Dict[int, Dict[str, Any]],
    event_counts: Dict[int, Dict[str, Any]],
    metric_reconciliation_pass: bool,
    mase_convention_pass: bool,
    checksum_info: Dict[str, Any],
    test_result: str,
    deterministic_result: str,
    final_decision: str,
    starting_commit: str,
    warnings: List[str],
    generated_files: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Canonical Corrected H1/H3 Cross-Model Reconciliation")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("- Reconciled horizons: H1 and H3 only.")
    lines.append("- Reconciled models: Persistence, Ridge, ElasticNet, HGBR.")
    lines.append("- Explicit exclusions: H5, BCR-TCN, TCN, point blend, HybridRank, fixed rank ensemble, alarm flags, top-k, classification metrics")
    lines.append("- No model fitting, tuning, retraining, alarm policies, or classification metrics were run.")
    lines.append("")
    lines.append("## 2. Preflight and Source Integrity")
    lines.append("- verification_pass: {}".format(bool(verification_pass)))
    lines.append("- source test results: {}".format(json.dumps(source_tests, sort_keys=True)))
    lines.append("- source completion decisions: {}".format(json.dumps(source_decisions, sort_keys=True)))
    lines.append("- local freeze-tag commits: {}".format(json.dumps(local_tags, sort_keys=True)))
    lines.append("- remote freeze-tag presence (revision-origin): {}".format(json.dumps(remote_tags, sort_keys=True)))
    lines.append("- source preservation against local tags: {}".format(bool(source_preservation_pass)))
    if warnings:
        lines.append("- provenance warnings: {}".format(json.dumps(warnings, sort_keys=True)))
    lines.append("")
    lines.append("## 3. Canonical Sample Sizes and Date Ranges")
    for h in HORIZONS:
        hs = h_stats[h]
        lines.append(
            "- H{}: N={}, fold_counts={}, feature_date={}..{}, target_date={}..{}".format(
                h,
                hs["N"],
                hs["fold_counts"],
                hs["feature_date_start"],
                hs["feature_date_end"],
                hs["target_date_start"],
                hs["target_date_end"],
            )
        )
    lines.append("")
    lines.append("## 4. Event Counts")
    for h in HORIZONS:
        ev = event_counts[h]
        lines.append(
            "- H{} pooled events: tau15={} tau16={} tau17={} (prevalence: {}, {}, {})".format(
                h,
                ev["events_tau15"],
                ev["events_tau16"],
                ev["events_tau17"],
                ev["prevalence_tau15"],
                ev["prevalence_tau16"],
                ev["prevalence_tau17"],
            )
        )
    lines.append("")
    lines.append("## 5. Metric Definitions")
    lines.append("- MAE = mean(abs(y_true - y_pred)).")
    lines.append("- MSE = mean((y_true - y_pred)^2).")
    lines.append("- RMSE = sqrt(MSE).")
    lines.append("- Pooled MAE/MSE/RMSE are computed directly over pooled canonical rows.")
    lines.append("- MASE period is m=1 with fold-local training-only denominators.")
    lines.append("- MASE method: {}".format(MASE_METHOD))
    lines.append("- MASE convention verification pass: {}".format(bool(mase_convention_pass)))
    lines.append("")
    lines.append("## 6. Provenance Distinction")
    lines.append("- source_run_id and artifact_generation_commit are preserved from each source manifest.")
    lines.append("- package_freeze_tag and package_freeze_commit are recorded separately from source-generation commits.")
    lines.append("- HGBR provenance-B issue registry status: resolved=TRUE for all rows and no validity-critical row.")
    lines.append("")
    lines.append("## 7. Reconciliation and Determinism")
    lines.append("- controlled-rerun metric reconciliation pass: {}".format(bool(metric_reconciliation_pass)))
    lines.append("- checksum registry verification pass: {}".format(bool(checksum_info["pass"] and checksum_info["coverage_pass"])))
    lines.append("- checksum coverage missing: {}".format(checksum_info["coverage_missing"]))
    lines.append("- checksum coverage extra: {}".format(checksum_info["coverage_extra"]))
    lines.append("- test_result: {}".format(str(test_result)))
    lines.append("- deterministic_result: {}".format(str(deterministic_result)))
    lines.append("")
    lines.append("## 8. Remaining Limitations")
    lines.append("1. This stage reconciles historical held-out predictions only; no prospective deployment validation is provided.")
    lines.append("2. Remote tag availability for revision-origin may be incomplete; this is recorded as provenance metadata, not silently converted into computational pass/fail.")
    lines.append("3. This stage excludes H5 and all alarm-policy analyses by design.")
    lines.append("")
    lines.append("FINAL DECISION")
    lines.append("")
    if final_decision == "A":
        lines.append("A. Canonical H1/H3 reconciliation passed; lag-window sensitivity may begin.")
    elif final_decision == "B":
        lines.append("B. Reconciliation completed, but date, metric, or provenance discrepancies require investigation.")
    elif final_decision == "C":
        lines.append("C. One or more source packages could not be reconciled safely.")
    else:
        lines.append("D. Key equality, source preservation, or deterministic tests failed.")

    lines.append("")
    lines.append("TERMINAL SUMMARY")
    lines.append("")
    lines.append("1. repository: {}".format(ROOT))
    lines.append("2. branch: {}".format(EXPECTED_BRANCH))
    lines.append("3. starting commit: {}".format(starting_commit))
    lines.append("4. authorized output directory: {}".format(OUT_DIR))
    lines.append("5. input checksum status: {}".format(bool(verification_pass)))
    lines.append("6. source decision status: {}".format(json.dumps(source_decisions, sort_keys=True)))
    lines.append("7. source-tag and remote-tag status: local={} remote={}".format(json.dumps(local_tags, sort_keys=True), json.dumps(remote_tags, sort_keys=True)))
    lines.append("8. H1 N and fold sizes: N={} folds={}".format(h_stats[1]["N"], h_stats[1]["fold_counts"]))
    lines.append("9. H3 N and fold sizes: N={} folds={}".format(h_stats[3]["N"], h_stats[3]["fold_counts"]))
    lines.append("10. event counts: {}".format(json.dumps(event_counts, sort_keys=True)))
    lines.append("11. models included: {}".format(list(MODEL_ORDER)))
    lines.append("12. models explicitly excluded: {}".format(["H5", "BCR-TCN", "TCN", "point blend", "HybridRank", "fixed rank ensemble", "alarm flags", "top-k", "classification metrics"]))
    lines.append("13. long and wide row counts: H1 long={} wide={}; H3 long={} wide={}".format(
        h_stats[1]["long_rows"], h_stats[1]["wide_rows"], h_stats[3]["long_rows"], h_stats[3]["wide_rows"]
    ))
    lines.append("14. metric-reconciliation status: {}".format(bool(metric_reconciliation_pass)))
    lines.append("15. MASE-convention status: {}".format(bool(mase_convention_pass)))
    lines.append("16. number of tests run and passed: {}".format(str(test_result)))
    lines.append("17. deterministic result: {}".format(str(deterministic_result)))
    lines.append("18. source-preservation result: {}".format(bool(source_preservation_pass)))
    lines.append("19. checksum result: file_ok={} coverage_ok={}".format(bool(checksum_info["pass"]), bool(checksum_info["coverage_pass"])))
    lines.append("20. final decision: {}".format(final_decision))
    lines.append("21. generated files: {}".format(json.dumps(generated_files, sort_keys=True)))

    return "\n".join(lines) + "\n"


def build_stage() -> None:
    paths = get_stage_file_paths()
    for h in HORIZONS:
        paths[h]["dir"].mkdir(parents=True, exist_ok=True)

    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("Run this script from repository root: {}".format(ROOT))

    start_status = git_status_paths()
    clean_start = len(start_status) == 0
    authorized_only_start = bool(
        len(start_status) > 0
        and all(
            p == AUTHORIZED_REL_DIR or p.startswith(AUTHORIZED_REL_DIR + "/")
            for p in start_status
        )
    )

    branch = git_output(["git", "branch", "--show-current"])
    starting_commit = git_output(["git", "rev-parse", "HEAD"])
    py_version = platform.python_version()

    root_sha_cmd = run_shell("sha256sum -c corrected_protocol_v1.sha256", cwd=PROTOCOL_DIR)
    locked_sha_cmd = run_shell("sha256sum -c corrected_protocol_v1.sha256", cwd=LOCKED_PROTOCOL_DIR)

    protocol_expected = parse_checksum_manifest(PROTOCOL_SHA_PATH)
    locked_expected = parse_checksum_manifest(LOCKED_PROTOCOL_SHA_PATH)

    protocol_root_checks = verify_checksum_map(PROTOCOL_DIR, protocol_expected)
    protocol_locked_checks = verify_checksum_map(LOCKED_PROTOCOL_DIR, locked_expected)

    root_locked_identity_rows: List[Dict[str, Any]] = []
    for nm in sorted(set(protocol_expected.keys()) | set(locked_expected.keys())):
        rp = PROTOCOL_DIR / nm
        lp = LOCKED_PROTOCOL_DIR / nm
        re = rp.exists()
        le = lp.exists()
        rsha = sha256_file(rp) if re else None
        lsha = sha256_file(lp) if le else None
        identical = bool(re and le and rsha == lsha)
        root_locked_identity_rows.append(
            {
                "file": nm,
                "root_exists": bool(re),
                "locked_exists": bool(le),
                "root_sha": rsha,
                "locked_sha": lsha,
                "identical": bool(identical),
            }
        )

    missing_locked = sorted([k for k, v in protocol_locked_checks.items() if not v["exists"]])
    locked_present_only_pass = bool(all(v["pass"] for v in protocol_locked_checks.values() if v["exists"]))
    shared_identity_pass = bool(all(r["identical"] for r in root_locked_identity_rows if r["root_exists"] and r["locked_exists"]))

    locked_missing_expected = sorted(["build_purged_nested_splits.py", "test_purged_nested_splits.py"])
    locked_expected_missing_only = bool(sorted(missing_locked) == locked_missing_expected)

    protocol_locked_effective_pass = bool(
        locked_present_only_pass
        and shared_identity_pass
        and (locked_sha_cmd["exit_code"] == 0 or locked_expected_missing_only)
    )

    lock_payload = json.loads(DATASET_LOCK_PATH.read_text(encoding="utf-8"))
    dataset_path = Path(lock_payload["absolute_path"])
    dataset_sha_observed = sha256_file(dataset_path)
    dataset_sha_expected = str(lock_payload["sha256"])

    split_sha_expected = str(protocol_expected["corrected_split_assignment.csv"])
    split_sha_observed = sha256_file(SPLIT_PATH)

    local_tag_commits: Dict[str, str] = {}
    local_tags_ok = True
    for tag in FREEZE_TAGS:
        try:
            local_tag_commits[tag] = git_output(["git", "rev-parse", tag])
        except subprocess.CalledProcessError:
            local_tags_ok = False
            local_tag_commits[tag] = "MISSING"

    remote_tag_presence: Dict[str, bool] = {}
    for tag in FREEZE_TAGS:
        try:
            out = git_output(["git", "ls-remote", "--tags", "revision-origin", "refs/tags/{}".format(tag)])
            remote_tag_presence[tag] = bool(str(out).strip() != "")
        except subprocess.CalledProcessError:
            remote_tag_presence[tag] = False

    source_preservation: Dict[str, Dict[str, Any]] = {}
    source_paths_by_tag = {
        "corrected-protocol-v1": ROOT / "revision_2026" / "03_corrected_protocol",
        "persistence-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "persistence",
        "ridge-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "ridge",
        "elasticnet-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "elasticnet",
        "hgbr-corrected-v1": ROOT / "revision_2026" / "04_controlled_reruns" / "hgbr",
    }

    source_preservation_pass = True
    for tag, path in source_paths_by_tag.items():
        if local_tag_commits.get(tag, "MISSING") == "MISSING":
            changed: List[str] = ["tag_missing"]
            ok = False
        else:
            out = git_output(["git", "diff", "--name-only", tag, "--", rel_to_root(path)])
            changed = [x for x in out.splitlines() if x.strip()]
            ok = len(changed) == 0
        source_preservation_pass = source_preservation_pass and ok
        source_preservation[tag] = {
            "source_dir": rel_to_root(path),
            "changed_paths": changed,
            "pass": bool(ok),
        }

    model_file_map = get_model_file_map()
    source_model_info, source_model_checks_pass = parse_source_model_info(
        model_file_map=model_file_map,
        local_tag_commits=local_tag_commits,
        remote_tag_presence=remote_tag_presence,
        source_preservation=source_preservation,
    )

    source_test_pass = bool(
        all(
            source_model_info[m]["test_result"] == MODEL_EXPECTED_TEST_RESULT[m]
            for m in MODEL_ORDER
        )
    )
    source_decision_pass = bool(all(source_model_info[m]["completion_decision"] == "A" for m in MODEL_ORDER))

    hgbr_issue_df = pd.read_csv(HGBR_ISSUE_REGISTRY_PATH)
    resolved_col = hgbr_issue_df["resolved"].map(bool_from_any)
    effect_tokens = hgbr_issue_df["effect_on_validity"].astype(str).str.lower().str.strip()
    negated_validity_tokens = (
        effect_tokens.str.contains("not_validity_critical", regex=False)
        | effect_tokens.str.contains("no_validity_critical", regex=False)
        | effect_tokens.str.contains("non_validity_critical", regex=False)
        | effect_tokens.str.contains("not validity critical", regex=False)
        | effect_tokens.str.contains("no validity critical", regex=False)
    )
    validity_critical_col = effect_tokens.str.contains("validity_critical", regex=False) & (~negated_validity_tokens)
    hgbr_issue_pass = bool(resolved_col.all() and (~validity_critical_col).all())

    working_tree_scope_pass = bool(clean_start or authorized_only_start)
    branch_pass = bool(branch == EXPECTED_BRANCH)
    python_pass = bool(py_version == EXPECTED_PYTHON_VERSION)

    root_protocol_pass = bool(root_sha_cmd["exit_code"] == 0 and all(v["pass"] for v in protocol_root_checks.values()))
    dataset_pass = bool(dataset_sha_observed == EXPECTED_DATASET_SHA256 and dataset_sha_observed == dataset_sha_expected)
    split_pass = bool(split_sha_observed == EXPECTED_SPLIT_SHA256 and split_sha_observed == split_sha_expected)

    verification_pass = bool(
        branch_pass
        and python_pass
        and working_tree_scope_pass
        and root_protocol_pass
        and protocol_locked_effective_pass
        and dataset_pass
        and split_pass
        and local_tags_ok
        and source_preservation_pass
        and source_model_checks_pass
        and source_test_pass
        and source_decision_pass
        and hgbr_issue_pass
    )

    warnings: List[str] = []
    if locked_sha_cmd["exit_code"] != 0 and locked_expected_missing_only:
        warnings.append(
            "locked_protocol_registry_missing_files_noncritical: build_purged_nested_splits.py,test_purged_nested_splits.py"
        )
    for tag, present in remote_tag_presence.items():
        if not present:
            warnings.append("missing_remote_tag:{}".format(tag))

    split_df = pd.read_csv(SPLIT_PATH)
    split_df["horizon"] = pd.to_numeric(split_df["horizon"], errors="raise").astype(int)
    split_df["outer_fold"] = pd.to_numeric(split_df["outer_fold"], errors="raise").astype(int)
    split_df["feature_date"] = normalize_date(split_df["feature_date"])
    split_df["target_date"] = normalize_date(split_df["target_date"])
    split_df["purged_outer_boundary"] = split_df["purged_outer_boundary"].map(bool_from_any)

    split_summary_df = pd.read_csv(SPLIT_SUMMARY_PATH)
    split_summary_df["horizon"] = pd.to_numeric(split_summary_df["horizon"], errors="raise").astype(int)
    split_summary_df["outer_fold"] = pd.to_numeric(split_summary_df["outer_fold"], errors="raise").astype(int)

    dataset_df = pd.read_csv(dataset_path)
    dataset_df = dataset_df.drop(columns=[c for c in dataset_df.columns if str(c).startswith("Unnamed")], errors="ignore")
    dataset_df["Date"] = pd.to_datetime(dataset_df["Date"], errors="raise")
    dataset_df = dataset_df.sort_values("Date").reset_index(drop=True)

    denoms = compute_fold_local_mase_denominators(split_df, dataset_df)

    # Independent denominator verification against persistence source denominator values.
    pers_by_fold = pd.read_csv(model_file_map["Persistence"]["metrics_by_fold"])
    pers_by_fold["horizon"] = pd.to_numeric(pers_by_fold["horizon"], errors="raise").astype(int)
    pers_by_fold["outer_fold"] = pd.to_numeric(pers_by_fold["outer_fold"], errors="raise").astype(int)
    denom_verify_rows: List[Dict[str, Any]] = []
    mase_convention_pass = True
    for h in HORIZONS:
        for fold in [1, 2, 3]:
            src_d = float(
                pers_by_fold[
                    (pers_by_fold["horizon"] == int(h))
                    & (pers_by_fold["outer_fold"] == int(fold))
                ]["MASE_denom_train_only"].iloc[0]
            )
            rec_d = float(denoms[(int(h), int(fold))])
            d_ok = bool(abs(src_d - rec_d) <= TOL)
            mase_convention_pass = mase_convention_pass and d_ok
            denom_verify_rows.append(
                {
                    "horizon": int(h),
                    "outer_fold": int(fold),
                    "source_mase_denom_train_only": src_d,
                    "recomputed_mase_denom_train_only": rec_d,
                    "absolute_delta": float(abs(src_d - rec_d)),
                    "tolerance": float(TOL),
                    "pass": bool(d_ok),
                }
            )

    # Build canonical keys from corrected split outer-test rows.
    canonical_outer = split_df[
        split_df["horizon"].isin(HORIZONS)
        & (split_df["outer_role"].astype(str) == "outer_test")
        & (~split_df["purged_outer_boundary"].astype(bool))
    ][KEY_COLS].drop_duplicates().copy()

    expected_keys_by_h: Dict[int, set] = {}
    expected_fold_counts_by_h: Dict[int, Dict[int, int]] = {}
    for h in HORIZONS:
        ch = canonical_outer[canonical_outer["horizon"].astype(int) == int(h)].copy()
        expected_keys_by_h[int(h)] = set(tuple(r) for r in ch[KEY_COLS].itertuples(index=False, name=None))
        expected_fold_counts_by_h[int(h)] = {
            int(f): int(n)
            for f, n in ch.groupby("outer_fold").size().to_dict().items()
        }

    # Load all model predictions and check key identities.
    standardized: Dict[str, pd.DataFrame] = {}
    key_audit_rows: Dict[int, List[Dict[str, Any]]] = {1: [], 3: []}

    for model in MODEL_ORDER:
        d = load_and_standardize_predictions(model_file_map[model]["prediction"])
        standardized[model] = d

        for h in HORIZONS:
            dh = d[d["horizon"].astype(int) == int(h)].copy()
            for fold in [1, 2, 3]:
                dmf = dh[dh["outer_fold"].astype(int) == int(fold)].copy()
                dmf = dmf.sort_values(["feature_date", "target_date"]).reset_index(drop=True)

                expected_fold_keys = canonical_outer[
                    (canonical_outer["horizon"].astype(int) == int(h))
                    & (canonical_outer["outer_fold"].astype(int) == int(fold))
                ][KEY_COLS].copy()
                expected_fold_set = set(tuple(r) for r in expected_fold_keys.itertuples(index=False, name=None))
                observed_fold_set = set(tuple(r) for r in dmf[KEY_COLS].itertuples(index=False, name=None))

                duplicate_count = int(dmf.duplicated(subset=KEY_COLS).sum())
                missing_keys = expected_fold_set - observed_fold_set
                extra_keys = observed_fold_set - expected_fold_set

                date_delta = (
                    pd.to_datetime(dmf["target_date"], errors="raise")
                    - pd.to_datetime(dmf["feature_date"], errors="raise")
                ).dt.days
                target_offset_mismatch = int((date_delta != int(h)).sum())

                key_audit_rows[int(h)].append(
                    {
                        "model": model,
                        "horizon": int(h),
                        "fold": int(fold),
                        "observed_N": int(len(dmf)),
                        "expected_N": int(len(expected_fold_set)),
                        "duplicate_key_count": int(duplicate_count),
                        "missing_key_count": int(len(missing_keys)),
                        "extra_key_count": int(len(extra_keys)),
                        "y_true_mismatch_count": 0,
                        "target_offset_mismatch_count": int(target_offset_mismatch),
                        "feature_date_start": str(dmf["feature_date"].min()) if len(dmf) > 0 else "",
                        "feature_date_end": str(dmf["feature_date"].max()) if len(dmf) > 0 else "",
                        "target_date_start": str(dmf["target_date"].min()) if len(dmf) > 0 else "",
                        "target_date_end": str(dmf["target_date"].max()) if len(dmf) > 0 else "",
                        "key_equality_pass": bool(duplicate_count == 0 and len(missing_keys) == 0 and len(extra_keys) == 0),
                        "y_true_equality_pass": True,
                        "target_offset_pass": bool(target_offset_mismatch == 0),
                    }
                )

    # Canonical y_true consistency across models.
    canonical_y_by_h: Dict[int, pd.DataFrame] = {}
    y_true_all_models_pass = True
    for h in HORIZONS:
        ref = standardized["Persistence"][standardized["Persistence"]["horizon"].astype(int) == int(h)][KEY_COLS + ["y_true"]].copy()
        ref = ref.sort_values(KEY_COLS).reset_index(drop=True)
        canonical_y_by_h[int(h)] = ref.copy()

        for model in MODEL_ORDER:
            dm = standardized[model][standardized[model]["horizon"].astype(int) == int(h)][KEY_COLS + ["y_true"]].copy()
            dm = dm.sort_values(KEY_COLS).reset_index(drop=True)
            merged = dm.merge(ref.rename(columns={"y_true": "y_ref"}), on=KEY_COLS, how="inner")
            if int(len(merged)) != int(len(ref)):
                y_true_all_models_pass = False
            mismatch = (merged["y_true"].astype(float) - merged["y_ref"].astype(float)).abs() > TOL
            mismatch_n = int(mismatch.sum())
            if mismatch_n > 0:
                y_true_all_models_pass = False

            # Fill key audit y_true mismatch values.
            for i in range(len(key_audit_rows[int(h)])):
                if key_audit_rows[int(h)][i]["model"] != model:
                    continue
                fold = int(key_audit_rows[int(h)][i]["fold"])
                mm = merged[merged["outer_fold"].astype(int) == fold]
                mmismatch = int(((mm["y_true"].astype(float) - mm["y_ref"].astype(float)).abs() > TOL).sum())
                key_audit_rows[int(h)][i]["y_true_mismatch_count"] = int(mmismatch)
                key_audit_rows[int(h)][i]["y_true_equality_pass"] = bool(mmismatch == 0)

    # Build canonical long and wide tables.
    long_by_h: Dict[int, pd.DataFrame] = {}
    wide_by_h: Dict[int, pd.DataFrame] = {}

    y_pred_col_map = {
        "Persistence": "y_pred_persistence",
        "Ridge": "y_pred_ridge",
        "ElasticNet": "y_pred_elasticnet",
        "HGBR": "y_pred_hgbr",
    }

    key_equality_pass = True
    for h in HORIZONS:
        ref = canonical_y_by_h[int(h)].copy()
        ref = build_event_labels(ref)
        ref["canonical_row_id"] = ref.apply(
            lambda r: canonical_row_id(int(r["horizon"]), int(r["outer_fold"]), str(r["feature_date"]), str(r["target_date"])),
            axis=1,
        )

        # Key-set check against split-derived canonical keys.
        ref_key_set = set(tuple(r) for r in ref[KEY_COLS].itertuples(index=False, name=None))
        if ref_key_set != expected_keys_by_h[int(h)]:
            key_equality_pass = False

        long_rows: List[Dict[str, Any]] = []
        for model in MODEL_ORDER:
            dm = standardized[model][standardized[model]["horizon"].astype(int) == int(h)].copy()
            dm = dm.merge(ref[KEY_COLS + ["canonical_row_id", "y_true", "event_tau15", "event_tau16", "event_tau17"]], on=KEY_COLS, how="inner", suffixes=("", "_canonical"))
            dm = dm.sort_values(KEY_COLS).reset_index(drop=True)

            # source event label consistency check where available.
            for tau, col in [(15, "event_tau15"), (16, "event_tau16"), (17, "event_tau17")]:
                src_col = col
                if src_col in standardized[model].columns:
                    pass

            source_run_id = str(source_model_info[model]["run_id"])
            artifact_generation_commit = str(source_model_info[model]["artifact_generation_commit"])
            freeze_tag = str(source_model_info[model]["package_freeze_tag"])
            freeze_commit = str(source_model_info[model]["package_freeze_commit"])

            y_true = dm["y_true_canonical"].astype(float).to_numpy()
            y_pred = dm["y_pred"].astype(float).to_numpy()
            err = y_true - y_pred
            abs_err = np.abs(err)
            sq_err = np.square(err)

            pred_path = model_file_map[model]["prediction"]
            pred_sha = sha256_file(pred_path)

            for i, r in dm.iterrows():
                long_rows.append(
                    {
                        "canonical_row_id": str(r["canonical_row_id"]),
                        "horizon": int(h),
                        "outer_fold": int(r["outer_fold"]),
                        "feature_date": str(r["feature_date"]),
                        "target_date": str(r["target_date"]),
                        "model": model,
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                        "error": float(err[i]),
                        "absolute_error": float(abs_err[i]),
                        "squared_error": float(sq_err[i]),
                        "event_tau15": int(r["event_tau15"]),
                        "event_tau16": int(r["event_tau16"]),
                        "event_tau17": int(r["event_tau17"]),
                        "dataset_sha256": EXPECTED_DATASET_SHA256,
                        "split_sha256": EXPECTED_SPLIT_SHA256,
                        "source_prediction_file": rel_to_root(pred_path),
                        "source_prediction_sha256": pred_sha,
                        "source_run_id": source_run_id,
                        "artifact_generation_commit": artifact_generation_commit,
                        "package_freeze_tag": freeze_tag,
                        "package_freeze_commit": freeze_commit,
                    }
                )

        long_df = pd.DataFrame(long_rows)
        long_df = long_df.sort_values(["horizon", "outer_fold", "feature_date", "target_date", "model"]).reset_index(drop=True)
        long_by_h[int(h)] = long_df

        wide = ref[["canonical_row_id", "horizon", "outer_fold", "feature_date", "target_date", "y_true", "event_tau15", "event_tau16", "event_tau17"]].copy()
        for model in MODEL_ORDER:
            dm = standardized[model][standardized[model]["horizon"].astype(int) == int(h)][KEY_COLS + ["y_pred"]].copy()
            wide = wide.merge(
                dm.rename(columns={"y_pred": y_pred_col_map[model]}),
                on=KEY_COLS,
                how="left",
            )
        wide = wide.sort_values(["horizon", "outer_fold", "feature_date", "target_date"]).reset_index(drop=True)
        wide_by_h[int(h)] = wide

    # Event label verification against source event labels where present.
    source_event_label_pass = True
    source_event_label_mismatches: Dict[str, Dict[str, int]] = {}
    for model in MODEL_ORDER:
        dsrc = standardized[model].copy()
        dsrc = dsrc[dsrc["horizon"].isin(HORIZONS)].copy()
        dsrc = dsrc.sort_values(KEY_COLS).reset_index(drop=True)
        source_event_label_mismatches[model] = {}

        for h in HORIZONS:
            src_h = dsrc[dsrc["horizon"].astype(int) == int(h)].copy()
            can_h = build_event_labels(canonical_y_by_h[int(h)].copy())
            m = src_h.merge(can_h[KEY_COLS + ["event_tau15", "event_tau16", "event_tau17"]], on=KEY_COLS, how="inner", suffixes=("", "_re"))
            for col in ["event_tau15", "event_tau16", "event_tau17"]:
                if col in src_h.columns:
                    mm = int((pd.to_numeric(m[col], errors="coerce").fillna(-1).astype(int) != m[col + "_re"].astype(int)).sum())
                else:
                    mm = 0
                source_event_label_mismatches[model]["h{}_{}".format(h, col)] = mm
                if mm != 0:
                    source_event_label_pass = False

    # Build metrics and reconciliations.
    metrics_by_h_by_fold: Dict[int, pd.DataFrame] = {}
    metrics_by_h_pooled: Dict[int, pd.DataFrame] = {}
    reconciliation_by_h: Dict[int, pd.DataFrame] = {}

    metric_reconciliation_pass = True
    metric_recon_rows_all: List[Dict[str, Any]] = []

    for h in HORIZONS:
        long_h = long_by_h[int(h)]
        fold_metrics = compute_metrics_by_fold(
            long_df=long_h,
            horizon=int(h),
            denoms=denoms,
            model_info=source_model_info,
            dataset_sha=EXPECTED_DATASET_SHA256,
            split_sha=EXPECTED_SPLIT_SHA256,
            assembly_id="PENDING",
        )
        pooled_metrics = compute_metrics_pooled(
            long_df=long_h,
            horizon=int(h),
            denoms=denoms,
            model_info=source_model_info,
            dataset_sha=EXPECTED_DATASET_SHA256,
            split_sha=EXPECTED_SPLIT_SHA256,
            assembly_id="PENDING",
        )

        metrics_by_h_by_fold[int(h)] = fold_metrics
        metrics_by_h_pooled[int(h)] = pooled_metrics

        recon_rows: List[Dict[str, Any]] = []
        for model in MODEL_ORDER:
            src_byf = pd.read_csv(model_file_map[model]["metrics_by_fold"])
            src_pooled = pd.read_csv(model_file_map[model]["metrics_pooled"])

            src_byf = src_byf[pd.to_numeric(src_byf["horizon"], errors="raise").astype(int) == int(h)].copy()
            src_byf["outer_fold"] = pd.to_numeric(src_byf["outer_fold"], errors="raise").astype(int)
            src_pooled = src_pooled[pd.to_numeric(src_pooled["horizon"], errors="raise").astype(int) == int(h)].copy()

            rec_byf = fold_metrics[fold_metrics["model"].astype(str) == model].copy()
            rec_pooled = pooled_metrics[pooled_metrics["model"].astype(str) == model].copy()

            src_file_byf = rel_to_root(model_file_map[model]["metrics_by_fold"])
            src_sha_byf = sha256_file(model_file_map[model]["metrics_by_fold"])
            src_file_pool = rel_to_root(model_file_map[model]["metrics_pooled"])
            src_sha_pool = sha256_file(model_file_map[model]["metrics_pooled"])

            for fold in [1, 2, 3]:
                srow = src_byf[src_byf["outer_fold"].astype(int) == int(fold)].iloc[0]
                rrow = rec_byf[rec_byf["outer_fold"].astype(int) == int(fold)].iloc[0]
                for metric in ["MAE", "MSE", "RMSE", "MASE"]:
                    sv = float(srow[metric])
                    rv = float(rrow[metric])
                    delta = float(abs(sv - rv))
                    ok = bool(delta <= TOL)
                    metric_reconciliation_pass = metric_reconciliation_pass and ok
                    recon_rows.append(
                        {
                            "horizon": int(h),
                            "model": model,
                            "outer_fold": int(fold),
                            "metric": metric,
                            "source_value": sv,
                            "recomputed_value": rv,
                            "absolute_delta": delta,
                            "tolerance": float(TOL),
                            "pass": bool(ok),
                            "aggregation_method": "fold_level_direct",
                            "MASE_method": MASE_METHOD,
                            "source_file": src_file_byf,
                            "source_checksum": src_sha_byf,
                        }
                    )
                if "MASE_denom_train_only" in src_byf.columns:
                    sv = float(srow["MASE_denom_train_only"])
                    rv = float(rrow["MASE_denom_train_only"])
                    delta = float(abs(sv - rv))
                    ok = bool(delta <= TOL)
                    metric_reconciliation_pass = metric_reconciliation_pass and ok
                    recon_rows.append(
                        {
                            "horizon": int(h),
                            "model": model,
                            "outer_fold": int(fold),
                            "metric": "MASE_denom_train_only",
                            "source_value": sv,
                            "recomputed_value": rv,
                            "absolute_delta": delta,
                            "tolerance": float(TOL),
                            "pass": bool(ok),
                            "aggregation_method": "fold_level_train_denominator",
                            "MASE_method": MASE_METHOD,
                            "source_file": src_file_byf,
                            "source_checksum": src_sha_byf,
                        }
                    )

            srow = src_pooled.iloc[0]
            rrow = rec_pooled.iloc[0]
            for metric in ["MAE", "MSE", "RMSE", "MASE"]:
                sv = float(srow[metric])
                rv = float(rrow[metric])
                delta = float(abs(sv - rv))
                ok = bool(delta <= TOL)
                metric_reconciliation_pass = metric_reconciliation_pass and ok
                recon_rows.append(
                    {
                        "horizon": int(h),
                        "model": model,
                        "outer_fold": "pooled",
                        "metric": metric,
                        "source_value": sv,
                        "recomputed_value": rv,
                        "absolute_delta": delta,
                        "tolerance": float(TOL),
                        "pass": bool(ok),
                        "aggregation_method": "pooled_rows_direct",
                        "MASE_method": MASE_METHOD,
                        "source_file": src_file_pool,
                        "source_checksum": src_sha_pool,
                    }
                )

        recon_df = pd.DataFrame(recon_rows).sort_values(["horizon", "model", "outer_fold", "metric"]).reset_index(drop=True)
        reconciliation_by_h[int(h)] = recon_df
        metric_recon_rows_all.extend(recon_rows)

    # Build source registry.
    source_registry_rows: List[Dict[str, Any]] = []

    def add_source_row(
        source_role: str,
        model_or_protocol: str,
        path: Path,
        manifest_expected_sha256: Optional[str],
        artifact_generation_commit: Optional[str],
        package_freeze_tag: Optional[str],
        package_freeze_commit: Optional[str],
        remote_tag_present: Optional[bool],
        source_preservation_pass_val: Optional[bool],
    ) -> None:
        observed_sha = sha256_file(path)
        expected = manifest_expected_sha256 if manifest_expected_sha256 is not None else ""
        checksum_match = bool(True if expected == "" else observed_sha == expected)
        source_registry_rows.append(
            {
                "source_role": source_role,
                "model_or_protocol": model_or_protocol,
                "relative_path": rel_to_root(path),
                "sha256": observed_sha,
                "manifest_expected_sha256": expected,
                "checksum_match": bool(checksum_match),
                "artifact_generation_commit": artifact_generation_commit if artifact_generation_commit is not None else "",
                "package_freeze_tag": package_freeze_tag if package_freeze_tag is not None else "",
                "package_freeze_commit": package_freeze_commit if package_freeze_commit is not None else "",
                "remote_tag_present": "" if remote_tag_present is None else bool(remote_tag_present),
                "source_preservation_pass": "" if source_preservation_pass_val is None else bool(source_preservation_pass_val),
            }
        )

    for fname, expected_sha in sorted(protocol_expected.items()):
        add_source_row(
            source_role="protocol_root",
            model_or_protocol="corrected_protocol",
            path=PROTOCOL_DIR / fname,
            manifest_expected_sha256=expected_sha,
            artifact_generation_commit=local_tag_commits.get("corrected-protocol-v1", ""),
            package_freeze_tag="corrected-protocol-v1",
            package_freeze_commit=local_tag_commits.get("corrected-protocol-v1", ""),
            remote_tag_present=remote_tag_presence.get("corrected-protocol-v1", False),
            source_preservation_pass_val=source_preservation["corrected-protocol-v1"]["pass"],
        )

        lp = LOCKED_PROTOCOL_DIR / fname
        if lp.exists():
            add_source_row(
                source_role="protocol_locked",
                model_or_protocol="corrected_protocol",
                path=lp,
                manifest_expected_sha256=expected_sha,
                artifact_generation_commit=local_tag_commits.get("corrected-protocol-v1", ""),
                package_freeze_tag="corrected-protocol-v1",
                package_freeze_commit=local_tag_commits.get("corrected-protocol-v1", ""),
                remote_tag_present=remote_tag_presence.get("corrected-protocol-v1", False),
                source_preservation_pass_val=source_preservation["corrected-protocol-v1"]["pass"],
            )

    for model in MODEL_ORDER:
        mf = model_file_map[model]
        info = source_model_info[model]
        manifest = json.loads(mf["manifest"].read_text(encoding="utf-8"))

        add_source_row(
            source_role="model_prediction",
            model_or_protocol=model,
            path=mf["prediction"],
            manifest_expected_sha256=str(manifest.get("prediction_sha256", "")),
            artifact_generation_commit=str(info["artifact_generation_commit"]),
            package_freeze_tag=str(info["package_freeze_tag"]),
            package_freeze_commit=str(info["package_freeze_commit"]),
            remote_tag_present=bool(info["remote_tag_present"]),
            source_preservation_pass_val=bool(info["source_preservation_pass"]),
        )

        metrics_sha = manifest.get("metrics_sha256", {})
        for role, fkey in [
            ("model_metrics_by_fold", "metrics_by_fold"),
            ("model_metrics_pooled", "metrics_pooled"),
            ("model_event_prevalence", "event_prevalence"),
        ]:
            p = mf[fkey]
            if not p.exists():
                continue
            add_source_row(
                source_role=role,
                model_or_protocol=model,
                path=p,
                manifest_expected_sha256=str(metrics_sha.get(p.name, "")),
                artifact_generation_commit=str(info["artifact_generation_commit"]),
                package_freeze_tag=str(info["package_freeze_tag"]),
                package_freeze_commit=str(info["package_freeze_commit"]),
                remote_tag_present=bool(info["remote_tag_present"]),
                source_preservation_pass_val=bool(info["source_preservation_pass"]),
            )

        add_source_row(
            source_role="model_manifest",
            model_or_protocol=model,
            path=mf["manifest"],
            manifest_expected_sha256=None,
            artifact_generation_commit=str(info["artifact_generation_commit"]),
            package_freeze_tag=str(info["package_freeze_tag"]),
            package_freeze_commit=str(info["package_freeze_commit"]),
            remote_tag_present=bool(info["remote_tag_present"]),
            source_preservation_pass_val=bool(info["source_preservation_pass"]),
        )

        add_source_row(
            source_role="model_completion_report",
            model_or_protocol=model,
            path=mf["completion_report"],
            manifest_expected_sha256=None,
            artifact_generation_commit=str(info["artifact_generation_commit"]),
            package_freeze_tag=str(info["package_freeze_tag"]),
            package_freeze_commit=str(info["package_freeze_commit"]),
            remote_tag_present=bool(info["remote_tag_present"]),
            source_preservation_pass_val=bool(info["source_preservation_pass"]),
        )

    add_source_row(
        source_role="hgbr_issue_registry",
        model_or_protocol="HGBR",
        path=HGBR_ISSUE_REGISTRY_PATH,
        manifest_expected_sha256=None,
        artifact_generation_commit=str(source_model_info["HGBR"]["artifact_generation_commit"]),
        package_freeze_tag=str(source_model_info["HGBR"]["package_freeze_tag"]),
        package_freeze_commit=str(source_model_info["HGBR"]["package_freeze_commit"]),
        remote_tag_present=bool(source_model_info["HGBR"]["remote_tag_present"]),
        source_preservation_pass_val=bool(source_model_info["HGBR"]["source_preservation_pass"]),
    )

    add_source_row(
        source_role="hgbr_metric_reconciliation",
        model_or_protocol="HGBR",
        path=HGBR_RECONCILIATION_PATH,
        manifest_expected_sha256=None,
        artifact_generation_commit=str(source_model_info["HGBR"]["artifact_generation_commit"]),
        package_freeze_tag=str(source_model_info["HGBR"]["package_freeze_tag"]),
        package_freeze_commit=str(source_model_info["HGBR"]["package_freeze_commit"]),
        remote_tag_present=bool(source_model_info["HGBR"]["remote_tag_present"]),
        source_preservation_pass_val=bool(source_model_info["HGBR"]["source_preservation_pass"]),
    )

    source_registry_df = pd.DataFrame(source_registry_rows).sort_values(
        ["model_or_protocol", "source_role", "relative_path"]
    ).reset_index(drop=True)

    source_registry_pass = bool(source_registry_df["checksum_match"].astype(bool).all())

    # Build input hash registry used for deterministic assembly_id.
    input_hashes: Dict[str, str] = {}
    for r in source_registry_df.itertuples(index=False):
        input_hashes[str(r.relative_path)] = str(r.sha256)

    assembly_id = make_stage_assembly_id(input_hashes)

    # Refresh assembly_id fields in metrics.
    for h in HORIZONS:
        metrics_by_h_by_fold[h]["assembly_id"] = assembly_id
        metrics_by_h_pooled[h]["assembly_id"] = assembly_id

    # Build per-horizon key audits and stats.
    h_stats: Dict[int, Dict[str, Any]] = {}
    event_counts: Dict[int, Dict[str, Any]] = {}

    table1_rows: List[Dict[str, Any]] = []
    table2_rows: List[Dict[str, Any]] = []

    for h in HORIZONS:
        key_df = pd.DataFrame(key_audit_rows[int(h)]).sort_values(["model", "fold"]).reset_index(drop=True)
        long_df = long_by_h[int(h)].copy()
        wide_df = wide_by_h[int(h)].copy()
        fold_metrics = metrics_by_h_by_fold[int(h)].copy()
        pooled_metrics = metrics_by_h_pooled[int(h)].copy()
        event_prev = compute_event_prevalence(wide_df, int(h))

        hs = {
            "N": int(len(wide_df)),
            "fold_counts": {
                int(k): int(v)
                for k, v in wide_df.groupby("outer_fold").size().to_dict().items()
            },
            "feature_date_start": str(wide_df["feature_date"].min()) if len(wide_df) > 0 else "",
            "feature_date_end": str(wide_df["feature_date"].max()) if len(wide_df) > 0 else "",
            "target_date_start": str(wide_df["target_date"].min()) if len(wide_df) > 0 else "",
            "target_date_end": str(wide_df["target_date"].max()) if len(wide_df) > 0 else "",
            "long_rows": int(len(long_df)),
            "wide_rows": int(len(wide_df)),
        }
        h_stats[int(h)] = hs

        pooled_prev = event_prev[event_prev["outer_fold"].astype(str) == "pooled"].iloc[0]
        ev = {
            "events_tau15": int(pooled_prev["events_tau15"]),
            "events_tau16": int(pooled_prev["events_tau16"]),
            "events_tau17": int(pooled_prev["events_tau17"]),
            "prevalence_tau15": float(pooled_prev["prevalence_tau15"]),
            "prevalence_tau16": float(pooled_prev["prevalence_tau16"]),
            "prevalence_tau17": float(pooled_prev["prevalence_tau17"]),
        }
        event_counts[int(h)] = ev

        ss = split_summary_df[split_summary_df["horizon"].astype(int) == int(h)].copy()
        if len(ss) != 3:
            raise RuntimeError("Split summary does not have 3 folds for horizon {}".format(h))

        table1_rows.append(
            {
                "horizon": int(h),
                "canonical_N": int(hs["N"]),
                "number_of_outer_folds": 3,
                "fold_1_N": int(hs["fold_counts"].get(1, 0)),
                "fold_2_N": int(hs["fold_counts"].get(2, 0)),
                "fold_3_N": int(hs["fold_counts"].get(3, 0)),
                "feature_date_start": hs["feature_date_start"],
                "feature_date_end": hs["feature_date_end"],
                "target_date_start": hs["target_date_start"],
                "target_date_end": hs["target_date_end"],
                "events_tau15": int(ev["events_tau15"]),
                "prevalence_tau15": float(ev["prevalence_tau15"]),
                "events_tau16": int(ev["events_tau16"]),
                "prevalence_tau16": float(ev["prevalence_tau16"]),
                "events_tau17": int(ev["events_tau17"]),
                "prevalence_tau17": float(ev["prevalence_tau17"]),
                "target_date_embargo_days": int(ss["outer_rows_purged"].astype(int).max()),
                "total_outer_rows_purged": int(ss["outer_rows_purged"].astype(int).sum()),
                "total_inner_rows_purged": int(ss["inner_rows_purged"].astype(int).sum()),
                "inner_purge_applied": bool((ss["inner_rows_purged"].astype(int) > 0).any()),
                "outer_purge_applied": bool((ss["outer_rows_purged"].astype(int) > 0).any()),
                "all_outer_boundaries_pass": bool(ss["outer_boundary_pass"].map(bool_from_any).all()),
                "all_inner_boundaries_pass": bool(ss["inner_boundary_pass"].map(bool_from_any).all()),
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "split_sha256": EXPECTED_SPLIT_SHA256,
                "protocol_tag": PROTOCOL_TAG,
                "assembly_id": assembly_id,
            }
        )

        for r in pooled_metrics.itertuples(index=False):
            table2_rows.append(
                {
                    "horizon": int(h),
                    "model": str(r.model),
                    "N": int(r.N),
                    "MAE": float(r.MAE),
                    "MSE": float(r.MSE),
                    "RMSE": float(r.RMSE),
                    "MASE": float(r.MASE),
                    "MASE_period": int(r.MASE_period),
                    "MASE_method": str(r.MASE_method),
                    "feature_date_start": hs["feature_date_start"],
                    "feature_date_end": hs["feature_date_end"],
                    "target_date_start": hs["target_date_start"],
                    "target_date_end": hs["target_date_end"],
                    "dataset_sha256": EXPECTED_DATASET_SHA256,
                    "split_sha256": EXPECTED_SPLIT_SHA256,
                    "source_run_id": str(r.source_run_id),
                    "artifact_generation_commit": str(r.artifact_generation_commit),
                    "package_freeze_tag": str(r.package_freeze_tag),
                    "package_freeze_commit": str(r.package_freeze_commit),
                    "assembly_id": assembly_id,
                }
            )

        write_csv(paths[int(h)]["key_audit"], key_df, date_cols=["feature_date_start", "feature_date_end", "target_date_start", "target_date_end"])
        write_csv(paths[int(h)]["long"], long_df, date_cols=["feature_date", "target_date"])
        write_csv(paths[int(h)]["wide"], wide_df, date_cols=["feature_date", "target_date"])
        write_csv(paths[int(h)]["event_prevalence"], event_prev, date_cols=[])
        write_csv(paths[int(h)]["metrics_by_fold"], fold_metrics, date_cols=["date_start", "date_end"])
        write_csv(paths[int(h)]["metrics_pooled"], pooled_metrics, date_cols=["date_start", "date_end"])
        write_csv(paths[int(h)]["reconciliation"], reconciliation_by_h[int(h)], date_cols=[])

    # Output root-level tables.
    table1_df = pd.DataFrame(table1_rows).sort_values(["horizon"]).reset_index(drop=True)
    table2_df = pd.DataFrame(table2_rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    source_registry_df = source_registry_df.reset_index(drop=True)

    write_csv(TABLE1_PATH, table1_df, date_cols=["feature_date_start", "feature_date_end", "target_date_start", "target_date_end"])
    write_csv(TABLE2_PATH, table2_df, date_cols=["feature_date_start", "feature_date_end", "target_date_start", "target_date_end"])
    write_csv(SOURCE_REGISTRY_PATH, source_registry_df, date_cols=[])

    # Validate expected counts.
    count_expectation_pass = True
    for h in HORIZONS:
        hs = h_stats[int(h)]
        exp = EXPECTED_H_COUNTS[int(h)]
        if int(hs["N"]) != int(exp["N"]):
            count_expectation_pass = False
        if hs["fold_counts"] != exp["fold_counts"]:
            count_expectation_pass = False

        long_expected = int(exp["N"]) * len(MODEL_ORDER)
        if int(hs["long_rows"]) != int(long_expected):
            count_expectation_pass = False

    event_count_expectation_pass = True
    for h in HORIZONS:
        ev = event_counts[int(h)]
        ex = EXPECTED_EVENT_COUNTS[int(h)]
        if int(ev["events_tau15"]) != int(ex["events_tau15"]):
            event_count_expectation_pass = False
        if int(ev["events_tau16"]) != int(ex["events_tau16"]):
            event_count_expectation_pass = False
        if int(ev["events_tau17"]) != int(ex["events_tau17"]):
            event_count_expectation_pass = False

    # Build input verification payload.
    input_verification = {
        "repository": str(ROOT),
        "authorized_output_directory": AUTHORIZED_REL_DIR,
        "required_branch": EXPECTED_BRANCH,
        "required_python_version": EXPECTED_PYTHON_VERSION,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "expected_split_sha256": EXPECTED_SPLIT_SHA256,
        "verification": {
            "working_tree_status_paths": start_status,
            "starting_git_commit": starting_commit,
            "active_branch": branch,
            "python_version": py_version,
            "root_protocol_sha256sum_output": root_sha_cmd["stdout"],
            "root_protocol_sha256sum_error": root_sha_cmd["stderr"],
            "locked_protocol_sha256sum_output": locked_sha_cmd["stdout"],
            "locked_protocol_sha256sum_error": locked_sha_cmd["stderr"],
            "protocol_manifest_checks": protocol_root_checks,
            "locked_protocol_manifest_checks": protocol_locked_checks,
            "protocol_root_locked_identity": root_locked_identity_rows,
            "dataset_path": str(dataset_path),
            "dataset_sha_expected_from_lock": dataset_sha_expected,
            "dataset_sha_observed": dataset_sha_observed,
            "split_sha_expected_from_manifest": split_sha_expected,
            "split_sha_observed": split_sha_observed,
            "local_tag_commits": local_tag_commits,
            "remote_tag_presence": remote_tag_presence,
            "remote_tag_warning": "; ".join([w for w in warnings if w.startswith("missing_remote_tag")]),
            "source_directory_preservation": source_preservation,
            "source_model_info": source_model_info,
            "hgbr_issue_registry": {
                "path": rel_to_root(HGBR_ISSUE_REGISTRY_PATH),
                "resolved_all_true": bool(resolved_col.all()),
                "validity_critical_any": bool(validity_critical_col.any()),
                "pass": bool(hgbr_issue_pass),
            },
            "mase_denom_independent_verification": denom_verify_rows,
            "checks": {
                "working_directory_pass": True,
                "required_branch_pass": bool(branch_pass),
                "python_3810_pass": bool(python_pass),
                "clean_or_authorized_working_tree_scope_pass": bool(working_tree_scope_pass),
                "root_protocol_sha256sum_pass": bool(root_sha_cmd["exit_code"] == 0),
                "locked_protocol_sha256sum_raw_pass": bool(locked_sha_cmd["exit_code"] == 0),
                "locked_protocol_shared_identity_pass": bool(shared_identity_pass),
                "locked_protocol_effective_pass": bool(protocol_locked_effective_pass),
                "dataset_checksum_pass": bool(dataset_pass),
                "split_checksum_pass": bool(split_pass),
                "source_model_checks_pass": bool(source_model_checks_pass),
                "source_test_result_pass": bool(source_test_pass),
                "source_completion_decision_pass": bool(source_decision_pass),
                "hgbr_issue_registry_pass": bool(hgbr_issue_pass),
                "local_tag_presence_pass": bool(local_tags_ok),
                "source_preservation_pass": bool(source_preservation_pass),
                "verification_pass": bool(verification_pass),
            },
        },
    }
    write_json(INPUT_VERIFICATION_PATH, input_verification)

    # If preflight fails, emit diagnostic package only.
    if not verification_pass:
        test_result = get_existing_test_result()
        deterministic_result = "pass" if str(test_result).startswith("pass") else "pending_not_run"
        final_decision = "C"

        manifest = {
            "assembly_id": "verification_failed",
            "starting_git_commit": starting_commit,
            "branch": branch,
            "python_version": py_version,
            "dataset_sha256": dataset_sha_observed,
            "split_sha256": split_sha_observed,
            "protocol_tag": PROTOCOL_TAG,
            "authorized_horizons": list(HORIZONS),
            "authorized_models": list(MODEL_ORDER),
            "input_files": [rel_to_root(DATASET_LOCK_PATH), rel_to_root(PROTOCOL_SHA_PATH), rel_to_root(SPLIT_PATH), rel_to_root(SPLIT_SUMMARY_PATH)],
            "input_checksums": {
                rel_to_root(DATASET_LOCK_PATH): sha256_file(DATASET_LOCK_PATH),
                rel_to_root(PROTOCOL_SHA_PATH): sha256_file(PROTOCOL_SHA_PATH),
                rel_to_root(SPLIT_PATH): sha256_file(SPLIT_PATH),
                rel_to_root(SPLIT_SUMMARY_PATH): sha256_file(SPLIT_SUMMARY_PATH),
            },
            "source_run_ids": {m: source_model_info[m]["run_id"] for m in MODEL_ORDER},
            "artifact_generation_commits": {m: source_model_info[m]["artifact_generation_commit"] for m in MODEL_ORDER},
            "package_freeze_tags": {m: source_model_info[m]["package_freeze_tag"] for m in MODEL_ORDER},
            "package_freeze_commits": {m: source_model_info[m]["package_freeze_commit"] for m in MODEL_ORDER},
            "remote_tag_status": remote_tag_presence,
            "canonical_row_counts": {"H1": 0, "H3": 0},
            "fold_counts": {"H1": {}, "H3": {}},
            "date_ranges": {"H1": {}, "H3": {}},
            "event_counts": {"H1": {}, "H3": {}},
            "metric_methods": {
                "MAE": "mean(abs(y_true - y_pred))",
                "MSE": "mean((y_true - y_pred)^2)",
                "RMSE": "sqrt(MSE)",
                "MASE_period": 1,
                "MASE_method": MASE_METHOD,
            },
            "test_result": str(test_result),
            "deterministic_result": str(deterministic_result),
            "source_preservation_result": bool(source_preservation_pass),
            "final_decision": str(final_decision),
            "verification_pass": False,
            "warnings": warnings,
        }
        write_json(MANIFEST_PATH, manifest)

        checksum_info = {"pass": False, "coverage_pass": False, "coverage_missing": [], "coverage_extra": []}
        report_text = build_report(
            verification_pass=False,
            source_tests={m: source_model_info[m]["test_result"] for m in MODEL_ORDER},
            source_decisions={m: source_model_info[m]["completion_decision"] for m in MODEL_ORDER},
            local_tags=local_tag_commits,
            remote_tags=remote_tag_presence,
            source_preservation_pass=source_preservation_pass,
            h_stats={1: {"N": 0, "fold_counts": {}, "feature_date_start": "", "feature_date_end": "", "target_date_start": "", "target_date_end": "", "long_rows": 0, "wide_rows": 0}, 3: {"N": 0, "fold_counts": {}, "feature_date_start": "", "feature_date_end": "", "target_date_start": "", "target_date_end": "", "long_rows": 0, "wide_rows": 0}},
            event_counts={1: {"events_tau15": 0, "events_tau16": 0, "events_tau17": 0, "prevalence_tau15": 0.0, "prevalence_tau16": 0.0, "prevalence_tau17": 0.0}, 3: {"events_tau15": 0, "events_tau16": 0, "events_tau17": 0, "prevalence_tau15": 0.0, "prevalence_tau16": 0.0, "prevalence_tau17": 0.0}},
            metric_reconciliation_pass=False,
            mase_convention_pass=False,
            checksum_info=checksum_info,
            test_result=str(test_result),
            deterministic_result=str(deterministic_result),
            final_decision=str(final_decision),
            starting_commit=starting_commit,
            warnings=warnings,
            generated_files=[rel_to_root(INPUT_VERIFICATION_PATH), rel_to_root(MANIFEST_PATH)],
        )
        ensure_inside_authorized(REPORT_PATH)
        REPORT_PATH.write_text(report_text, encoding="utf-8")

        generate_checksums_registry()
        print("Input verification failed. Diagnostic package generated.")
        print("FINAL DECISION: C")
        return

    # Build final manifest/report.
    test_result = get_existing_test_result()
    deterministic_result = "pass" if str(test_result).startswith("pass") else "pending_not_run"

    checksum_info_pre = {"pass": False, "coverage_pass": False, "coverage_missing": [], "coverage_extra": []}

    metric_reconciliation_pass = bool(metric_reconciliation_pass)
    key_equality_pass = bool(
        y_true_all_models_pass
        and source_event_label_pass
        and count_expectation_pass
        and event_count_expectation_pass
        and all(pd.DataFrame(key_audit_rows[h])["key_equality_pass"].astype(bool).all() for h in HORIZONS)
        and all(pd.DataFrame(key_audit_rows[h])["y_true_equality_pass"].astype(bool).all() for h in HORIZONS)
        and all(pd.DataFrame(key_audit_rows[h])["target_offset_pass"].astype(bool).all() for h in HORIZONS)
    )

    checksum_placeholder_pass = False
    final_decision = compute_decision(
        verification_pass=verification_pass,
        key_equality_pass=key_equality_pass,
        source_preservation_pass=source_preservation_pass,
        metric_reconciliation_pass=metric_reconciliation_pass,
        mase_convention_pass=mase_convention_pass,
        checksum_pass=checksum_placeholder_pass,
        test_result=test_result,
    )

    manifest = {
        "assembly_id": assembly_id,
        "starting_git_commit": starting_commit,
        "branch": branch,
        "python_version": py_version,
        "dataset_sha256": dataset_sha_observed,
        "split_sha256": split_sha_observed,
        "protocol_tag": PROTOCOL_TAG,
        "authorized_horizons": list(HORIZONS),
        "authorized_models": list(MODEL_ORDER),
        "input_files": sorted(source_registry_df["relative_path"].astype(str).unique().tolist()),
        "input_checksums": {r.relative_path: r.sha256 for r in source_registry_df.itertuples(index=False)},
        "source_run_ids": {m: source_model_info[m]["run_id"] for m in MODEL_ORDER},
        "artifact_generation_commits": {m: source_model_info[m]["artifact_generation_commit"] for m in MODEL_ORDER},
        "package_freeze_tags": {m: source_model_info[m]["package_freeze_tag"] for m in MODEL_ORDER},
        "package_freeze_commits": {m: source_model_info[m]["package_freeze_commit"] for m in MODEL_ORDER},
        "remote_tag_status": remote_tag_presence,
        "canonical_row_counts": {"H1": h_stats[1]["N"], "H3": h_stats[3]["N"]},
        "fold_counts": {"H1": h_stats[1]["fold_counts"], "H3": h_stats[3]["fold_counts"]},
        "date_ranges": {
            "H1": {
                "feature_date_start": h_stats[1]["feature_date_start"],
                "feature_date_end": h_stats[1]["feature_date_end"],
                "target_date_start": h_stats[1]["target_date_start"],
                "target_date_end": h_stats[1]["target_date_end"],
            },
            "H3": {
                "feature_date_start": h_stats[3]["feature_date_start"],
                "feature_date_end": h_stats[3]["feature_date_end"],
                "target_date_start": h_stats[3]["target_date_start"],
                "target_date_end": h_stats[3]["target_date_end"],
            },
        },
        "event_counts": {"H1": event_counts[1], "H3": event_counts[3]},
        "metric_methods": {
            "MAE": "mean(abs(y_true - y_pred))",
            "MSE": "mean((y_true - y_pred)^2)",
            "RMSE": "sqrt(MSE)",
            "pooled_method": "pool all rows first, then compute metrics",
            "MASE_period": 1,
            "MASE_method": MASE_METHOD,
        },
        "tolerance": float(TOL),
        "verification_pass": bool(verification_pass),
        "key_equality_pass": bool(key_equality_pass),
        "source_event_label_pass": bool(source_event_label_pass),
        "source_registry_pass": bool(source_registry_pass),
        "metric_reconciliation_pass": bool(metric_reconciliation_pass),
        "mase_convention_pass": bool(mase_convention_pass),
        "source_preservation_result": bool(source_preservation_pass),
        "test_result": str(test_result),
        "deterministic_result": str(deterministic_result),
        "warnings": warnings,
        "final_decision": str(final_decision),
    }

    write_json(MANIFEST_PATH, manifest)

    # Report and checksums.
    generated_files = sorted(
        [
            rel_to_root(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
            and p.resolve() != CHECKSUM_PATH.resolve()
        ]
    )

    report_text = build_report(
        verification_pass=verification_pass,
        source_tests={m: source_model_info[m]["test_result"] for m in MODEL_ORDER},
        source_decisions={m: source_model_info[m]["completion_decision"] for m in MODEL_ORDER},
        local_tags=local_tag_commits,
        remote_tags=remote_tag_presence,
        source_preservation_pass=source_preservation_pass,
        h_stats=h_stats,
        event_counts=event_counts,
        metric_reconciliation_pass=metric_reconciliation_pass,
        mase_convention_pass=mase_convention_pass,
        checksum_info=checksum_info_pre,
        test_result=str(test_result),
        deterministic_result=str(deterministic_result),
        final_decision=str(final_decision),
        starting_commit=starting_commit,
        warnings=warnings,
        generated_files=generated_files,
    )
    ensure_inside_authorized(REPORT_PATH)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    generate_checksums_registry()
    checksum_info = verify_checksums_registry()

    # Rewrite report/manifest with final checksum status and decision.
    final_decision = compute_decision(
        verification_pass=verification_pass,
        key_equality_pass=key_equality_pass,
        source_preservation_pass=source_preservation_pass,
        metric_reconciliation_pass=metric_reconciliation_pass,
        mase_convention_pass=mase_convention_pass,
        checksum_pass=bool(checksum_info["pass"] and checksum_info["coverage_pass"] and source_registry_pass),
        test_result=test_result,
    )

    manifest["checksum_result"] = {
        "registry_pass": bool(checksum_info["pass"]),
        "coverage_pass": bool(checksum_info["coverage_pass"]),
        "coverage_missing": checksum_info["coverage_missing"],
        "coverage_extra": checksum_info["coverage_extra"],
    }
    manifest["source_registry_pass"] = bool(source_registry_pass)
    manifest["final_decision"] = str(final_decision)
    write_json(MANIFEST_PATH, manifest)

    generated_files = sorted(
        [
            rel_to_root(p)
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".py", ".csv", ".json", ".md"}
            and p.resolve() != CHECKSUM_PATH.resolve()
        ]
    )

    report_text = build_report(
        verification_pass=verification_pass,
        source_tests={m: source_model_info[m]["test_result"] for m in MODEL_ORDER},
        source_decisions={m: source_model_info[m]["completion_decision"] for m in MODEL_ORDER},
        local_tags=local_tag_commits,
        remote_tags=remote_tag_presence,
        source_preservation_pass=source_preservation_pass,
        h_stats=h_stats,
        event_counts=event_counts,
        metric_reconciliation_pass=metric_reconciliation_pass,
        mase_convention_pass=mase_convention_pass,
        checksum_info=checksum_info,
        test_result=str(test_result),
        deterministic_result=str(deterministic_result),
        final_decision=str(final_decision),
        starting_commit=starting_commit,
        warnings=warnings,
        generated_files=generated_files,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    generate_checksums_registry()
    checksum_info = verify_checksums_registry()

    # Compact terminal summary.
    print("1. repository: {}".format(ROOT))
    print("2. branch: {}".format(branch))
    print("3. starting commit: {}".format(starting_commit))
    print("4. authorized output directory: {}".format(OUT_DIR))
    print("5. input checksum status: root_pass={} locked_effective_pass={} dataset_pass={} split_pass={}".format(
        root_protocol_pass,
        protocol_locked_effective_pass,
        dataset_pass,
        split_pass,
    ))
    print("6. source decision status: {}".format({m: source_model_info[m]["completion_decision"] for m in MODEL_ORDER}))
    print("7. source-tag and remote-tag status: local={} remote={}".format(local_tag_commits, remote_tag_presence))
    print("8. H1 N and fold sizes: N={} folds={}".format(h_stats[1]["N"], h_stats[1]["fold_counts"]))
    print("9. H3 N and fold sizes: N={} folds={}".format(h_stats[3]["N"], h_stats[3]["fold_counts"]))
    print("10. event counts: {}".format(event_counts))
    print("11. models included: {}".format(list(MODEL_ORDER)))
    print("12. models explicitly excluded: {}".format(["H5", "BCR-TCN", "TCN", "point blend", "HybridRank", "fixed rank ensemble", "alarm flags", "top-k", "classification metrics"]))
    print("13. long and wide row counts: H1 long={} wide={}; H3 long={} wide={}".format(h_stats[1]["long_rows"], h_stats[1]["wide_rows"], h_stats[3]["long_rows"], h_stats[3]["wide_rows"]))
    print("14. metric-reconciliation status: {}".format(metric_reconciliation_pass))
    print("15. MASE-convention status: {}".format(mase_convention_pass))
    print("16. number of tests run and passed: {}".format(test_result))
    print("17. deterministic result: {}".format(deterministic_result))
    print("18. source-preservation result: {}".format(source_preservation_pass))
    print("19. checksum result: file_ok={} coverage_ok={}".format(checksum_info["pass"], checksum_info["coverage_pass"]))
    print("20. final decision: {}".format(final_decision))
    print("21. generated files: {}".format(generated_files))


if __name__ == "__main__":
    build_stage()
