from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "revision_2026" / "03_corrected_protocol"

LOCK_PATH = PROTOCOL_DIR / "canonical_dataset_lock.json"
SPLIT_PATH = PROTOCOL_DIR / "corrected_split_assignment.csv"
SUMMARY_PATH = PROTOCOL_DIR / "corrected_split_summary.csv"
PROTOCOL_SHA_PATH = PROTOCOL_DIR / "corrected_protocol_v1.sha256"

INPUT_VERIFICATION_PATH = OUT_DIR / "input_verification.json"
PREDICTIONS_PATH = OUT_DIR / "persistence_predictions.csv"
METRICS_BY_FOLD_PATH = OUT_DIR / "persistence_metrics_by_fold.csv"
METRICS_POOLED_PATH = OUT_DIR / "persistence_metrics_pooled.csv"
EVENT_PREVALENCE_PATH = OUT_DIR / "persistence_event_prevalence.csv"
MANIFEST_PATH = OUT_DIR / "persistence_run_manifest.json"

PROTOCOL_TAG = "corrected_protocol_v1"
MODEL_NAME = "Persistence"
HORIZONS = (1, 3, 5)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_output(args: List[str]) -> str:
    out = subprocess.check_output(args, cwd=ROOT, text=True)
    return out.strip()


def load_lock(lock_path: Path) -> dict:
    with lock_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_protocol_checksums(protocol_sha_path: Path, protocol_dir: Path) -> List[dict]:
    checks: List[dict] = []

    with protocol_sha_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for line in lines:
        # format: <hash><two spaces><filename>
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f"Malformed checksum line: {line}")
        expected_hash = parts[0]
        rel_name = parts[-1]
        path = protocol_dir / rel_name
        if not path.exists():
            raise RuntimeError(f"Protocol checksum target file is missing: {path}")
        observed_hash = sha256_file(path)
        ok = observed_hash == expected_hash
        checks.append(
            {
                "file": str(path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "pass": ok,
            }
        )
        if not ok:
            raise RuntimeError(
                f"Protocol checksum mismatch for {path.name}: expected {expected_hash}, observed {observed_hash}"
            )

    return checks


def normalize_date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_canonical_dataset(lock: dict) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, float], str]:
    dataset_path = Path(lock["absolute_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Locked dataset file does not exist: {dataset_path}")

    observed_sha = sha256_file(dataset_path)
    expected_sha = lock["sha256"]
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Dataset checksum mismatch: expected {expected_sha}, observed {observed_sha}"
        )

    df = pd.read_csv(dataset_path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    if date_col not in df.columns:
        raise RuntimeError(f"Locked date column is missing: {date_col}")
    if target_col not in df.columns:
        raise RuntimeError(f"Locked target column is missing: {target_col}")

    df[date_col] = normalize_date_col(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Lock metadata consistency
    if len(df) != int(lock["row_count"]):
        raise RuntimeError(
            f"Dataset row count mismatch: expected {lock['row_count']}, observed {len(df)}"
        )
    if int(df[date_col].nunique()) != int(lock["unique_date_count"]):
        raise RuntimeError(
            "Dataset unique-date count mismatch: "
            f"expected {lock['unique_date_count']}, observed {int(df[date_col].nunique())}"
        )
    if int(df.duplicated(subset=[date_col]).sum()) != int(lock["duplicate_date_count"]):
        raise RuntimeError("Dataset duplicate-date count mismatch")

    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    if min_date != lock["minimum_date"] or max_date != lock["maximum_date"]:
        raise RuntimeError(
            "Dataset date-range mismatch: "
            f"expected [{lock['minimum_date']}, {lock['maximum_date']}], observed [{min_date}, {max_date}]"
        )

    all_days = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    missing_days = int(len(all_days) - int(df[date_col].nunique()))
    if missing_days != int(lock["missing_calendar_date_count"]):
        raise RuntimeError(
            "Dataset missing-calendar-date mismatch: "
            f"expected {lock['missing_calendar_date_count']}, observed {missing_days}"
        )

    if df[target_col].isna().any():
        raise RuntimeError("Target column contains NaN values in the locked dataset")

    date_to_tnout: Dict[pd.Timestamp, float] = {}
    for _, row in df[[date_col, target_col]].iterrows():
        d = row[date_col]
        if d in date_to_tnout:
            raise RuntimeError(f"Duplicate date encountered when building date map: {d}")
        date_to_tnout[d] = float(row[target_col])

    return df, date_to_tnout, observed_sha


def load_split_assignments(path: Path) -> pd.DataFrame:
    split = pd.read_csv(path)
    required = {
        "horizon",
        "outer_fold",
        "feature_date",
        "target_date",
        "outer_role",
        "inner_fold",
        "inner_role",
        "original_row_index",
        "purged_outer_boundary",
        "purged_inner_boundary",
        "purge_reason",
    }
    missing = required.difference(split.columns)
    if missing:
        raise RuntimeError(f"Split assignment is missing required columns: {sorted(missing)}")

    split["feature_date"] = normalize_date_col(split["feature_date"])
    split["target_date"] = normalize_date_col(split["target_date"])

    if split["feature_date"].isna().any() or split["target_date"].isna().any():
        raise RuntimeError("Split assignment has unparsable feature_date or target_date")

    return split


def compute_mase_denom(y_train: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        return np.nan
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    software_versions = {
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    verification_payload = {
        "dataset_path": "",
        "dataset_sha256_expected": "",
        "dataset_sha256_observed": "",
        "split_file": str(SPLIT_PATH.resolve()),
        "split_sha256": "",
        "protocol_tag": PROTOCOL_TAG,
        "git_commit": "",
        "verification_pass": False,
        "timestamp": utc_now_iso(),
        **software_versions,
    }

    try:
        lock = load_lock(LOCK_PATH)
        verification_payload["dataset_path"] = str(Path(lock["absolute_path"]).resolve())
        verification_payload["dataset_sha256_expected"] = lock["sha256"]

        active_branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if active_branch != "controlled-reruns-v1":
            raise RuntimeError(
                f"Active branch is {active_branch}, expected controlled-reruns-v1"
            )

        git_commit = git_output(["git", "rev-parse", "HEAD"])
        verification_payload["git_commit"] = git_commit

        protocol_checks = verify_protocol_checksums(PROTOCOL_SHA_PATH, PROTOCOL_DIR)

        _, date_to_tnout, dataset_sha = load_canonical_dataset(lock)
        verification_payload["dataset_sha256_observed"] = dataset_sha

        split_sha = sha256_file(SPLIT_PATH)
        verification_payload["split_sha256"] = split_sha

        verification_payload["verification_pass"] = True
        verification_payload["protocol_checksum_checks"] = protocol_checks
        write_json(INPUT_VERIFICATION_PATH, verification_payload)

    except Exception as exc:
        verification_payload["verification_pass"] = False
        verification_payload["error"] = str(exc)
        write_json(INPUT_VERIFICATION_PATH, verification_payload)
        raise

    # Verified inputs are now guaranteed.
    split = load_split_assignments(SPLIT_PATH)
    split_sha = verification_payload["split_sha256"]
    git_commit = verification_payload["git_commit"]
    dataset_sha = verification_payload["dataset_sha256_observed"]

    run_id = (
        f"persistence_corrected_{git_commit[:12]}_{dataset_sha[:8]}_{split_sha[:8]}"
    )

    outer_test = split[
        (split["outer_role"] == "outer_test") & (split["horizon"].isin(HORIZONS))
    ].copy()

    if outer_test.empty:
        raise RuntimeError("No outer_test rows found in corrected split assignment")

    records: List[dict] = []
    missing_rows: List[str] = []

    for row in outer_test.itertuples(index=False):
        feature_date = row.feature_date
        target_date = row.target_date

        if feature_date not in date_to_tnout:
            missing_rows.append(
                f"missing feature_date {feature_date.strftime('%Y-%m-%d')} for H={row.horizon} F={row.outer_fold}"
            )
            continue
        if target_date not in date_to_tnout:
            missing_rows.append(
                f"missing target_date {target_date.strftime('%Y-%m-%d')} for H={row.horizon} F={row.outer_fold}"
            )
            continue

        y_pred = float(date_to_tnout[feature_date])
        y_true = float(date_to_tnout[target_date])
        err = y_pred - y_true

        records.append(
            {
                "model": MODEL_NAME,
                "horizon": int(row.horizon),
                "outer_fold": int(row.outer_fold),
                "feature_date": feature_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "y_true": y_true,
                "y_pred": y_pred,
                "error": err,
                "absolute_error": abs(err),
                "squared_error": err * err,
                "event_tau15": int(y_true >= 15.0),
                "event_tau16": int(y_true >= 16.0),
                "event_tau17": int(y_true >= 17.0),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "git_commit": git_commit,
                "run_id": run_id,
            }
        )

    if missing_rows:
        raise RuntimeError("Canonical test row lookup failure: " + " | ".join(missing_rows[:10]))

    pred = pd.DataFrame(records)
    pred = pred.sort_values(["horizon", "outer_fold", "feature_date"]).reset_index(drop=True)

    dup = pred.duplicated(subset=["horizon", "outer_fold", "feature_date"])
    if bool(dup.any()):
        raise RuntimeError("Duplicate horizon-fold-feature_date predictions detected")

    expected_counts = (
        outer_test.groupby(["horizon", "outer_fold"]).size().rename("expected_n").reset_index()
    )
    observed_counts = (
        pred.groupby(["horizon", "outer_fold"]).size().rename("observed_n").reset_index()
    )
    merged_counts = expected_counts.merge(observed_counts, on=["horizon", "outer_fold"], how="left")
    if bool((merged_counts["expected_n"] != merged_counts["observed_n"]).any()):
        raise RuntimeError("Predicted row counts do not match corrected split assignment")

    pred.to_csv(PREDICTIONS_PATH, index=False)

    # Build training denominators for MASE (m=1) from purged outer training rows only.
    mase_denoms: Dict[Tuple[int, int], float] = {}
    for h in HORIZONS:
        for fold in (1, 2, 3):
            tr = split[
                (split["horizon"] == h)
                & (split["outer_fold"] == fold)
                & (split["outer_role"] == "outer_train")
                & (~split["purged_outer_boundary"].astype(bool))
            ].copy()
            tr = tr.sort_values("feature_date")
            y_train = np.array([date_to_tnout[d] for d in tr["target_date"].tolist()], dtype=float)
            mase_denoms[(h, fold)] = compute_mase_denom(y_train, m=1)

    metrics_by_fold_rows: List[dict] = []
    for h in HORIZONS:
        for fold in (1, 2, 3):
            d = pred[(pred["horizon"] == h) & (pred["outer_fold"] == fold)].copy()
            if d.empty:
                raise RuntimeError(f"Missing prediction rows for horizon={h}, fold={fold}")

            mae = float(d["absolute_error"].mean())
            mse = float(d["squared_error"].mean())
            rmse = float(np.sqrt(mse))
            denom = mase_denoms[(h, fold)]
            mase = float(mae / (denom + 1e-9)) if np.isfinite(denom) else np.nan

            metrics_by_fold_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": h,
                    "outer_fold": fold,
                    "N": int(len(d)),
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MASE": mase,
                    "MASE_period": 1,
                    "MASE_denom_train_only": denom,
                    "date_start": d["feature_date"].min(),
                    "date_end": d["feature_date"].max(),
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "run_id": run_id,
                }
            )

    metrics_by_fold = pd.DataFrame(metrics_by_fold_rows).sort_values(["horizon", "outer_fold"])
    metrics_by_fold.to_csv(METRICS_BY_FOLD_PATH, index=False)

    # Pooled metrics from concatenated held-out rows (not fold-average).
    pooled_rows: List[dict] = []
    for h in HORIZONS:
        d = pred[pred["horizon"] == h].copy()
        if d.empty:
            raise RuntimeError(f"Missing pooled rows for horizon={h}")

        mae = float(d["absolute_error"].mean())
        mse = float(d["squared_error"].mean())
        rmse = float(np.sqrt(mse))

        # Direct pooled scaled-error mean using fold-local training denominators.
        denoms_for_rows = d["outer_fold"].map(lambda f: mase_denoms[(h, int(f))]).astype(float)
        if np.isfinite(denoms_for_rows.to_numpy()).all():
            mase_values = d["absolute_error"].to_numpy(dtype=float) / (denoms_for_rows.to_numpy(dtype=float) + 1e-9)
            mase = float(np.mean(mase_values))
        else:
            mase = np.nan

        pooled_rows.append(
            {
                "model": MODEL_NAME,
                "horizon": h,
                "outer_fold": "pooled",
                "N": int(len(d)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MASE": mase,
                "MASE_period": 1,
                "MASE_method": "mean_absolute_scaled_error_over_pooled_rows_using_fold_local_train_denominator",
                "date_start": d["feature_date"].min(),
                "date_end": d["feature_date"].max(),
                "dataset_sha256": dataset_sha,
                "split_sha256": split_sha,
                "run_id": run_id,
            }
        )

    metrics_pooled = pd.DataFrame(pooled_rows).sort_values(["horizon"])
    metrics_pooled.to_csv(METRICS_POOLED_PATH, index=False)

    prevalence_rows: List[dict] = []
    for h in HORIZONS:
        for fold in (1, 2, 3):
            d = pred[(pred["horizon"] == h) & (pred["outer_fold"] == fold)].copy()
            n = int(len(d))
            ev15 = int(d["event_tau15"].sum())
            ev16 = int(d["event_tau16"].sum())
            ev17 = int(d["event_tau17"].sum())

            prevalence_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": h,
                    "outer_fold": fold,
                    "N": n,
                    "events_tau15": ev15,
                    "events_tau16": ev16,
                    "events_tau17": ev17,
                    "prevalence_tau15": float(ev15 / n) if n else np.nan,
                    "prevalence_tau16": float(ev16 / n) if n else np.nan,
                    "prevalence_tau17": float(ev17 / n) if n else np.nan,
                    "dataset_sha256": dataset_sha,
                    "split_sha256": split_sha,
                    "run_id": run_id,
                }
            )

    prevalence_df = pd.DataFrame(prevalence_rows).sort_values(["horizon", "outer_fold"])
    prevalence_df.to_csv(EVENT_PREVALENCE_PATH, index=False)

    manifest = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "horizons": [1, 3, 5],
        "dataset_path": verification_payload["dataset_path"],
        "dataset_sha256": dataset_sha,
        "split_path": str(SPLIT_PATH.resolve()),
        "split_sha256": split_sha,
        "protocol_tag": PROTOCOL_TAG,
        "git_commit": git_commit,
        "execution_command": f"{sys.executable} {Path(__file__).resolve()}",
        "timestamp": utc_now_iso(),
        "software_versions": software_versions,
        "prediction_file": str(PREDICTIONS_PATH.resolve()),
        "prediction_sha256": sha256_file(PREDICTIONS_PATH),
        "metrics_files": [
            str(METRICS_BY_FOLD_PATH.resolve()),
            str(METRICS_POOLED_PATH.resolve()),
            str(EVENT_PREVALENCE_PATH.resolve()),
        ],
        "metrics_sha256": {
            str(METRICS_BY_FOLD_PATH.name): sha256_file(METRICS_BY_FOLD_PATH),
            str(METRICS_POOLED_PATH.name): sha256_file(METRICS_POOLED_PATH),
            str(EVENT_PREVALENCE_PATH.name): sha256_file(EVENT_PREVALENCE_PATH),
        },
        "test_result": "pending_not_run",
        "legacy_definition_source": [
            "src/run_linear_alarm.py:42-49",
            "src/run_linear_alarm_v2.py:26-32",
            "src/save_fold_predictions.py:15-20",
            "reports/_tmp_manuscript_extracted_latest.txt:230",
        ],
        "notes": "Persistence regenerated on locked canonical dataset and corrected split assignment only; no fitted models were run.",
    }
    write_json(MANIFEST_PATH, manifest)

    print(f"Wrote {INPUT_VERIFICATION_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_BY_FOLD_PATH}")
    print(f"Wrote {METRICS_POOLED_PATH}")
    print(f"Wrote {EVENT_PREVALENCE_PATH}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
