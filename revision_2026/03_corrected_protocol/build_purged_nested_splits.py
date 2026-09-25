from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = Path(__file__).resolve().parent

LOCK_PATH = WORKDIR / "canonical_dataset_lock.json"
SPLIT_OUT = WORKDIR / "corrected_split_assignment.csv"
SUMMARY_OUT = WORKDIR / "corrected_split_summary.csv"

HORIZONS = (1, 3, 5)
OUTER_FOLDS = (1, 2, 3)
INNER_VAL_FRAC = 0.15


@dataclass(frozen=True)
class CanonicalDataset:
    frame: pd.DataFrame
    date_column: str
    target_column: str
    date_to_original_index: Dict[pd.Timestamp, int]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_lock() -> dict:
    if not LOCK_PATH.exists():
        raise FileNotFoundError(f"Missing lock file: {LOCK_PATH}")
    with LOCK_PATH.open("r", encoding="utf-8") as f:
        lock = json.load(f)
    return lock


def load_canonical_dataset(lock: dict) -> CanonicalDataset:
    data_path = Path(lock["absolute_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Locked dataset path does not exist: {data_path}")

    observed_sha = sha256_file(data_path)
    expected_sha = lock["sha256"]
    if observed_sha != expected_sha:
        raise RuntimeError(
            "Canonical dataset checksum mismatch: "
            f"expected {expected_sha}, observed {observed_sha}"
        )

    raw = pd.read_csv(data_path)
    raw["__raw_row_index"] = range(len(raw))

    unnamed = [c for c in raw.columns if str(c).startswith("Unnamed")]
    if unnamed:
        raw = raw.drop(columns=unnamed)

    date_col = lock["date_column"]
    target_col = lock["target_column"]

    if date_col not in raw.columns:
        raise RuntimeError(f"Date column {date_col} is not present in locked dataset")
    if target_col not in raw.columns:
        raise RuntimeError(f"Target column {target_col} is not present in locked dataset")

    raw[date_col] = normalize_date_series(raw[date_col])
    frame = raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if len(frame) != int(lock["row_count"]):
        raise RuntimeError(f"Row count mismatch: expected {lock['row_count']}, observed {len(frame)}")

    unique_dates = int(frame[date_col].nunique())
    if unique_dates != int(lock["unique_date_count"]):
        raise RuntimeError(
            f"Unique date count mismatch: expected {lock['unique_date_count']}, observed {unique_dates}"
        )

    min_date = frame[date_col].min().strftime("%Y-%m-%d")
    max_date = frame[date_col].max().strftime("%Y-%m-%d")
    if min_date != lock["minimum_date"]:
        raise RuntimeError(f"Minimum date mismatch: expected {lock['minimum_date']}, observed {min_date}")
    if max_date != lock["maximum_date"]:
        raise RuntimeError(f"Maximum date mismatch: expected {lock['maximum_date']}, observed {max_date}")

    dup_dates = int(frame.duplicated(subset=[date_col]).sum())
    if dup_dates != int(lock["duplicate_date_count"]):
        raise RuntimeError(
            f"Duplicate-date count mismatch: expected {lock['duplicate_date_count']}, observed {dup_dates}"
        )

    all_days = pd.date_range(frame[date_col].min(), frame[date_col].max(), freq="D")
    missing_days = int(len(all_days) - unique_dates)
    if missing_days != int(lock["missing_calendar_date_count"]):
        raise RuntimeError(
            "Missing-calendar-date count mismatch: "
            f"expected {lock['missing_calendar_date_count']}, observed {missing_days}"
        )

    date_to_original_index: Dict[pd.Timestamp, int] = {}
    for _, row in frame[[date_col, "__raw_row_index"]].iterrows():
        d = row[date_col]
        if d in date_to_original_index:
            raise RuntimeError(f"Duplicate date encountered in canonical dataset: {d}")
        date_to_original_index[d] = int(row["__raw_row_index"])

    return CanonicalDataset(
        frame=frame,
        date_column=date_col,
        target_column=target_col,
        date_to_original_index=date_to_original_index,
    )


def load_assignment_matrix(horizon: int) -> pd.DataFrame:
    path = ROOT / "results" / "tables" / f"blocked_cv_assignment_matrix_H{horizon}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical assignment matrix: {path}")

    df = pd.read_csv(path)
    required = ["date", "fold1_set", "fold2_set", "fold3_set"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Assignment matrix is missing required columns: {missing}")

    df["date"] = normalize_date_series(df["date"])
    if df["date"].isna().any():
        raise RuntimeError(f"Assignment matrix H{horizon} has unparsable dates")

    return df


def split_inner_validation(dates: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    n_total = len(dates)
    n_val = max(1, int(math.ceil(INNER_VAL_FRAC * n_total)))
    if n_val >= n_total:
        raise RuntimeError(
            "Inner validation split would leave no inner-train rows after outer purge: "
            f"n_total={n_total}, n_val={n_val}"
        )
    return dates[:-n_val], dates[-n_val:]


def fmt_date(x: pd.Timestamp) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def build() -> Tuple[pd.DataFrame, pd.DataFrame]:
    lock = load_lock()
    canonical = load_canonical_dataset(lock)

    split_rows: List[dict] = []
    summary_rows: List[dict] = []

    for horizon in HORIZONS:
        assignment = load_assignment_matrix(horizon)

        # Canonical windows are read from assignment matrix and reproduced exactly.
        for outer_fold in OUTER_FOLDS:
            fold_col = f"fold{outer_fold}_set"

            train_dates_all = sorted(assignment.loc[assignment[fold_col] == "train", "date"].tolist())
            test_dates = sorted(assignment.loc[assignment[fold_col] == "test", "date"].tolist())

            if not train_dates_all or not test_dates:
                raise RuntimeError(
                    f"Empty train/test split for horizon={horizon}, outer_fold={outer_fold}"
                )

            missing_from_canonical = [
                d for d in (train_dates_all + test_dates) if d not in canonical.date_to_original_index
            ]
            if missing_from_canonical:
                raise RuntimeError(
                    f"Assignment dates not found in locked dataset for horizon={horizon}, fold={outer_fold}: "
                    f"{[fmt_date(x) for x in missing_from_canonical[:5]]}"
                )

            min_outer_test_feature = min(test_dates)

            outer_purged_dates = [
                d for d in train_dates_all if (d + pd.Timedelta(days=horizon)) >= min_outer_test_feature
            ]
            outer_train_after = [
                d for d in train_dates_all if (d + pd.Timedelta(days=horizon)) < min_outer_test_feature
            ]

            if not outer_train_after:
                raise RuntimeError(
                    f"Outer purge removed all training rows for horizon={horizon}, fold={outer_fold}"
                )

            outer_boundary_pass = (
                max(d + pd.Timedelta(days=horizon) for d in outer_train_after) < min_outer_test_feature
            )
            if not outer_boundary_pass:
                raise RuntimeError(
                    f"Outer boundary leakage remains after purge for horizon={horizon}, fold={outer_fold}"
                )

            inner_train_before, inner_val_dates = split_inner_validation(outer_train_after)
            min_inner_val_feature = min(inner_val_dates)

            inner_purged_dates = [
                d
                for d in inner_train_before
                if (d + pd.Timedelta(days=horizon)) >= min_inner_val_feature
            ]
            inner_train_after = [
                d
                for d in inner_train_before
                if (d + pd.Timedelta(days=horizon)) < min_inner_val_feature
            ]

            if not inner_train_after:
                raise RuntimeError(
                    f"Inner purge removed all inner-train rows for horizon={horizon}, fold={outer_fold}"
                )

            inner_boundary_pass = (
                max(d + pd.Timedelta(days=horizon) for d in inner_train_after) < min_inner_val_feature
            )
            if not inner_boundary_pass:
                raise RuntimeError(
                    f"Inner boundary leakage remains after purge for horizon={horizon}, fold={outer_fold}"
                )

            inner_val_set = set(inner_val_dates)
            inner_train_before_set = set(inner_train_before)
            outer_purged_set = set(outer_purged_dates)
            inner_purged_set = set(inner_purged_dates)

            for d in train_dates_all:
                target_date = d + pd.Timedelta(days=horizon)
                purged_outer = d in outer_purged_set
                purged_inner = False
                inner_fold = 0
                inner_role = "not_applicable"
                purge_reason = ""

                if purged_outer:
                    purge_reason = "outer_target_date_not_before_outer_test_start"
                else:
                    inner_fold = 1
                    if d in inner_val_set:
                        inner_role = "inner_validation"
                    elif d in inner_train_before_set:
                        inner_role = "inner_train"
                        if d in inner_purged_set:
                            purged_inner = True
                            purge_reason = "inner_target_date_not_before_inner_validation_start"
                    else:
                        raise RuntimeError(
                            f"Date was not assigned to inner train/validation: {fmt_date(d)} "
                            f"(h={horizon}, fold={outer_fold})"
                        )

                split_rows.append(
                    {
                        "horizon": horizon,
                        "outer_fold": outer_fold,
                        "feature_date": fmt_date(d),
                        "target_date": fmt_date(target_date),
                        "outer_role": "outer_train",
                        "inner_fold": inner_fold,
                        "inner_role": inner_role,
                        "original_row_index": canonical.date_to_original_index[d],
                        "purged_outer_boundary": purged_outer,
                        "purged_inner_boundary": purged_inner,
                        "purge_reason": purge_reason,
                    }
                )

            for d in test_dates:
                target_date = d + pd.Timedelta(days=horizon)
                split_rows.append(
                    {
                        "horizon": horizon,
                        "outer_fold": outer_fold,
                        "feature_date": fmt_date(d),
                        "target_date": fmt_date(target_date),
                        "outer_role": "outer_test",
                        "inner_fold": 0,
                        "inner_role": "not_applicable",
                        "original_row_index": canonical.date_to_original_index[d],
                        "purged_outer_boundary": False,
                        "purged_inner_boundary": False,
                        "purge_reason": "",
                    }
                )

            summary_rows.append(
                {
                    "horizon": horizon,
                    "outer_fold": outer_fold,
                    "outer_train_n_before_purge": len(train_dates_all),
                    "outer_train_n_after_purge": len(outer_train_after),
                    "outer_test_n": len(test_dates),
                    "outer_rows_purged": len(outer_purged_dates),
                    "outer_train_feature_start": fmt_date(min(outer_train_after)),
                    "outer_train_feature_end": fmt_date(max(outer_train_after)),
                    "outer_train_target_end": fmt_date(
                        max(d + pd.Timedelta(days=horizon) for d in outer_train_after)
                    ),
                    "outer_test_feature_start": fmt_date(min(test_dates)),
                    "outer_test_feature_end": fmt_date(max(test_dates)),
                    "inner_train_n_before_purge": len(inner_train_before),
                    "inner_train_n_after_purge": len(inner_train_after),
                    "inner_validation_n": len(inner_val_dates),
                    "inner_rows_purged": len(inner_purged_dates),
                    "inner_train_target_end": fmt_date(
                        max(d + pd.Timedelta(days=horizon) for d in inner_train_after)
                    ),
                    "inner_validation_feature_start": fmt_date(min_inner_val_feature),
                    "outer_boundary_pass": outer_boundary_pass,
                    "inner_boundary_pass": inner_boundary_pass,
                }
            )

    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame(summary_rows)

    required_split_cols = [
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
    ]
    split_df = split_df[required_split_cols]

    required_summary_cols = [
        "horizon",
        "outer_fold",
        "outer_train_n_before_purge",
        "outer_train_n_after_purge",
        "outer_test_n",
        "outer_rows_purged",
        "outer_train_feature_start",
        "outer_train_feature_end",
        "outer_train_target_end",
        "outer_test_feature_start",
        "outer_test_feature_end",
        "inner_train_n_before_purge",
        "inner_train_n_after_purge",
        "inner_validation_n",
        "inner_rows_purged",
        "inner_train_target_end",
        "inner_validation_feature_start",
        "outer_boundary_pass",
        "inner_boundary_pass",
    ]
    summary_df = summary_df[required_summary_cols]

    outer_order = {"outer_train": 0, "outer_test": 1}
    inner_order = {"inner_train": 0, "inner_validation": 1, "not_applicable": 2}

    split_df["_outer_order"] = split_df["outer_role"].map(outer_order)
    split_df["_inner_order"] = split_df["inner_role"].map(inner_order)
    split_df = split_df.sort_values(
        ["horizon", "outer_fold", "feature_date", "_outer_order", "_inner_order", "inner_fold"]
    ).drop(columns=["_outer_order", "_inner_order"]).reset_index(drop=True)

    summary_df = summary_df.sort_values(["horizon", "outer_fold"]).reset_index(drop=True)

    return split_df, summary_df


def validate_outputs(split_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    # Basic integrity checks performed in generator itself; these are final hard-stops.
    dup = split_df.duplicated(subset=["horizon", "outer_fold", "feature_date", "outer_role"])
    if bool(dup.any()):
        raise RuntimeError("Duplicate feature_date within same horizon/fold/outer_role detected")

    split_df = split_df.copy()
    split_df["feature_date"] = normalize_date_series(split_df["feature_date"])
    split_df["target_date"] = normalize_date_series(split_df["target_date"])

    for horizon in HORIZONS:
        for fold in OUTER_FOLDS:
            sub = split_df[(split_df["horizon"] == horizon) & (split_df["outer_fold"] == fold)]

            outer_train_kept = sub[(sub["outer_role"] == "outer_train") & (~sub["purged_outer_boundary"])]
            outer_test = sub[sub["outer_role"] == "outer_test"]
            if outer_train_kept.empty or outer_test.empty:
                raise RuntimeError(f"Unexpected empty partition for horizon={horizon}, fold={fold}")

            if outer_train_kept["target_date"].max() >= outer_test["feature_date"].min():
                raise RuntimeError(
                    f"Outer boundary condition failed for horizon={horizon}, fold={fold}"
                )

            inner_train_kept = outer_train_kept[
                (outer_train_kept["inner_role"] == "inner_train")
                & (~outer_train_kept["purged_inner_boundary"])
            ]
            inner_val = outer_train_kept[outer_train_kept["inner_role"] == "inner_validation"]

            if inner_train_kept.empty or inner_val.empty:
                raise RuntimeError(
                    f"Unexpected empty inner partition for horizon={horizon}, fold={fold}"
                )

            if inner_train_kept["target_date"].max() >= inner_val["feature_date"].min():
                raise RuntimeError(
                    f"Inner boundary condition failed for horizon={horizon}, fold={fold}"
                )

    if not bool(summary_df["outer_boundary_pass"].all()):
        raise RuntimeError("One or more outer boundary pass flags are False")
    if not bool(summary_df["inner_boundary_pass"].all()):
        raise RuntimeError("One or more inner boundary pass flags are False")


def main() -> None:
    split_df, summary_df = build()
    validate_outputs(split_df, summary_df)

    split_df.to_csv(SPLIT_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print(f"Wrote {SPLIT_OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
