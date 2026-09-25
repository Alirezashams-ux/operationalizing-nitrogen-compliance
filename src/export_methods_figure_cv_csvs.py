from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
OUT_DIR = ROOT / "results" / "methods_figure_csvs"

HORIZONS = [1, 3, 5]
N_SPLITS = 3
VAL_FRAC = 0.15


def resolve_npz_path(horizon: int) -> Path:
    # Match project convention: prefer v2 feature build when available.
    candidates = [
        FEATURE_DIR / f"ulsan_H{horizon}_features_v2.npz",
        FEATURE_DIR / f"ulsan_H{horizon}_features.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No feature NPZ found for H={horizon}. Checked: {candidates}")


def load_dates_for_horizon(horizon: int) -> tuple[Path, pd.Series]:
    npz_path = resolve_npz_path(horizon)
    z = np.load(npz_path, allow_pickle=True)
    dates = pd.to_datetime(pd.Series(z["dates"].astype(str)), errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Invalid date parsing in {npz_path}")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"Dates are not sorted in {npz_path}")
    if dates.duplicated().any():
        dupes = int(dates.duplicated().sum())
        raise ValueError(f"Found {dupes} duplicate dates in {npz_path}")
    return npz_path, dates


def _phase_row(fold: int, phase: str, idx: np.ndarray, dates: pd.Series) -> dict:
    return {
        "fold": fold,
        "phase": phase,
        "start_date": dates.iloc[int(idx[0])].strftime("%Y-%m-%d"),
        "end_date": dates.iloc[int(idx[-1])].strftime("%Y-%m-%d"),
        "n_samples": int(len(idx)),
    }


def build_fold_boundaries(dates: pd.Series) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(np.arange(len(dates))), start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        te_idx = np.asarray(te_idx, dtype=int)

        cut = int(math.floor(len(tr_idx) * (1.0 - VAL_FRAC)))
        if cut <= 0 or cut >= len(tr_idx):
            raise ValueError(f"Invalid inner split for fold {fold}: cut={cut}, train={len(tr_idx)}")

        sub_tr = tr_idx[:cut]
        sub_va = tr_idx[cut:]

        # Validation checks requested by user.
        if np.intersect1d(sub_tr, sub_va).size != 0:
            raise ValueError(f"Overlap between subtrain and validation in fold {fold}")
        if np.intersect1d(sub_va, te_idx).size != 0:
            raise ValueError(f"Overlap between validation and test in fold {fold}")
        if not (sub_tr[-1] + 1 == sub_va[0]):
            raise ValueError(f"Non-contiguous subtrain->validation boundary in fold {fold}")
        if not (sub_va[-1] + 1 == te_idx[0]):
            raise ValueError(f"Non-contiguous validation->test boundary in fold {fold}")

        rows.append(_phase_row(fold, "subtrain", sub_tr, dates))
        rows.append(_phase_row(fold, "validation", sub_va, dates))
        rows.append(_phase_row(fold, "test", te_idx, dates))

    out = pd.DataFrame(rows)

    # Final strict chronology check in date-space for each fold.
    for fold in sorted(out["fold"].unique()):
        d = out[out["fold"] == fold].set_index("phase")
        st = pd.to_datetime(d.loc["subtrain", "start_date"])
        en_st = pd.to_datetime(d.loc["subtrain", "end_date"])
        va_st = pd.to_datetime(d.loc["validation", "start_date"])
        va_en = pd.to_datetime(d.loc["validation", "end_date"])
        te_st = pd.to_datetime(d.loc["test", "start_date"])
        if not (st <= en_st < va_st <= va_en < te_st):
            raise ValueError(f"Chronology violation in fold {fold}")

    return out


def export_one(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    npz_path, dates = load_dates_for_horizon(horizon)
    ready_df = pd.DataFrame({"Date": dates.dt.strftime("%Y-%m-%d")})
    folds_df = build_fold_boundaries(dates)
    return ready_df, folds_df, npz_path


def save_with_horizon_suffix(horizon: int, ready_df: pd.DataFrame, folds_df: pd.DataFrame) -> tuple[Path, Path]:
    date_path = OUT_DIR / f"model_ready_dates_ulsan_H{horizon}.csv"
    fold_path = OUT_DIR / f"cv_fold_boundaries_ulsan_H{horizon}.csv"
    ready_df.to_csv(date_path, index=False)
    folds_df.to_csv(fold_path, index=False)
    return date_path, fold_path


def save_common(ready_df: pd.DataFrame, folds_df: pd.DataFrame) -> tuple[Path, Path]:
    date_path = OUT_DIR / "model_ready_dates_ulsan.csv"
    fold_path = OUT_DIR / "cv_fold_boundaries_ulsan.csv"
    ready_df.to_csv(date_path, index=False)
    folds_df.to_csv(fold_path, index=False)
    return date_path, fold_path


def preview_csv(path: Path, n: int = 5) -> None:
    df = pd.read_csv(path)
    print(f"\nPreview: {path.relative_to(ROOT)}")
    print(df.head(n).to_string(index=False))


def print_fold_phase_stats(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"\nFold phase ranges and counts: {path.relative_to(ROOT)}")
    print(df.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_h = {}
    for h in HORIZONS:
        ready_df, folds_df, npz_path = export_one(h)
        per_h[h] = {
            "ready": ready_df,
            "folds": folds_df,
            "npz": npz_path,
        }

    ready_equal = (
        per_h[1]["ready"]["Date"].equals(per_h[3]["ready"]["Date"])
        and per_h[1]["ready"]["Date"].equals(per_h[5]["ready"]["Date"])
    )

    folds_equal = (
        per_h[1]["folds"].equals(per_h[3]["folds"]) and per_h[1]["folds"].equals(per_h[5]["folds"])
    )

    print("Source feature NPZ files:")
    for h in HORIZONS:
        print(f"  H{h}: {per_h[h]['npz'].relative_to(ROOT)}")

    print("\nUsable model-ready date index identical across H=1/3/5:", ready_equal)
    print("Fold boundary table identical across H=1/3/5:", folds_equal)

    exported: list[Path] = []
    if ready_equal and folds_equal:
        date_path, fold_path = save_common(per_h[1]["ready"], per_h[1]["folds"])
        exported.extend([date_path, fold_path])
    else:
        for h in HORIZONS:
            date_path, fold_path = save_with_horizon_suffix(h, per_h[h]["ready"], per_h[h]["folds"])
            exported.extend([date_path, fold_path])

    for p in exported:
        preview_csv(p, n=5)
        if "cv_fold_boundaries" in p.name:
            print_fold_phase_stats(p)


if __name__ == "__main__":
    main()