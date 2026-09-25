from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"


def resolve_npz_path(horizon: int) -> Path:
    candidates = [
        FEATURE_DIR / f"ulsan_H{horizon}_features_v2.npz",
        FEATURE_DIR / f"ulsan_H{horizon}_features.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No feature NPZ found for H={horizon}. Checked: {candidates}")


def build_assignment_matrix(sample_count: int, split_indices: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    matrix = np.zeros((len(split_indices), sample_count), dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(split_indices):
        matrix[fold_idx, train_idx] = 1
        matrix[fold_idx, test_idx] = 2
    return matrix


def export_for_horizon(horizon: int, n_splits: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    npz_path = resolve_npz_path(horizon)
    z = np.load(npz_path, allow_pickle=True)
    dates = z["dates"].astype(str)
    sample_count = len(dates)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    indices = list(splitter.split(np.arange(sample_count)))

    summary_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(indices, start=1):
        summary_rows.append(
            {
                "horizon": horizon,
                "fold": fold_id,
                "n_samples": sample_count,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_idx_start": int(train_idx.min()),
                "train_idx_end": int(train_idx.max()),
                "test_idx_start": int(test_idx.min()),
                "test_idx_end": int(test_idx.max()),
                "train_date_start": dates[train_idx.min()],
                "train_date_end": dates[train_idx.max()],
                "test_date_start": dates[test_idx.min()],
                "test_date_end": dates[test_idx.max()],
                "feature_npz": str(npz_path.relative_to(ROOT)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    matrix = build_assignment_matrix(sample_count=sample_count, split_indices=indices)
    assign_df = pd.DataFrame({"sample_idx": np.arange(sample_count), "date": dates})
    for fold_id in range(1, n_splits + 1):
        codes = matrix[fold_id - 1]
        labels = np.where(codes == 1, "train", np.where(codes == 2, "test", "unused"))
        assign_df[f"fold{fold_id}_set"] = labels

    assign_df.insert(0, "horizon", horizon)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(TABLE_DIR / f"blocked_cv_split_summary_H{horizon}.csv", index=False)
    assign_df.to_csv(TABLE_DIR / f"blocked_cv_assignment_matrix_H{horizon}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.0, 2.8), constrained_layout=True)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=plt.cm.get_cmap("viridis", 3), vmin=0, vmax=2)
    ax.set_yticks(np.arange(n_splits))
    ax.set_yticklabels([f"Fold {i}" for i in range(1, n_splits + 1)])
    tick_pos = np.linspace(0, sample_count - 1, num=6, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([dates[i] for i in tick_pos], rotation=20, ha="right")
    ax.set_xlabel("Chronological sample index / date")
    ax.set_title(f"Blocked TimeSeriesSplit strategy (H={horizon}, n_splits={n_splits})")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="unused", markerfacecolor=plt.cm.viridis(0 / 2), markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label="train", markerfacecolor=plt.cm.viridis(1 / 2), markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label="test", markerfacecolor=plt.cm.viridis(2 / 2), markersize=8),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    fig.savefig(FIG_DIR / f"blocked_cv_strategy_H{horizon}.png", dpi=300)
    fig.savefig(FIG_DIR / f"blocked_cv_strategy_H{horizon}.pdf")
    plt.close(fig)

    return summary_df, assign_df


def main() -> None:
    all_summary = []
    for horizon in [1, 3, 5]:
        summary_df, _ = export_for_horizon(horizon=horizon, n_splits=3)
        all_summary.append(summary_df)

    combined = pd.concat(all_summary, ignore_index=True)
    combined.to_csv(TABLE_DIR / "blocked_cv_split_summary_all_horizons.csv", index=False)
    print("Saved:", TABLE_DIR / "blocked_cv_split_summary_all_horizons.csv")
    for horizon in [1, 3, 5]:
        print("Saved:", TABLE_DIR / f"blocked_cv_split_summary_H{horizon}.csv")
        print("Saved:", TABLE_DIR / f"blocked_cv_assignment_matrix_H{horizon}.csv")
        print("Saved:", FIG_DIR / f"blocked_cv_strategy_H{horizon}.png")
        print("Saved:", FIG_DIR / f"blocked_cv_strategy_H{horizon}.pdf")


if __name__ == "__main__":
    main()
