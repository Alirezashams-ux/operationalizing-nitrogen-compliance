#!/usr/bin/env python3
"""Generate Figure 5 decision curves (ACS style) for H=5, tau=16.

This script plots TP(k) decision curves for alarm-budgeted early warning,
using mean TP and 95% CI across blocked time-series CV folds.

Example:
    python3 src/paper_make_fig5_decision_curve_acs.py \
      --csv results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Fig5_decision_curve_H5_tau16_data.csv \
      --outdir results/paper_artifacts/fig5_clean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_model(name: str) -> str:
    key = str(name).strip().lower()
    mapped: Dict[str, str] = {
        "hybrid (guarded)": "HybridRank v2",
        "hybrid guarded": "HybridRank v2",
        "hybrid (paper)": "Hybrid (paper)",
        "elasticnet": "ElasticNet",
        "persistence": "Persistence",
        "tcn": "TCN",
        "hgbr": "HGBR",
    }
    return mapped.get(key, str(name))


def load_curve_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    needed = ["model", "k_mean", "tp_mean", "tp_ci95"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    out = df.copy()
    out["model"] = out["model"].map(_normalize_model)
    out["k_mean"] = pd.to_numeric(out["k_mean"], errors="coerce")
    out["tp_mean"] = pd.to_numeric(out["tp_mean"], errors="coerce")
    out["tp_ci95"] = pd.to_numeric(out["tp_ci95"], errors="coerce")
    out = out.dropna(subset=["model", "k_mean", "tp_mean", "tp_ci95"]).copy()
    if out.empty:
        raise ValueError("No valid rows after parsing the decision-curve CSV.")
    return out


def load_prevalence_and_events(horizon: int = 5, tau: int = 16) -> tuple[float, int]:
    table2 = Path(
        "results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/"
        "Table2_exceedance_prevalence_random_baseline.csv"
    )
    if not table2.exists():
        raise FileNotFoundError(f"Could not find prevalence table: {table2}")

    df = pd.read_csv(table2)
    sel = df[(df["horizon"] == horizon) & (df["tau"] == tau)]
    if sel.empty:
        raise ValueError(f"No row found for horizon={horizon}, tau={tau} in {table2}")

    prevalence = float(sel.iloc[0]["prevalence"])
    events = int(sel.iloc[0]["events"])
    return prevalence, events


def make_figure(
    df: pd.DataFrame,
    prevalence: float,
    total_events: int,
    out_pdf: Path,
    out_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    # Order models by TP at max k (descending) for legible legend priority.
    idx_max_k = df.groupby("model")["k_mean"].idxmax()
    model_order = (
        df.loc[idx_max_k, ["model", "tp_mean"]]
        .sort_values("tp_mean", ascending=False)["model"]
        .tolist()
    )

    for model in model_order:
        sub = df[df["model"] == model].sort_values("k_mean")
        k = sub["k_mean"].to_numpy()
        tp = sub["tp_mean"].to_numpy()
        ci = sub["tp_ci95"].to_numpy()

        low = np.maximum(0, tp - ci)
        high = tp + ci

        line, = ax.plot(k, tp, linewidth=1.4, marker="o", markersize=3.0, label=model)
        ax.fill_between(k, low, high, alpha=0.18, linewidth=0, color=line.get_color())

    k_all = np.sort(df["k_mean"].unique())
    random_tp = k_all * prevalence
    ax.plot(
        k_all,
        random_tp,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label="Random baseline: TP(k)=k×prevalence",
    )
    ax.axhline(
        y=total_events,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="Total exceedances in test set (upper bound)",
    )

    ax.set_xlabel("Number of alarms, k")
    ax.set_ylabel("True positives captured, TP(k)")
    ax.set_title("H=5 d, τ=16 mg L$^{-1}$")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 5 decision curve (TP vs k).")
    parser.add_argument(
        "--csv",
        default="results/paper_artifacts/20260220_133021_ACS_Uncertainty_Decision/tables/Fig5_decision_curve_H5_tau16_data.csv",
        help="Input CSV with columns model,k_mean,tp_mean,tp_ci95.",
    )
    parser.add_argument(
        "--outdir",
        default="results/paper_artifacts/fig5_clean",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_curve_data(args.csv)
    prevalence, events = load_prevalence_and_events(horizon=5, tau=16)

    outdir = Path(args.outdir)
    out_pdf = outdir / "fig5_decision_curve_H5_tau16_clean.pdf"
    out_png = outdir / "fig5_decision_curve_H5_tau16_clean.png"

    make_figure(df, prevalence=prevalence, total_events=events, out_pdf=out_pdf, out_png=out_png)

    print(f"Rows read: {len(df)}")
    print(f"Models: {sorted(df['model'].unique().tolist())}")
    print(f"k range: {df['k_mean'].min():.1f} to {df['k_mean'].max():.1f}")
    print(f"Prevalence (H=5, tau=16): {prevalence:.6f}")
    print(f"Total exceedances (upper bound): {events}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
