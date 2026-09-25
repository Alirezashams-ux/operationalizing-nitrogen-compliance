#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABEL = {
    "persistence": "Persistence",
    "enet_a0.1_l0.5": "ElasticNet",
    "bcr_tcn_v11": "BCR-TCN v1.1",
    "hybrid_rank_ens": "Hybrid A",
}

MODEL_ORDER = [
    "hybrid_rank_ens",
    "bcr_tcn_v11",
    "enet_a0.1_l0.5",
    "persistence",
]

MODEL_STYLE = {
    "hybrid_rank_ens": {"color": "#0B4F6C", "linestyle": "-", "marker": "o", "linewidth": 1.8, "markersize": 5.0},
    "bcr_tcn_v11": {"color": "#C9485B", "linestyle": "--", "marker": "s", "linewidth": 1.6, "markersize": 4.8},
    "enet_a0.1_l0.5": {"color": "#8A9A5B", "linestyle": "-.", "marker": "^", "linewidth": 1.6, "markersize": 4.8},
    "persistence": {"color": "#6B7280", "linestyle": ":", "marker": "D", "linewidth": 1.6, "markersize": 4.8},
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    in_csv = (
        repo
        / "results"
        / "paper_artifacts"
        / "20260219_183649_H5_v2_Table3_Fig3"
        / "tables"
        / "table3_alarm_budget_H5_v2_long.csv"
    )
    out_dir = (
        repo
        / "results"
        / "paper_artifacts"
        / "20260219_183649_H5_v2_Table3_Fig3"
        / "paper_ready"
        / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    required = {"model", "tau", "r", "recall"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Counterintuitive framing:
    # q = quiet-days fraction, m = missed-event fraction.
    df = df.copy()
    df["q"] = 1.0 - df["r"].astype(float)
    df["m"] = 1.0 - df["recall"].astype(float)

    taus = [15, 16]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True, constrained_layout=True)

    all_q = df[df["tau"].isin(taus)]["q"].astype(float)
    all_m = df[df["tau"].isin(taus)]["m"].astype(float)
    x_min = max(0.0, all_q.min() - 0.015)
    x_max = min(1.0, all_q.max() + 0.015)
    y_min = max(0.0, all_m.min() - 0.03)
    y_max = min(1.0, all_m.max() + 0.03)

    for ax, tau in zip(axes, taus):
        ax.plot([x_min, x_max], [x_min, x_max], color="#9CA3AF", linestyle="--", linewidth=1.2, label="Random (m=q)")

        panel = df[df["tau"] == tau].copy()
        for model in MODEL_ORDER:
            dm = panel[panel["model"] == model].sort_values("q")
            if dm.empty:
                continue
            style = MODEL_STYLE.get(model, {})
            ax.plot(dm["q"], dm["m"], label=MODEL_LABEL.get(model, model), **style)

        ax.set_xlabel("Quiet-days fraction q = 1 - r")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.22)
        ax.tick_params(axis="both", labelsize=8)
        ax.text(
            0.02,
            0.96,
            f"tau = {tau}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.8},
        )

    axes[0].set_ylabel("Missed-event fraction m = 1 - recall")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.3,
    )

    png_path = out_dir / "Fig3_H5_v2_silence_safety_frontier_tau15_16_1200dpi.png"
    pdf_path = out_dir / "Fig3_H5_v2_silence_safety_frontier_tau15_16.pdf"
    fig.savefig(png_path, dpi=1200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("Wrote:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
