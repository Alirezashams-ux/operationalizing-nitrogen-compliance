#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    grad_csv = root / "results" / "tables" / "si_gradient_ve_summary_H5.csv"
    fit_csv = root / "results" / "tables" / "si_overfit_underfit_counts_H5.csv"

    out_png = root / "reports" / "FigureS.png"
    out_pdf = root / "reports" / "FigureS.pdf"

    grad = pd.read_csv(grad_csv).sort_values("fold")
    fit = pd.read_csv(fit_csv).sort_values(["fold", "fit_regime"])

    folds = grad["fold"].astype(int).tolist()
    x = np.arange(len(folds))
    labels = [f"Fold {f}" for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)

    # Panel A
    ax0 = axes[0]
    w = 0.34
    ax0.bar(x - w / 2, grad["exploding_epoch_rate"], width=w, label="Exploding-risk rate")
    ax0.bar(x + w / 2, grad["clip_fraction_mean"], width=w, label="Clip fraction")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Fraction")
    ax0.set_title("(A) Gradient stability summary", fontsize=16, pad=10)

    # Move legend outside panel at top-left and raise by exactly 2 cm.
    dy_inches = 2.0 / 2.54
    legend_transform = ax0.transAxes + mtransforms.ScaledTranslation(0.0, dy_inches, fig.dpi_scale_trans)
    ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, bbox_transform=legend_transform)

    # Panel B
    ax1 = axes[1]
    pivot = (
        fit.pivot(index="fold", columns="fit_regime", values="count_epochs")
        .fillna(0)
        .reindex(folds)
    )

    order = ["warmup", "stable_or_improving", "underfitting_risk", "overfitting_risk"]
    label_map = {
        "warmup": "warmup",
        "stable_or_improving": "stable or improving",
        "underfitting_risk": "underfitting risk",
        "overfitting_risk": "overfitting risk",
    }

    bottom = np.zeros(len(folds), dtype=float)
    for key in order:
        vals = pivot[key].to_numpy(dtype=float) if key in pivot.columns else np.zeros(len(folds), dtype=float)
        ax1.bar(x, vals, bottom=bottom, label=label_map[key])
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Epoch count")
    ax1.set_title("(B) Training-regime counts", fontsize=16, pad=10)
    ax1.legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)

    fig.subplots_adjust(top=0.76, bottom=0.12, wspace=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=1200)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
