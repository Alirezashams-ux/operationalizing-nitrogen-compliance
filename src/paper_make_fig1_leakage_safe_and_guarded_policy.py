#!/usr/bin/env python3
"""Generate Figure 1 for manuscript: leakage-safe workflow + guarded policy evidence.

Panel A: Real project workflow with explicit anti-leakage boundaries.
Panel B: Real H5 key operating-point comparison (paper-fixed hybrid vs guarded hybrid v2).

Usage:
  python3 src/paper_make_fig1_leakage_safe_and_guarded_policy.py \
    --csv results/tables/discussion_key_results.csv \
    --outdir results/figures/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


def _box(ax, x, y, w, h, text, fc="#f7f7f7", ec="#4c4c4c", lw=1.0, fs=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def _arrow(ax, x0, y0, x1, y1, color="#4c4c4c"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=11,
        lw=1.0,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def draw_panel_a(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.01, 0.98, "(A)", transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.text(
        0.06,
        0.95,
        "Leakage-safe forecasting framework used in this study",
        fontsize=10,
        fontweight="bold",
        ha="left",
    )

    y = 0.72
    w = 0.18
    h = 0.13
    x_list = [0.06, 0.29, 0.52, 0.75]

    _box(
        ax,
        x_list[0],
        y,
        w,
        h,
        "Raw daily data\nUlsan + weather\n(Date-indexed)",
        fc="#eef4ff",
        ec="#4169a8",
    )
    _box(
        ax,
        x_list[1],
        y,
        w,
        h,
        "Horizon-specific\nfeature build\n(H=1,3,5)",
        fc="#ecfaf0",
        ec="#3f8f5a",
    )
    _box(
        ax,
        x_list[2],
        y,
        w,
        h,
        "Blocked CV\n(TimeSeriesSplit)\nchronological",
        fc="#fff6ea",
        ec="#b6782c",
    )
    _box(
        ax,
        x_list[3],
        y,
        w,
        h,
        "Point + alarm\nevaluation\n(tau,r,k)",
        fc="#f8efff",
        ec="#7a4fa3",
    )

    for i in range(3):
        _arrow(ax, x_list[i] + w, y + h / 2, x_list[i + 1], y + h / 2)

    y2 = 0.47
    _box(
        ax,
        0.20,
        y2,
        0.24,
        0.12,
        "Train-only scaling\nand fitting\nwithin each fold",
        fc="#f2f2f2",
        ec="#595959",
    )
    _box(
        ax,
        0.56,
        y2,
        0.30,
        0.12,
        "Models: Persistence, ENet, HGBR,\nTCN, hybrid rank ensemble",
        fc="#f2f2f2",
        ec="#595959",
    )
    _arrow(ax, 0.44, y2 + 0.06, 0.56, y2 + 0.06)

    ax.text(0.06, 0.41, "Temporal leakage controls", fontsize=9.5, fontweight="bold")

    t_x0, t_y0, t_w, t_h = 0.06, 0.16, 0.88, 0.20
    ax.add_patch(Rectangle((t_x0, t_y0), t_w, t_h, ec="#777777", fc="#fcfcfc", lw=0.9))

    seg = np.array([0.00, 0.46, 0.67, 0.87, 1.00])
    colors = ["#d9ecff", "#e8f8e8", "#ffeecf", "#f6e6ff"]
    labels = ["Fold-1 train", "Fold-2 train", "Fold-3 train", "Held-out test window"]
    for i in range(4):
        sx = t_x0 + t_w * seg[i]
        sw = t_w * (seg[i + 1] - seg[i])
        ax.add_patch(Rectangle((sx, t_y0), sw, t_h, ec="#aaaaaa", fc=colors[i], lw=0.6))
        ax.text(sx + sw / 2, t_y0 + t_h / 2, labels[i], ha="center", va="center", fontsize=8)

    for b in seg[1:-1]:
        bx = t_x0 + t_w * b
        ax.plot([bx, bx], [t_y0 - 0.01, t_y0 + t_h + 0.01], color="#c73737", ls="--", lw=1.1)
    ax.text(
        0.07,
        0.11,
        "Dashed red boundaries indicate no future information is used for feature creation, scaling, or model fitting.",
        fontsize=8,
        color="#7a1e1e",
    )


def _load_key_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Context", "Model", "Tau", "Budget_r", "Recall"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    d = df.copy()
    d = d[d["Context"].astype(str).str.lower() == "h5_v2_guarded"].copy()
    d["Model"] = d["Model"].astype(str)
    d["Tau"] = pd.to_numeric(d["Tau"], errors="coerce")
    d["Budget_r"] = pd.to_numeric(d["Budget_r"], errors="coerce")
    d["Recall"] = pd.to_numeric(d["Recall"], errors="coerce")
    d = d.dropna(subset=["Tau", "Budget_r", "Recall"])

    keep_ops = {(15.0, 0.05), (15.0, 0.10), (16.0, 0.05), (16.0, 0.10)}
    d = d[d[["Tau", "Budget_r"]].apply(lambda r: (float(r[0]), float(r[1])) in keep_ops, axis=1)]
    d = d[d["Model"].isin(["hybrid_paper_fixed", "hybrid_rank_v2"])]

    if d.empty:
        raise ValueError("No matching rows found for H5_v2_guarded and target operating points.")
    return d


def draw_panel_b(ax, df_key: pd.DataFrame):
    ax.text(0.01, 0.98, "(B)", transform=ax.transAxes, va="top", ha="left", fontsize=11)

    ops = [(15.0, 0.05), (15.0, 0.10), (16.0, 0.05), (16.0, 0.10)]
    op_labels = ["tau15-r0.05", "tau15-r0.10", "tau16-r0.05", "tau16-r0.10"]

    rec_pf = []
    rec_v2 = []
    for tau, br in ops:
        sub = df_key[(df_key["Tau"] == tau) & (df_key["Budget_r"] == br)]
        pf = sub[sub["Model"] == "hybrid_paper_fixed"]["Recall"]
        v2 = sub[sub["Model"] == "hybrid_rank_v2"]["Recall"]
        rec_pf.append(float(pf.iloc[0]) if not pf.empty else np.nan)
        rec_v2.append(float(v2.iloc[0]) if not v2.empty else np.nan)

    x = np.arange(len(ops))
    width = 0.36

    c_pf = "#8d8d8d"
    c_v2 = "#1f77b4"

    ax.bar(x - width / 2, rec_pf, width=width, color=c_pf, label="Hybrid paper-fixed")
    ax.bar(x + width / 2, rec_v2, width=width, color=c_v2, label="Hybrid rank v2 (guarded)")

    for i, (a, b) in enumerate(zip(rec_pf, rec_v2)):
        if np.isfinite(a) and np.isfinite(b):
            delta = b - a
            y = max(a, b) + 0.012
            ax.text(i, y, f"dR={delta:+.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(op_labels, rotation=0)
    ax.set_ylabel("Recall at fixed alarm budget")
    ax.set_ylim(0, max(np.nanmax(rec_pf), np.nanmax(rec_v2)) + 0.08)
    ax.set_title("Guarded policy preserves low-budget behavior and improves selective higher-budget points", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    callout = (
        "Real project values from H5_v2_guarded key operating points\n"
        "Expected pattern: parity at r=0.05, gains at selected r=0.10 settings"
    )
    ax.text(0.02, 0.02, callout, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, color="#333333")


def make_figure(csv_path: Path, outdir: Path, stem: str = "Fig1_leakage_safe_and_guarded_policy"):
    df_key = _load_key_results(csv_path)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.8), constrained_layout=True, gridspec_kw={"height_ratios": [1.15, 1.0]})
    draw_panel_a(axes[0])
    draw_panel_b(axes[1], df_key)

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{stem}.png"
    out_pdf = outdir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"- {out_png}")
    print(f"- {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript Figure 1 with panels A and B.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/tables/discussion_key_results.csv"),
        help="CSV with H5_v2_guarded key operating points.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/figures/final"),
        help="Output directory for Figure 1 files.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="Fig1_leakage_safe_and_guarded_policy",
        help="Output filename stem (without extension).",
    )
    args = parser.parse_args()

    make_figure(args.csv, args.outdir, args.stem)


if __name__ == "__main__":
    main()
