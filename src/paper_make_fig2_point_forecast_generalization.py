#!/usr/bin/env python3
"""Generate manuscript Figure 2: point-forecast generalization across horizons.

Design intent (per manuscript transition):
- 3 side-by-side panels for MAE, RMSE, MASE.
- x-axis: forecast horizon H in {1, 3, 5} days.
- model-class lines: Persistence, ElasticNet, HGBR, TCN, Hybrid/Blend.
- highlight best point per horizon (lower is better).

Notes on availability:
- Full point-metric records for all five classes are only available at H=5.
- HGBR also has an H=3 prediction file; this script computes its point metrics.
- TCN and Hybrid/Blend point predictions are available only for H=5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HORIZONS = [1, 3, 5]
MODEL_ORDER = ["Persistence", "ElasticNet", "HGBR", "TCN", "Hybrid/Blend"]
MODEL_COLORS = {
    "Persistence": "#4D4D4D",
    "ElasticNet": "#1B9E77",
    "HGBR": "#D95F02",
    "TCN": "#7570B3",
    "Hybrid/Blend": "#E7298A",
}


def _compute_point_metrics(pred_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(pred_csv)
    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pred_csv} must include columns {sorted(required)}")

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


def _load_base_point_table(repo: Path) -> pd.DataFrame:
    path = repo / "results" / "tables" / "model_point_performance_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing point table: {path}")
    return pd.read_csv(path)


def _build_denominators(point_df: pd.DataFrame) -> Dict[int, float]:
    # Use persistence MAE/MASE1 as horizon-level denominator proxy.
    denoms: Dict[int, float] = {}

    # H1/H3 from main_linear block.
    for h in [1, 3]:
        row = point_df[
            (point_df["model"] == "persistence")
            & (point_df["source_group"] == "main_linear")
            & (point_df["horizon"] == h)
        ]
        if row.empty:
            raise ValueError(f"Could not find main_linear persistence row for H={h}")
        r = row.iloc[0]
        denoms[h] = float(r["MAE"] / r["MASE1"])

    # H5 from guarded block so TCN/Hybrid/HGBR comparisons share context.
    row_h5 = point_df[
        (point_df["model"] == "persistence")
        & (point_df["source_group"] == "hybrid_rank_v2_guarded")
        & (point_df["horizon"] == 5)
    ]
    if row_h5.empty:
        raise ValueError("Could not find guarded persistence row for H=5")
    r5 = row_h5.iloc[0]
    denoms[5] = float(r5["MAE"] / r5["MASE1"])

    return denoms


def _pick_elasticnet_rows(point_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # H1/H3: best MAE among main_linear ElasticNet variants.
    for h in [1, 3]:
        cand = point_df[
            point_df["model"].astype(str).str.startswith("enet")
            & (point_df["source_group"] == "main_linear")
            & (point_df["horizon"] == h)
        ].copy()
        if cand.empty:
            continue
        best = cand.sort_values("MAE", ascending=True).iloc[0]
        rows.append(
            {
                "model_class": "ElasticNet",
                "horizon": int(h),
                "MAE": float(best["MAE"]),
                "RMSE": float(best["RMSE"]),
                "MASE": float(best["MASE1"]),
            }
        )

    # H5: guarded ElasticNet summary row.
    cand_h5 = point_df[
        (point_df["model"] == "enet")
        & (point_df["source_group"] == "hybrid_rank_v2_guarded")
        & (point_df["horizon"] == 5)
    ]
    if not cand_h5.empty:
        best = cand_h5.iloc[0]
        rows.append(
            {
                "model_class": "ElasticNet",
                "horizon": 5,
                "MAE": float(best["MAE"]),
                "RMSE": float(best["RMSE"]),
                "MASE": float(best["MASE1"]),
            }
        )

    return rows


def assemble_figure2_data(repo: Path) -> pd.DataFrame:
    point_df = _load_base_point_table(repo)
    denoms = _build_denominators(point_df)

    rows: List[Dict[str, object]] = []

    # Persistence: H1/H3 from main_linear, H5 from guarded context.
    for h in [1, 3]:
        r = point_df[
            (point_df["model"] == "persistence")
            & (point_df["source_group"] == "main_linear")
            & (point_df["horizon"] == h)
        ].iloc[0]
        rows.append(
            {
                "model_class": "Persistence",
                "horizon": int(h),
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "MASE": float(r["MASE1"]),
            }
        )

    r5 = point_df[
        (point_df["model"] == "persistence")
        & (point_df["source_group"] == "hybrid_rank_v2_guarded")
        & (point_df["horizon"] == 5)
    ].iloc[0]
    rows.append(
        {
            "model_class": "Persistence",
            "horizon": 5,
            "MAE": float(r5["MAE"]),
            "RMSE": float(r5["RMSE"]),
            "MASE": float(r5["MASE1"]),
        }
    )

    rows.extend(_pick_elasticnet_rows(point_df))

    # HGBR: H5 from guarded table, H3 recomputed from predictions.
    hgbr_h5 = point_df[
        (point_df["model"] == "hgbr")
        & (point_df["source_group"] == "hybrid_rank_v2_guarded")
        & (point_df["horizon"] == 5)
    ]
    if not hgbr_h5.empty:
        r = hgbr_h5.iloc[0]
        rows.append(
            {
                "model_class": "HGBR",
                "horizon": 5,
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "MASE": float(r["MASE1"]),
            }
        )

    hgbr_h3_path = repo / "results" / "predictions" / "hgbr_optuna_H3_preds.csv"
    if hgbr_h3_path.exists():
        m = _compute_point_metrics(hgbr_h3_path)
        rows.append(
            {
                "model_class": "HGBR",
                "horizon": 3,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MASE": float(m["MAE"] / denoms[3]),
            }
        )

    # TCN: use calibrated H5 point predictions (mg/L scale).
    tcn_h5_path = repo / "results" / "predictions" / "bcr_tcn_H5_v2_preds.csv"
    if tcn_h5_path.exists():
        m = _compute_point_metrics(tcn_h5_path)
        rows.append(
            {
                "model_class": "TCN",
                "horizon": 5,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MASE": float(m["MAE"] / denoms[5]),
            }
        )

    # Hybrid/Blend: use H5 point-blend benchmark from guarded table.
    hyb_h5 = point_df[
        (point_df["model"] == "hybrid_point")
        & (point_df["source_group"] == "hybrid_rank_v2_guarded")
        & (point_df["horizon"] == 5)
    ]
    if not hyb_h5.empty:
        r = hyb_h5.iloc[0]
        rows.append(
            {
                "model_class": "Hybrid/Blend",
                "horizon": 5,
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "MASE": float(r["MASE1"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows assembled for Figure 2 data.")

    # Keep one row per model/horizon.
    df = df.sort_values(["model_class", "horizon"]).drop_duplicates(["model_class", "horizon"], keep="first")
    return df


def make_figure(df: pd.DataFrame, out_png: Path, out_pdf: Optional[Path] = None) -> None:
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

    metrics = [
        ("MAE", "MAE (mg L$^{-1}$)"),
        ("RMSE", "RMSE (mg L$^{-1}$)"),
        ("MASE", "MASE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)

    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]

        for model in MODEL_ORDER:
            sub = df[df["model_class"] == model].sort_values("horizon")
            if sub.empty:
                continue

            x = sub["horizon"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                ms=4.5,
                lw=1.4,
                color=MODEL_COLORS[model],
                alpha=0.95,
                label=model,
            )

        # Highlight best available model per horizon for this metric.
        for h in HORIZONS:
            cand = df[df["horizon"] == h].copy()
            if cand.empty:
                continue
            idx = cand[metric].idxmin()
            best = cand.loc[idx]
            ax.scatter(
                [h],
                [best[metric]],
                marker="*",
                s=85,
                color="#FFD166",
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon H (days)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, color="0.9", linewidth=0.6, alpha=0.9)

        panel_label = f"({chr(ord('A') + i)})"
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, va="top", ha="left", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))

    # Coverage note clarifies partial horizon availability for some classes.
    fig.text(
        0.01,
        -0.10,
        "Note: HGBR point forecasts available at H=3 and H=5; TCN and Hybrid/Blend point forecasts available at H=5.",
        fontsize=8,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=600, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 2 point-forecast generalization across horizons.")
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Output directory for 'Figure 2.png' and 'Figure 2.pdf'.",
    )
    parser.add_argument(
        "--save_table",
        default="results/tables/figure2_point_generalization_data.csv",
        help="Path to save the assembled long-format Figure 2 data table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]

    outdir = (repo / args.outdir).resolve()
    out_png = outdir / "Figure 2.png"
    out_pdf = outdir / "Figure 2.pdf"
    out_table = (repo / args.save_table).resolve()

    df = assemble_figure2_data(repo)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)

    make_figure(df, out_png=out_png, out_pdf=out_pdf)

    print("Saved Figure 2 data:", out_table)
    print("Saved Figure 2 PNG:", out_png)
    print("Saved Figure 2 PDF:", out_pdf)
    print("Rows used:")
    print(df.sort_values(["horizon", "model_class"]).to_string(index=False))


if __name__ == "__main__":
    main()
