#!/usr/bin/env python3
"""Generate Figure 3: Alarm-budget leaderboard across operating points.

This script builds a publication-ready 3x2 grid of heatmaps (H in rows, budget r in columns)
from one or more long-format CSV files containing alarm-budget metrics.

Example:
    python3 src/paper_make_fig3_alarm_budget_leaderboard.py \
        --csv results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv \
        --csv results/tables/leaderboard_alarm_by_operating_point.csv \
        --outdir results/paper_artifacts/fig3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COL_ALIASES: Dict[str, Sequence[str]] = {
    "model": ("model", "Model", "learner"),
    "H": ("H", "horizon"),
    "tau": ("tau", "threshold", "tau_mgL"),
    "r": ("r", "budget", "budget_r"),
    "recall": ("recall_mean", "recall", "Recall"),
    "precision": ("precision_mean", "precision", "Precision"),
}


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for name in aliases:
        if name in cols:
            return name
    for name in aliases:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _infer_h_from_source(source_name: str) -> int | None:
    matches = re.findall(r"(?:^|[^0-9A-Za-z])H\s*([135])(?:[^0-9A-Za-z]|$)", source_name, flags=re.IGNORECASE)
    if not matches:
        return None
    return int(matches[-1])


def normalize_columns(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    col_model = _find_column(df, COL_ALIASES["model"])
    col_h = _find_column(df, COL_ALIASES["H"])
    col_tau = _find_column(df, COL_ALIASES["tau"])
    col_r = _find_column(df, COL_ALIASES["r"])
    col_recall = _find_column(df, COL_ALIASES["recall"])
    col_precision = _find_column(df, COL_ALIASES["precision"])

    missing = []
    if col_model is None:
        missing.append("model")
    inferred_h = None
    if col_h is None:
        inferred_h = _infer_h_from_source(source_name)
        if inferred_h is None:
            missing.append("H/horizon")
    if col_tau is None:
        missing.append("tau/threshold")
    if col_r is None:
        missing.append("r/budget")
    if col_recall is None:
        missing.append("recall")
    if col_precision is None:
        missing.append("precision")

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "model": df[col_model].astype(str),
            "H": pd.to_numeric(df[col_h], errors="coerce") if col_h is not None else inferred_h,
            "tau": pd.to_numeric(df[col_tau], errors="coerce"),
            "r": pd.to_numeric(df[col_r], errors="coerce"),
            "recall_mean": pd.to_numeric(df[col_recall], errors="coerce"),
            "precision_mean": pd.to_numeric(df[col_precision], errors="coerce"),
        }
    )

    out = out.dropna(subset=["H", "tau", "r", "recall_mean", "precision_mean"]).copy()
    out["H"] = out["H"].astype(int)
    out["tau"] = out["tau"].astype(int)
    return out


def load_inputs(csv_paths: Sequence[str]) -> pd.DataFrame:
    all_frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        raw = pd.read_csv(path)
        norm = normalize_columns(raw, source_name=str(path))
        all_frames.append(norm)

    if not all_frames:
        raise ValueError("No input CSVs provided.")

    df = pd.concat(all_frames, ignore_index=True)
    if df.empty:
        raise ValueError("No valid rows after parsing input CSV(s).")
    return df


def _model_key(name: str) -> str:
    cleaned = str(name).lower().strip()
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def format_model_name(name: str) -> str:
    key = _model_key(name)

    # ENet patterns: enet_a0.01_l0.5, enet a0.01 l0.5, enet-a1.0-l0.2, etc.
    m = re.search(r"enet\s*a\s*([0-9]+(?:\.[0-9]+)?)\s*l\s*([0-9]+(?:\.[0-9]+)?)", key)
    if m:
        alpha = m.group(1)
        l1_ratio = m.group(2)
        return f"ENet {alpha}/{l1_ratio}"

    fixed_map = {
        "persistence": "Persistence",
        "hgbr": "HGBR",
        "tcn": "TCN",
        "bcr tcn v11": "BCR-TCN v1.1",
        "hybrid rank v2": "HybridRank v2",
        "hybrid rank ens": "HybridRank ens",
        "hybrid paper fixed": "Hybrid (paper)",
        "hybrid paper fixed weights": "Hybrid (paper)",
        "hybrid paper": "Hybrid (paper)",
    }
    if key in fixed_map:
        return fixed_map[key]

    return str(name).replace("_", " ")


def build_full_grid(
    df: pd.DataFrame,
    h_levels: Sequence[int],
    r_levels: Sequence[float],
    tau_levels: Sequence[int],
) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "H", "r", "tau"], as_index=False)[["recall_mean", "precision_mean"]]
        .mean()
    )

    models = sorted(grouped["model"].unique().tolist())
    full_idx = pd.MultiIndex.from_product(
        [models, h_levels, r_levels, tau_levels],
        names=["model", "H", "r", "tau"],
    )

    full = (
        grouped.set_index(["model", "H", "r", "tau"])
        .reindex(full_idx)
        .reset_index()
    )
    return full


def filter_models_by_mode(
    full_df: pd.DataFrame,
    mode: str,
    h_levels: Sequence[int],
    r_levels: Sequence[float],
    tau_levels: Sequence[int],
) -> Tuple[pd.DataFrame, List[str], str]:
    expected_n = len(h_levels) * len(r_levels) * len(tau_levels)
    counts = full_df.groupby("model")["recall_mean"].apply(lambda s: int(s.notna().sum()))

    if mode == "main":
        keep_models = counts[counts == expected_n].index.tolist()
        rule = f"complete grid required ({expected_n}/{expected_n} recall cells present)"
        if not keep_models:
            raise ValueError(
                "No models satisfy MAIN completeness rule. "
                f"Need all {expected_n} operating points per model."
            )
    else:
        keep_models = counts[counts >= 1].index.tolist()
        rule = "at least one recall value per model (missing shown as NA cells)"

    filtered = full_df[full_df["model"].isin(keep_models)].copy()
    return filtered, keep_models, rule


def compute_model_order(df: pd.DataFrame) -> List[str]:
    scores = (
        df.groupby("model", as_index=False)["recall_mean"]
        .mean()
        .sort_values("recall_mean", ascending=False)
    )
    return scores["model"].tolist()


def make_heatmap_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    metric: str,
    annotate: str,
    width: str,
    mode: str,
) -> None:
    h_levels = [1, 3, 5]
    r_levels = [0.05, 0.10]
    tau_levels = [15, 16, 17]

    full = build_full_grid(df, h_levels=h_levels, r_levels=r_levels, tau_levels=tau_levels)
    filtered, keep_models, _ = filter_models_by_mode(
        full,
        mode=mode,
        h_levels=h_levels,
        r_levels=r_levels,
        tau_levels=tau_levels,
    )

    model_order = compute_model_order(filtered)
    model_labels = [format_model_name(m) for m in model_order]

    figsize = (7.2, 7.2) if width == "double" else (4.2, 7.2)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    metric_vals = filtered[metric].to_numpy(dtype=float)
    vmin = 0.0
    vmax = np.nanmax(metric_vals)
    if not np.isfinite(vmax):
        vmax = 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    cmap = plt.cm.viridis.copy()
    if mode == "si":
        cmap.set_bad(color="0.92")
    last_im = None

    for i, h in enumerate(h_levels):
        for j, r in enumerate(r_levels):
            ax = axes[i, j]
            panel_df = filtered[(filtered["H"] == h) & np.isclose(filtered["r"], r)].copy()

            mat = np.full((len(model_order), len(tau_levels)), np.nan)
            rec_mat = np.full_like(mat, np.nan, dtype=float)
            pre_mat = np.full_like(mat, np.nan, dtype=float)

            agg_cols = list(dict.fromkeys([metric, "recall_mean", "precision_mean"]))
            grouped = panel_df.groupby(["model", "tau"], as_index=False)[agg_cols].mean()

            model_idx = {m: idx for idx, m in enumerate(model_order)}
            tau_idx = {t: idx for idx, t in enumerate(tau_levels)}

            for _, row in grouped.iterrows():
                m = row["model"]
                t = int(row["tau"])
                if m in model_idx and t in tau_idx:
                    y = model_idx[m]
                    x = tau_idx[t]
                    mat[y, x] = float(row[metric])
                    rec_mat[y, x] = float(row["recall_mean"])
                    pre_mat[y, x] = float(row["precision_mean"])

            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            last_im = im

            ax.set_xticks(np.arange(len(tau_levels)))
            ax.set_xticklabels([str(t) for t in tau_levels])
            ax.set_yticks(np.arange(len(model_order)))
            if j == 0:
                ax.set_yticklabels(model_labels)
                ax.set_ylabel("Model")
            else:
                ax.set_yticklabels([])

            if i == len(h_levels) - 1:
                ax.set_xlabel("Threshold τ (mg L$^{-1}$)")
            else:
                ax.set_xlabel("")
            ax.set_title(f"H={h}, r={r:.2f}")

            for yy in range(len(model_order)):
                for xx in range(len(tau_levels)):
                    rv = rec_mat[yy, xx]
                    pv = pre_mat[yy, xx]
                    if np.isnan(rv) and np.isnan(pv):
                        # MAIN mode has no missing by design; SI leaves NA cells blank/light gray.
                        continue

                    if annotate == "recall_precision":
                        r_txt = "NA" if np.isnan(rv) else f"{rv*100:.0f}%"
                        p_txt = "NA" if np.isnan(pv) else f"{pv*100:.0f}%"
                        txt = f"R {r_txt}\nP {p_txt}"
                        fontsize = 6.4
                    else:
                        txt = "NA" if np.isnan(rv) else f"{rv*100:.0f}%"
                        fontsize = 7.0

                    bg = mat[yy, xx]
                    if np.isnan(bg):
                        color = "black"
                    else:
                        rgba = cmap((bg - vmin) / (vmax - vmin))
                        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                        color = "white" if luminance < 0.5 else "black"

                    ax.text(xx, yy, txt, ha="center", va="center", fontsize=fontsize, color=color)

            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

    if last_im is None:
        raise RuntimeError("Heatmap generation failed: no panels were rendered.")

    cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.98, pad=0.02)
    cbar_label = "Recall" if metric == "recall_mean" else "Precision"
    cbar.set_label(cbar_label)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 3 alarm-budget leaderboard heatmaps.")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to long-format CSV (use repeated --csv for multiple files).",
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--metric",
        default="recall_mean",
        choices=["recall_mean", "precision_mean"],
        help="Metric used for heatmap color encoding.",
    )
    parser.add_argument(
        "--annotate",
        default="recall_only",
        choices=["recall_only", "recall_precision"],
        help="Annotation style inside cells.",
    )
    parser.add_argument(
        "--width",
        default="double",
        choices=["single", "double"],
        help="Figure width preset.",
    )
    parser.add_argument(
        "--mode",
        default="main",
        choices=["main", "si"],
        help="Model inclusion policy: main=complete 18-point grid only, si=all models with >=1 value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_inputs(args.csv)

    h_levels = [1, 3, 5]
    r_levels = [0.05, 0.10]
    tau_levels = [15, 16, 17]
    full = build_full_grid(df, h_levels=h_levels, r_levels=r_levels, tau_levels=tau_levels)
    filtered, keep_models, rule = filter_models_by_mode(
        full,
        mode=args.mode,
        h_levels=h_levels,
        r_levels=r_levels,
        tau_levels=tau_levels,
    )

    outdir = Path(args.outdir)
    out_pdf = outdir / "fig3_alarm_budget_leaderboard_clean.pdf"
    out_png = outdir / "fig3_alarm_budget_leaderboard_clean.png"

    make_heatmap_figure(
        df=filtered,
        out_pdf=out_pdf,
        out_png=out_png,
        metric=args.metric,
        annotate=args.annotate,
        width=args.width,
        mode=args.mode,
    )

    print(f"Rows read: {len(df)}")
    print(f"Mode: {args.mode}")
    print(f"Completeness rule: {rule}")
    print("Models included: " + ", ".join(format_model_name(m) for m in keep_models))
    print(f"Unique models included: {len(keep_models)}")
    print(f"Available H values: {sorted(df['H'].dropna().astype(int).unique().tolist())}")
    print(f"Available tau values: {sorted(df['tau'].dropna().astype(int).unique().tolist())}")
    r_vals = sorted(df['r'].dropna().unique().tolist())
    print("Available r values: [" + ", ".join(f"{r:.3f}" for r in r_vals) + "]")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
