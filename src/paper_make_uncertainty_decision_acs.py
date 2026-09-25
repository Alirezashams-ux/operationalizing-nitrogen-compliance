#!/usr/bin/env python3
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_acs_rcparams() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ci95(series: pd.Series) -> float:
    vals = pd.Series(series, dtype="float64").dropna().to_numpy(dtype=float)
    n = len(vals)
    if n <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(n))


def fold_alarm_metrics(y_true: Any, score: Any, tau: int, budget_r: float) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(float(budget_r) * n)))
    event = (y_true >= float(tau)).astype(int)
    order = np.argsort(-score)
    alarm_idx = order[:k]
    tp = int(event[alarm_idx].sum())
    events = int(event.sum())
    return {
        "n": int(n),
        "k": int(k),
        "events": int(events),
        "tp": int(tp),
        "precision": float(tp / k if k > 0 else np.nan),
        "recall": float(tp / events if events > 0 else np.nan),
    }


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "fold" not in df.columns:
        raise ValueError(f"Missing 'fold' in {path}")
    if "y_true" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": "y_true"})
    if "y_true" not in df.columns:
        raise ValueError(f"Missing 'y_true' in {path}")
    df["fold"] = df["fold"].astype(str)
    return df


def compute_fold_rows(df: pd.DataFrame, score_col: str, tau: int, budget_r: float, model: str, horizon: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold, g in df.groupby("fold", sort=True):
        m = fold_alarm_metrics(
            y_true=g["y_true"].to_numpy(dtype=float),
            score=g[score_col].to_numpy(dtype=float),
            tau=tau,
            budget_r=budget_r,
        )
        rows.append(
            {
                "horizon": int(horizon),
                "model": model,
                "fold": str(fold),
                "tau": int(tau),
                "budget_r": float(budget_r),
                **m,
            }
        )
    return rows


def build_keypoint_uncertainty(repo: Path, h5_run_dir: Path, key_points: List[Tuple[int, int, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_dir = repo / "results" / "predictions"

    files = {
        1: {
            "persistence": pred_dir / "persistence_H1_preds.csv",
            "enet": pred_dir / "enet_a0.1_l0.5_H1_preds.csv",
        },
        3: {
            "persistence": pred_dir / "persistence_H3_preds.csv",
            "enet": pred_dir / "enet_a0.1_l0.5_H3_preds.csv",
            "hgbr": pred_dir / "hgbr_optuna_H3_preds.csv",
        },
    }

    h5_alarm = pd.read_csv(h5_run_dir / "tables" / "alarm_budget_metrics_H5.csv")
    h5_fold = h5_alarm[h5_alarm["scope"].astype(str).str.startswith("fold")].copy()
    h5_fold["horizon"] = 5
    h5_fold["tp"] = h5_fold["TP"]
    h5_fold["fold"] = h5_fold["scope"].astype(str)

    rows: List[Dict[str, Any]] = []

    for horizon, model_map in files.items():
        loaded = {name: load_pred(path) for name, path in model_map.items()}
        for h, tau, budget_r in key_points:
            if h != horizon:
                continue
            for model_name, df in loaded.items():
                rows.extend(compute_fold_rows(df, "y_pred", tau, budget_r, model_name, horizon))

    h5_models_keep = ["hybrid_rank_v2", "hybrid_paper_fixed", "persistence", "enet", "hgbr", "tcn"]
    for h, tau, budget_r in key_points:
        if h != 5:
            continue
        sub = h5_fold[
            (h5_fold["tau"] == int(tau))
            & (np.isclose(h5_fold["budget_r"].astype(float), float(budget_r)))
            & (h5_fold["model"].isin(h5_models_keep))
        ].copy()
        for _, r in sub.iterrows():
            rows.append(
                {
                    "horizon": 5,
                    "model": str(r["model"]),
                    "fold": str(r["fold"]),
                    "tau": int(r["tau"]),
                    "budget_r": float(r["budget_r"]),
                    "n": int(r["n"]),
                    "k": int(r["k"]),
                    "events": int(r["events"]),
                    "tp": int(r["tp"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                }
            )

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["horizon", "tau", "budget_r", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            k_mean=("k", "mean"),
            tp_mean=("tp", "mean"),
            tp_sd=("tp", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
    )
    summary["precision_ci95"] = fold_df.groupby(["horizon", "tau", "budget_r", "model"])["precision"].apply(ci95).to_numpy()
    summary["recall_ci95"] = fold_df.groupby(["horizon", "tau", "budget_r", "model"])["recall"].apply(ci95).to_numpy()
    summary["tp_ci95"] = fold_df.groupby(["horizon", "tau", "budget_r", "model"])["tp"].apply(ci95).to_numpy()

    return fold_df, summary


def plot_keypoint_uncertainty(summary: pd.DataFrame, out_fig: Path, key_points: List[Tuple[int, int, float]]) -> None:
    top_n = 3
    plot_rows = []
    for h, tau, r in key_points:
        block = summary[
            (summary["horizon"] == int(h))
            & (summary["tau"] == int(tau))
            & (np.isclose(summary["budget_r"].astype(float), float(r)))
        ].copy()
        if block.empty:
            continue
        order = np.lexsort(
            (
                -block["precision_mean"].to_numpy(dtype=float),
                -block["recall_mean"].to_numpy(dtype=float),
            )
        )
        block = block.iloc[order].head(top_n)
        label = f"H{h} τ={tau} r={r:.2f}"
        block["op_label"] = label
        plot_rows.append(block)

    d = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    if d.empty:
        return

    model_label = {
        "hybrid_rank_v2": "Hybrid (guarded)",
        "hybrid_paper_fixed": "Hybrid (paper-fixed)",
        "persistence": "Persistence",
        "enet": "ElasticNet",
        "hgbr": "HGBR",
        "tcn": "TCN",
    }
    d["model_label"] = d["model"].map(lambda x: model_label.get(x, x))

    op_order = [f"H{h} τ={tau} r={r:.2f}" for h, tau, r in key_points]
    model_order = ["Hybrid (guarded)", "ElasticNet", "Persistence", "HGBR", "TCN", "Hybrid (paper-fixed)"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
    width = 0.11
    x = np.arange(len(op_order))

    for i, mname in enumerate(model_order):
        dm = d[d["model_label"] == mname].copy()
        if dm.empty:
            continue
        idx_map = {lbl: j for j, lbl in enumerate(op_order)}
        dm["x"] = dm["op_label"].map(idx_map)
        off = (i - (len(model_order) - 1) / 2.0) * width

        axes[0].errorbar(
            dm["x"] + off,
            dm["recall_mean"],
            yerr=dm["recall_ci95"],
            fmt="o",
            capsize=2.0,
            label=mname,
        )
        axes[1].errorbar(
            dm["x"] + off,
            dm["precision_mean"],
            yerr=dm["precision_ci95"],
            fmt="o",
            capsize=2.0,
            label=mname,
        )

    axes[0].set_ylabel("Recall (mean ± 95% CI)")
    axes[1].set_ylabel("Precision (mean ± 95% CI)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(op_order, rotation=18, ha="right")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].set_title("Fold-uncertainty at key operating points")

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    axes[0].legend(uniq.values(), uniq.keys(), loc="upper left", frameon=False, ncol=3)

    fig.savefig(str(out_fig.with_suffix(".png")), dpi=600)
    fig.savefig(str(out_fig.with_suffix(".pdf")))
    plt.close(fig)


def decision_curve_h5(repo: Path, h5_run_dir: Path, tau: int, out_fig: Path) -> pd.DataFrame:
    pred = pd.read_csv(h5_run_dir / "hybrid_rank_v2_H5_preds.csv")

    score_map = {
        "Hybrid (guarded)": f"s_tau{tau}_hybrid",
        "ElasticNet": "y_pred_enet",
        "Persistence": "y_pred_persist",
        "HGBR": "y_pred_hgbr",
        "TCN": f"p_tau{tau}",
    }
    score_map = {k: v for k, v in score_map.items() if v in pred.columns}

    budgets = np.round(np.arange(0.02, 0.201, 0.02), 2)
    rows: List[Dict[str, Any]] = []

    for model_name, sc in score_map.items():
        for fold, g in pred.groupby("fold", sort=True):
            yv = g["y_true"].to_numpy(dtype=float)
            sv = g[sc].to_numpy(dtype=float)
            for r in budgets:
                m = fold_alarm_metrics(yv, sv, tau=tau, budget_r=float(r))
                rows.append(
                    {
                        "model": model_name,
                        "fold": str(fold),
                        "tau": int(tau),
                        "budget_r": float(r),
                        "k": int(m["k"]),
                        "tp": int(m["tp"]),
                        "recall": float(m["recall"]),
                    }
                )

    curve = pd.DataFrame(rows)
    agg = (
        curve.groupby(["model", "budget_r"], as_index=False)
        .agg(
            k_mean=("k", "mean"),
            tp_mean=("tp", "mean"),
            tp_sd=("tp", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
    )
    agg["tp_ci95"] = curve.groupby(["model", "budget_r"])["tp"].apply(ci95).to_numpy()
    agg["recall_ci95"] = curve.groupby(["model", "budget_r"])["recall"].apply(ci95).to_numpy()

    key = agg[np.isclose(agg["budget_r"].astype(float), 0.10)].copy()
    order = np.lexsort(
        (
            -key["recall_mean"].to_numpy(dtype=float),
            -key["tp_mean"].to_numpy(dtype=float),
        )
    )
    key = key.iloc[order]
    top_models = key.head(4)["model"].tolist()
    if len(top_models) < 4:
        extras = [m for m in ["Hybrid (guarded)", "ElasticNet", "Persistence", "HGBR", "TCN"] if m in agg["model"].unique()]
        for m in extras:
            if m not in top_models:
                top_models.append(m)
            if len(top_models) == 4:
                break

    fig, ax = plt.subplots(figsize=(3.35, 2.8), constrained_layout=True)
    for model_name in top_models:
        d = agg[agg["model"] == model_name].sort_values("k_mean")
        ax.errorbar(
            d["k_mean"],
            d["tp_mean"],
            yerr=d["tp_ci95"],
            marker="o",
            capsize=2,
            label=model_name,
        )

    ax.set_title(f"Decision curve at H=5, τ={tau} mg/L")
    ax.set_xlabel("Alarm count k")
    ax.set_ylabel("TP captured (mean ± 95% CI)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.savefig(str(out_fig.with_suffix(".png")), dpi=600)
    fig.savefig(str(out_fig.with_suffix(".pdf")))
    plt.close(fig)

    return agg


def normalized_h5_table(h5_run_dir: Path) -> pd.DataFrame:
    alarm = pd.read_csv(h5_run_dir / "tables" / "alarm_budget_metrics_H5.csv")
    fold = alarm[alarm["scope"].astype(str).str.startswith("fold")].copy()
    fold = fold[fold["tau"].isin([15, 16]) & fold["budget_r"].isin([0.05, 0.10])].copy()
    keep_models = ["hybrid_rank_v2", "hybrid_paper_fixed", "enet", "persistence", "hgbr", "tcn"]
    fold = fold[fold["model"].isin(keep_models)].copy()

    summary = (
        fold.groupby(["model", "tau", "budget_r"], as_index=False)
        .agg(
            folds=("scope", "nunique"),
            k_mean=("k", "mean"),
            tp_mean=("TP", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            tp_sd=("TP", "std"),
            precision_sd=("precision", "std"),
            recall_sd=("recall", "std"),
        )
    )
    summary["tp_ci95"] = fold.groupby(["model", "tau", "budget_r"])["TP"].apply(ci95).to_numpy()
    summary["precision_ci95"] = fold.groupby(["model", "tau", "budget_r"])["precision"].apply(ci95).to_numpy()
    summary["recall_ci95"] = fold.groupby(["model", "tau", "budget_r"])["recall"].apply(ci95).to_numpy()
    return summary.sort_values(["tau", "budget_r", "recall_mean"], ascending=[True, True, False])


def build_manifest(repo: Path, out_root: Path, h5_run_dir: Path) -> pd.DataFrame:
    items = [
        {
            "type": "Table",
            "id": "Table 1",
            "title": "Leakage-safe point forecasting performance across H=1/3/5",
            "path": "results/tables/leaderboard_point_by_horizon.csv",
            "status": "use",
            "note": "Main text Section 3.2",
        },
        {
            "type": "Table",
            "id": "Table 2",
            "title": "Exceedance prevalence and random-baseline recall by horizon/threshold",
            "path": str((out_root / "tables" / "Table2_exceedance_prevalence_random_baseline.csv").relative_to(repo)),
            "status": "new",
            "note": "Main text Section 3.1",
        },
        {
            "type": "Table",
            "id": "Table 3",
            "title": "Alarm-budget metrics by model at H=5 (core operating points)",
            "path": "results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/tables/Table3_main_long.csv",
            "status": "use",
            "note": "Main text Section 3.3",
        },
        {
            "type": "Table",
            "id": "Table 4",
            "title": "Normalized H5 comparison on shared folds/windows (fold mean ± CI)",
            "path": str((out_root / "tables" / "Table4_H5_normalized_context_comparison.csv").relative_to(repo)),
            "status": "new",
            "note": "Main text Section 3.4",
        },
        {
            "type": "Table",
            "id": "Table 5",
            "title": "Key operating-point uncertainty (precision/recall, fold CI)",
            "path": str((out_root / "tables" / "Table5_key_operating_points_fold_uncertainty.csv").relative_to(repo)),
            "status": "new",
            "note": "Main text Section 3.5",
        },
        {
            "type": "Figure",
            "id": "Figure 1",
            "title": "Study workflow and leakage-safe evaluation protocol",
            "path": "(prepare in manuscript graphics folder)",
            "status": "needed",
            "note": "Methods overview",
        },
        {
            "type": "Figure",
            "id": "Figure 2",
            "title": "Cross-horizon point-performance summary",
            "path": "(derive from results/tables/leaderboard_point_by_horizon.csv)",
            "status": "needed",
            "note": "Main text Section 3.2",
        },
        {
            "type": "Figure",
            "id": "Figure 3",
            "title": "Budget–recall curves at H=5 (τ=15,16)",
            "path": "results/paper_artifacts/20260219_183649_H5_v2_Table3_Fig3/paper_ready/figures/Fig3_H5_v2_panel_tau15_16_budget_recall.pdf",
            "status": "use",
            "note": "Main text Section 3.4",
        },
        {
            "type": "Figure",
            "id": "Figure 4",
            "title": "Fold-uncertainty bars at key operating points",
            "path": str((out_root / "figures" / "Fig4_key_operating_points_uncertainty.pdf").relative_to(repo)),
            "status": "new",
            "note": "Main text Section 3.5",
        },
        {
            "type": "Figure",
            "id": "Figure 5",
            "title": "Decision figure (H=5, τ=16): TP captured vs k with fold CI",
            "path": str((out_root / "figures" / "Fig5_decision_curve_H5_tau16_tp_vs_k.pdf").relative_to(repo)),
            "status": "new",
            "note": "Decision-oriented summary",
        },
    ]
    return pd.DataFrame(items)


def build_exceedance_prevalence_table(repo: Path, out_csv: Path) -> None:
    pred_dir = repo / "results" / "predictions"
    sources = {
        1: pred_dir / "persistence_H1_preds.csv",
        3: pred_dir / "persistence_H3_preds.csv",
        5: pred_dir / "persistence_H5_preds.csv",
    }
    rows = []
    for h, p in sources.items():
        df = pd.read_csv(p)
        y = df["y_true"].to_numpy(dtype=float)
        n = len(y)
        for tau in [15, 16, 17]:
            events = int(np.sum(y >= tau))
            prevalence = float(events / n) if n > 0 else np.nan
            rows.append(
                {
                    "horizon": h,
                    "tau": tau,
                    "n_test": n,
                    "events": events,
                    "prevalence": prevalence,
                    "random_recall_at_r005": 0.05,
                    "random_recall_at_r010": 0.10,
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    set_acs_rcparams()
    repo = Path(__file__).resolve().parents[1]

    h5_runs = sorted((repo / "results" / "hybrid_rank_v2_runs").glob("*_H5_hybrid_rank_v2"))
    if not h5_runs:
        raise FileNotFoundError("No H5 hybrid runs found under results/hybrid_rank_v2_runs")

    # Prefer the freshly normalized run if present, otherwise latest.
    preferred = [d for d in h5_runs if d.name.startswith("20260220_131820")]
    h5_run_dir = preferred[-1] if preferred else h5_runs[-1]

    out_root = repo / "results" / "paper_artifacts" / f"{now_stamp()}_ACS_Uncertainty_Decision"
    out_tables = out_root / "tables"
    out_figs = out_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    key_points = [
        (1, 15, 0.05),
        (1, 16, 0.10),
        (3, 15, 0.10),
        (3, 16, 0.05),
        (5, 15, 0.10),
        (5, 16, 0.10),
    ]

    fold_df, summary_df = build_keypoint_uncertainty(repo, h5_run_dir, key_points)
    fold_df.to_csv(out_tables / "Table5_key_operating_points_fold_rows.csv", index=False)
    summary_df.to_csv(out_tables / "Table5_key_operating_points_fold_uncertainty.csv", index=False)

    plot_keypoint_uncertainty(summary_df, out_figs / "Fig4_key_operating_points_uncertainty", key_points)

    decision_df = decision_curve_h5(repo, h5_run_dir, tau=16, out_fig=out_figs / "Fig5_decision_curve_H5_tau16_tp_vs_k")
    decision_df.to_csv(out_tables / "Fig5_decision_curve_H5_tau16_data.csv", index=False)

    norm_df = normalized_h5_table(h5_run_dir)
    norm_df.to_csv(out_tables / "Table4_H5_normalized_context_comparison.csv", index=False)

    build_exceedance_prevalence_table(repo, out_tables / "Table2_exceedance_prevalence_random_baseline.csv")

    manifest_df = build_manifest(repo, out_root, h5_run_dir)
    manifest_df.to_csv(out_tables / "paper_figures_tables_manifest.csv", index=False)

    print("H5 normalized context run:", h5_run_dir)
    print("Wrote ACS uncertainty + decision artifacts to:", out_root)
    print("-", out_figs / "Fig4_key_operating_points_uncertainty.pdf")
    print("-", out_figs / "Fig5_decision_curve_H5_tau16_tp_vs_k.pdf")
    print("-", out_tables / "Table4_H5_normalized_context_comparison.csv")
    print("-", out_tables / "Table5_key_operating_points_fold_uncertainty.csv")
    print("-", out_tables / "paper_figures_tables_manifest.csv")


if __name__ == "__main__":
    main()
