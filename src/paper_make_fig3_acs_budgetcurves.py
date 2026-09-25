#!/usr/bin/env python3
from __future__ import annotations
import re, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# ACS-ish figure defaults
# -----------------------------
def set_acs_rcparams():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,        # >= 0.5 pt guideline
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,           # embed TrueType
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Use a sans-serif; if Arial isn't available, matplotlib will fallback
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def guess_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def infer_ytrue(df: pd.DataFrame) -> str:
    c = guess_col(df, ["y_true", "true", "target", "label", "tnout_true", "tnout"])
    if c is not None:
        return c
    # fallback: first column containing "true"
    for col in df.columns:
        if "true" in col.lower():
            return col
    raise ValueError(f"Could not infer y-true column. Columns={list(df.columns)}")


def infer_yhat(df: pd.DataFrame) -> str:
    c = guess_col(df, ["y_hat", "yhat", "y_pred", "pred", "prediction", "tnout_pred", "tnout_hat"])
    if c is not None:
        return c
    # fallback: first column containing "pred"
    for col in df.columns:
        if "pred" in col.lower() or "hat" in col.lower():
            return col
    raise ValueError(f"Could not infer y-hat column. Columns={list(df.columns)}")


def infer_fold(df: pd.DataFrame) -> str | None:
    for cand in ["fold", "cv_fold", "split", "fold_id"]:
        c = guess_col(df, [cand])
        if c is not None:
            return c
    return None


def infer_score_col(df: pd.DataFrame, tau: int, prefer_prefixes) -> str | None:
    # prefer exact-ish patterns for p_tau15 / s_tau15 etc.
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    patterns = []
    for pref in prefer_prefixes:
        # accept p_tau15, p-tau15, p15, prob_tau15, score_tau15 etc
        patterns += [
            rf"^{pref}[_\-]?tau{tau}$",
            rf"^{pref}[_\-]?{tau}$",
            rf"^{pref}{tau}$",
        ]
    # also accept contains match (for more forgiving column names)
    contains = []
    for pref in prefer_prefixes:
        contains += [rf"{pref}.*tau{tau}", rf"{pref}.*{tau}"]

    for pat in patterns:
        for c, cl in zip(cols, lower):
            if re.fullmatch(pat, cl):
                return c

    for pat in contains:
        for c, cl in zip(cols, lower):
            if re.search(pat, cl):
                return c

    return None


def compute_budget_recall(df: pd.DataFrame, ytrue_col: str, score_col: str, tau: int, budgets: np.ndarray, fold_col: str | None):
    """
    Computes:
      - overall recall curve across all rows (pooled)
      - fold-wise recall curve if fold_col exists (dict fold -> recalls)
    Ranking: higher score = higher risk.
    """
    y = df[ytrue_col].to_numpy(dtype=float)
    score = df[score_col].to_numpy(dtype=float)
    event = (y >= tau).astype(int)

    def recall_for_subset(idx):
        n = len(idx)
        if n == 0:
            return np.full_like(budgets, np.nan, dtype=float)
        ev = event[idx]
        sc = score[idx]
        events = int(ev.sum())
        order = np.argsort(-sc)

        out = []
        for r in budgets:
            k = int(math.ceil(float(r) * n))
            k = max(k, 1)
            alarm_idx = order[:k]
            tp = int(ev[alarm_idx].sum())
            rec = tp / events if events > 0 else np.nan
            out.append(rec)
        return np.array(out, dtype=float)

    # pooled
    pooled = recall_for_subset(np.arange(len(df)))

    # fold-wise
    fold_curves = {}
    if fold_col is not None:
        for fval, g in df.groupby(fold_col):
            idx = g.index.to_numpy()
            fold_curves[fval] = recall_for_subset(idx)

    return pooled, fold_curves


def main():
    set_acs_rcparams()

    repo = Path(__file__).resolve().parents[1]
    pred_dir = repo / "results" / "predictions"

    # ---- inputs (H=5 v2)
    paths = {
        "Hybrid A (rank ensemble)": pred_dir / "hybrid_rank_ens_H5_v2_preds.csv",
        "TCN (BCR-TCN v1.1)":       pred_dir / "bcr_tcn_v11_H5_v2_preds.csv",
        "Persistence":              pred_dir / "persistence_v2_H5_preds.csv",
        # Optional: keep ENet for SI (set include_enet=True)
        "ElasticNet":               pred_dir / "enet_a0.1_l0.5_v2_H5_preds.csv",
    }

    include_enet = False  # main text: keep clean

    model_order = [
        "Hybrid A (rank ensemble)",
        "TCN (BCR-TCN v1.1)",
        "Persistence",
    ]
    if include_enet:
        model_order.append("ElasticNet")

    taus = [15, 16]
    budgets = np.round(np.linspace(0.01, 0.20, 20), 2)  # 1% ... 20%
    # ensure key budgets included
    budgets = np.unique(np.concatenate([budgets, np.array([0.05, 0.10])]))
    budgets.sort()

    # ---- load + compute curves
    curves = {}          # curves[(model, tau)] = pooled recall array
    fold_curves_hybrid = {}  # only for Hybrid A shading (optional)

    for name in model_order:
        p = paths[name]
        if not p.exists():
            raise FileNotFoundError(f"Missing predictions file: {p}")
        df = pd.read_csv(p)

        ytrue_col = infer_ytrue(df)
        yhat_col = infer_yhat(df)
        fold_col = infer_fold(df)

        for tau in taus:
            if "Hybrid" in name:
                # hybrid score usually stored as s_tau{tau}
                score_col = infer_score_col(df, tau, prefer_prefixes=["s", "score", "risk", "p", "prob"])
            elif "TCN" in name:
                # tcn score usually stored as p_tau{tau}
                score_col = infer_score_col(df, tau, prefer_prefixes=["p", "prob", "score", "s"])
            else:
                # regression baselines use yhat as risk score
                score_col = yhat_col

            if score_col is None:
                score_col = yhat_col  # hard fallback

            pooled, foldc = compute_budget_recall(df, ytrue_col, score_col, tau, budgets, fold_col)
            curves[(name, tau)] = pooled

            if name == "Hybrid A (rank ensemble)" and foldc:
                fold_curves_hybrid[tau] = foldc

    # ---- output folder (no overwrite)
    out_root = repo / "results" / "paper_artifacts" / f"{now_stamp()}_Fig3_ACS"
    out_fig = out_root / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    # ---- plot (two-panel, double-column width)
    # ACS widths: single ~3.33 in, double up to ~7 in. We'll use 7 in wide. :contentReference[oaicite:1]{index=1}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)

    # styles: do not rely on color only—use line styles too
    styles = {
        "Hybrid A (rank ensemble)": dict(linestyle="-"),
        "TCN (BCR-TCN v1.1)":       dict(linestyle="--"),
        "Persistence":              dict(linestyle=":"),
        "ElasticNet":               dict(linestyle="-."),
    }

    for ax, tau in zip(axes, taus):
        # random baseline
        ax.plot(budgets, budgets, linestyle="--", linewidth=1.0, color="0.6", label="Random baseline (recall = r)")

        for name in model_order:
            yrec = curves[(name, tau)]
            st = styles.get(name, {})
            ax.plot(budgets, yrec, label=name, **st)

        # highlight r=0.05 and 0.10 for Hybrid A
        for r0 in [0.05, 0.10]:
            i = int(np.where(np.isclose(budgets, r0))[0][0])
            y0 = curves[("Hybrid A (rank ensemble)", tau)][i]
            ax.scatter([r0], [y0], s=18, zorder=5)
            ax.annotate(f"{y0:.3f}", (r0, y0), textcoords="offset points", xytext=(4, 4))

        # optional fold-variability band for Hybrid A (min-max across folds)
        if tau in fold_curves_hybrid:
            mat = np.vstack([v for v in fold_curves_hybrid[tau].values()])
            lo = np.nanmin(mat, axis=0)
            hi = np.nanmax(mat, axis=0)
            ax.fill_between(budgets, lo, hi, alpha=0.12, linewidth=0)

        ax.set_title(f"( {'a' if tau==15 else 'b'} )  Threshold τ = {tau} mg/L")
        ax.set_xlabel("Alarm budget r (fraction of days alarmed)")
        ax.set_ylabel("Recall (TP / events)")
        ax.set_xlim(0.01, 0.20)
        ax.set_ylim(0.0, 0.65)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    # shared legend (top)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    # ---- save: vector PDF + high-DPI TIFF
    pdf_path = out_fig / "Fig3_budget_recall_curves_H5_v2.pdf"
    tif_path = out_fig / "Fig3_budget_recall_curves_H5_v2.tif"
    png_path = out_fig / "Fig3_budget_recall_curves_H5_v2.png"

    fig.savefig(pdf_path)                 # vector (preferred for line charts)
    fig.savefig(tif_path, dpi=600)        # >= 300 dpi for color art minimum :contentReference[oaicite:2]{index=2}
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print("DONE")
    print(f"Outputs: {out_root}")
    print(f"PDF:  {pdf_path}")
    print(f"TIFF: {tif_path}")
    print(f"PNG:  {png_path}")


if __name__ == "__main__":
    main()
