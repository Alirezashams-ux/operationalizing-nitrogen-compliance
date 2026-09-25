#!/usr/bin/env python3
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# CONFIG (edit only if needed)
# -------------------------
REPO = Path(".").resolve()
H = 5
TAUS = [15, 16]
BUDGETS = [0.05, 0.10]

PRED_FILES = {
    "persistence": REPO / f"results/predictions/persistence_v2_H{H}_preds.csv",
    "enet_a0.1_l0.5": REPO / f"results/predictions/enet_a0.1_l0.5_v2_H{H}_preds.csv",
    "bcr_tcn_v11": REPO / f"results/predictions/bcr_tcn_v11_H{H}_v2_preds.csv",
    "hybrid_rank_ens": REPO / f"results/predictions/hybrid_rank_ens_H{H}_v2_preds.csv",
}

MODEL_ORDER = ["persistence", "enet_a0.1_l0.5", "bcr_tcn_v11", "hybrid_rank_ens"]


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_y_cols(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    y_true = None
    y_hat = None
    for c in ["y_true", "y", "true", "tnout_true", "target", "label"]:
        if c in cols:
            y_true = cols[c]
            break
    for c in ["y_hat", "yhat", "pred", "y_pred", "prediction", "tnout_pred", "tnout_hat"]:
        if c in cols:
            y_hat = cols[c]
            break
    if y_true is None or y_hat is None:
        raise ValueError(f"Cannot infer y_true/y_hat columns. Columns={list(df.columns)}")
    return y_true, y_hat


def score_col_for_tau(df: pd.DataFrame, tau: int, default_col: str) -> str:
    # If the file has tau-specific scores (p_tau16, s_tau16, etc.), use them; otherwise use y_hat.
    low = [c.lower() for c in df.columns]
    patterns = [
        f"s_tau{tau}", f"s-tau{tau}", f"score_tau{tau}", f"p_tau{tau}", f"prob_tau{tau}",
        f"p{tau}", f"prob{tau}", f"score{tau}"
    ]
    for p in patterns:
        for i, c in enumerate(low):
            if p == c:
                return df.columns[i]
    # fallback
    return default_col


def compute_alarm_metrics(pred_path: Path, model: str) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    df = pd.read_csv(pred_path)
    y_true, y_hat = infer_y_cols(df)
    n = len(df)

    rows = []
    for tau in TAUS:
        score_col = score_col_for_tau(df, tau=tau, default_col=y_hat)
        scores = df[score_col].astype(float).values
        y = df[y_true].astype(float).values
        event = (y >= tau).astype(int)
        events = int(event.sum())

        order = np.argsort(-scores)  # high score = high risk
        for r in BUDGETS:
            k = int(math.ceil(r * n))
            alarm_idx = order[:k]
            tp = int(event[alarm_idx].sum())
            precision = tp / k if k > 0 else np.nan
            recall = tp / events if events > 0 else np.nan
            rows.append(
                dict(
                    model=model,
                    tau=tau,
                    r=r,
                    n=n,
                    k=k,
                    events=events,
                    tp=tp,
                    precision=precision,
                    recall=recall,
                    source="recomputed_from_preds",
                )
            )
    return pd.DataFrame(rows)


def make_outputs(df_long: pd.DataFrame, out_root: Path) -> None:
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Long table (stable for journal)
    df_long = df_long.sort_values(["model", "tau", "r"]).reset_index(drop=True)
    df_long.to_csv(tables_dir / "table3_alarm_budget_H5_v2_long.csv", index=False)

    # Wide table (Word-friendly)
    d = df_long.copy()
    d["r_str"] = d["r"].map(lambda x: f"{x:.2f}")
    wide = d.pivot_table(
        index="model",
        columns=["tau", "r_str"],
        values=["k", "events", "tp", "precision", "recall"],
        aggfunc="first",
    )
    wide.columns = [f"tau{t}_r{r}_{m}" for (m, t, r) in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(tables_dir / "table3_alarm_budget_H5_v2_wide.tsv", sep="\t", index=False)

    # Fig 3: budget–recall curves (one plot per tau)
    for tau in TAUS:
        plt.figure()
        for m in MODEL_ORDER:
            dm = df_long[(df_long["tau"] == tau) & (df_long["model"] == m)].sort_values("r")
            if dm.empty:
                continue
            plt.plot(dm["r"].values, dm["recall"].values, marker="o", label=m)
        plt.xlabel("Alarm budget r (fraction of days alarmed)")
        plt.ylabel(f"Recall for exceedance TNout ≥ {tau} mg/L")
        plt.title(f"Budget–recall curve (H=5, Ulsan v2, τ={tau})")
        plt.xticks(BUDGETS)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"fig3_budget_curves_H5_v2_tau{tau}.png", dpi=300)
        plt.savefig(figs_dir / f"fig3_budget_curves_H5_v2_tau{tau}.pdf")
        plt.close()


def main():
    out_root = REPO / "results" / "paper_artifacts" / f"{stamp()}_H{H}_v2_Table3_Fig3"
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model, path in PRED_FILES.items():
        all_rows.append(compute_alarm_metrics(path, model=model))
    df_long = pd.concat(all_rows, ignore_index=True)

    make_outputs(df_long, out_root)

    print("DONE.")
    print("Outputs:", out_root)
    print("Table 3:", out_root / "tables" / "table3_alarm_budget_H5_v2_long.csv")
    print("Fig 3:", out_root / "figures" / "fig3_budget_curves_H5_v2_tau15.png", "and tau16")


if __name__ == "__main__":
    main()
