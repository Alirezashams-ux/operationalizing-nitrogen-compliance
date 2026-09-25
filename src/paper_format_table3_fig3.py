#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

MODEL_LABEL = {
    "persistence": "Persistence",
    "enet_a0.1_l0.5": "ElasticNet (α=0.1, ℓ1=0.5)",
    "bcr_tcn_v11": "BCR-TCN v1.1",
    "hybrid_rank_ens": "Hybrid A (rank ensemble)",
}

MODEL_ORDER = ["persistence", "enet_a0.1_l0.5", "bcr_tcn_v11", "hybrid_rank_ens"]

MODEL_STYLE = {
    "persistence": {"color": "#2ca02c", "linestyle": ":", "linewidth": 2.0, "marker": "o"},
    "enet_a0.1_l0.5": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0, "marker": "o"},
    "bcr_tcn_v11": {"color": "#d62728", "linestyle": "-.", "linewidth": 2.0, "marker": "o"},
    "hybrid_rank_ens": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.2, "marker": "o"},
}

def df_to_latex_simple(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    colspec = "l" + "r" * (len(cols) - 1)
    row_end = r" \\" 
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(str(c) for c in cols) + row_end,
        "\\hline",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(str(v) for v in row) + row_end)
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)

def find_latest_artifact_dir() -> Path:
    base = Path("results/paper_artifacts")
    dirs = sorted(base.glob("*_H5_v2_Table3_Fig3"))
    if not dirs:
        raise FileNotFoundError("No results/paper_artifacts/*_H5_v2_Table3_Fig3 folders found.")
    return dirs[-1]

def main():
    in_dir = find_latest_artifact_dir()
    long_csv = in_dir / "tables" / "table3_alarm_budget_H5_v2_long.csv"
    if not long_csv.exists():
        raise FileNotFoundError(f"Missing {long_csv}")

    out_dir = in_dir / "paper_ready"
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(long_csv)
    df["model_label"] = df["model"].map(lambda x: MODEL_LABEL.get(x, x))
    df["r"] = df["r"].astype(float)
    df["tau"] = df["tau"].astype(int)

    # ---- Long table (journal-stable)
    df_long = df.copy()
    df_long = df_long.sort_values(["model", "tau", "r"])
    df_long["r"] = df_long["r"].round(2)
    df_long["precision"] = df_long["precision"].round(3)
    df_long["recall"] = df_long["recall"].round(3)
    df_long.to_csv(tables_dir / "Table3_main_long.csv", index=False)

    tex_long = df_to_latex_simple(
        df_long[["model_label","tau","r","n","k","events","tp","precision","recall"]],
        caption="Alarm-budget performance at $H=5$ (Ulsan, v2).",
        label="tab:alarm_budget_h5_v2_long",
    )
    (tables_dir / "Table3_main_long.tex").write_text(tex_long, encoding="utf-8")

    # ---- Wide table (Word-friendly)
    d = df.copy()
    d = d[d["model"].isin(MODEL_ORDER)].copy()
    d["model"] = pd.Categorical(d["model"], categories=MODEL_ORDER, ordered=True)
    d["r_str"] = d["r"].map(lambda x: f"{x:.2f}")
    d = d.sort_values(["model","tau","r"])

    wide = d.pivot_table(
        index="model", columns=["tau","r_str"],
        values=["tp","precision","recall"],
        aggfunc="first"
    )
    # flatten
    wide.columns = [f"tau{t}_r{r}_{m}" for (m,t,r) in wide.columns]
    wide = wide.reset_index()
    wide["Model"] = wide["model"].map(lambda x: MODEL_LABEL.get(x, x))
    wide = wide.drop(columns=["model"])
    # reorder: Model first
    cols = ["Model"] + [c for c in wide.columns if c != "Model"]
    wide = wide[cols]

    # rounding for display
    for c in wide.columns:
        if c.endswith("_tp"):
            wide[c] = wide[c].round(0).astype("Int64")
        elif "precision" in c or "recall" in c:
            wide[c] = wide[c].astype(float).round(3)

    wide.to_csv(tables_dir / "Table3_main_wide.tsv", sep="\t", index=False)

    tex_wide = df_to_latex_simple(
        wide,
        caption="Alarm-budget performance at $H=5$ (Ulsan, v2). Values are TP, precision, recall for each $(\\tau,r)$.",
        label="tab:alarm_budget_h5_v2_wide",
    )
    (tables_dir / "Table3_main_wide.tex").write_text(tex_wide, encoding="utf-8")

    # ---- Fig 3 (paper-ready)
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })

    r_vals = sorted(df["r"].unique())
    r_min, r_max = min(r_vals), max(r_vals)

    for tau in sorted(df["tau"].unique()):
        fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)

        # Random baseline for ranking policy
        ax.plot(r_vals, r_vals, color="#7f7f7f", linestyle="--", linewidth=1.8,
                label="Random baseline (recall = r)")

        for m in MODEL_ORDER:
            dm = df[(df["tau"] == tau) & (df["model"] == m)].sort_values("r")
            if dm.empty:
                continue
            style = MODEL_STYLE.get(m, {})
            ax.plot(dm["r"].values, dm["recall"].values, label=MODEL_LABEL.get(m, m), **style)

        ax.set_xlabel("Alarm budget r (fraction of days alarmed)")
        ax.set_ylabel("Recall (TP / events)")
        ax.set_title(f"Threshold τ = {tau} mg/L")
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(r_vals)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.25)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.5,
        )

        single_stem = f"Fig3_H5_v2_single_tau{tau}_budget_recall"
        fig.savefig(figs_dir / f"{single_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(figs_dir / f"{single_stem}.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # Combined 2-panel figure for paper layout (tau=15,16)
    tau_candidates = sorted(df["tau"].unique())
    panel_taus = [t for t in [15, 16] if t in tau_candidates]
    if len(panel_taus) < 2 and len(tau_candidates) >= 2:
        panel_taus = tau_candidates[:2]

    if len(panel_taus) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
        panel_tags = ["(a)", "(b)"]

        for idx, (ax, tau) in enumerate(zip(axes, panel_taus)):
            ax.plot(r_vals, r_vals, color="#7f7f7f", linestyle="--", linewidth=1.8,
                    label="Random baseline (recall = r)")

            for m in MODEL_ORDER:
                dm = df[(df["tau"] == tau) & (df["model"] == m)].sort_values("r")
                if dm.empty:
                    continue
                style = MODEL_STYLE.get(m, {})
                ax.plot(dm["r"].values, dm["recall"].values, label=MODEL_LABEL.get(m, m), **style)

            ax.set_title(f"{panel_tags[idx]} Threshold τ = {tau} mg/L")
            ax.set_xlim(r_min, r_max)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(r_vals)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Alarm budget r (fraction of days alarmed)")

        axes[0].set_ylabel("Recall (TP / events)")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        tau_tag = "_".join(str(t) for t in panel_taus)
        panel_stem = f"Fig3_H5_v2_panel_tau{tau_tag}_budget_recall"
        fig.savefig(figs_dir / f"{panel_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(figs_dir / f"{panel_stem}.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print("Wrote paper-ready outputs to:", out_dir)
    print("Tables:", tables_dir)
    print("Figures:", figs_dir)

if __name__ == "__main__":
    main()
