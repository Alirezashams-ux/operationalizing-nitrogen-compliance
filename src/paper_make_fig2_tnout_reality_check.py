"""Generate Figure 2 (TNout reality check) for ACS ES&T Water manuscripts.

This script creates a 2-panel figure to show why average MAE alone is insufficient:
Panel (A) visualizes temporal dynamics and exceedance clustering; Panel (B) shows
distribution + ECDF with threshold base rates.

Usage:
    python3 src/paper_make_fig2_tnout_reality_check.py --csv <PATH> --outdir <OUTDIR>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


KNOWN_DATE_COLS = ["Date", "date", "timestamp", "Datetime", "datetime"]
KNOWN_TN_COLS = ["TNout", "TN_out", "TN", "TN_effluent", "Effluent_TN"]


def _datetime_success_rate(series: pd.Series) -> float:
    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    non_missing = series.notna().sum()
    if non_missing == 0:
        return 0.0
    return float(parsed.notna().sum() / non_missing)


def _numeric_success_rate(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0:
        return 0.0
    return float(numeric.notna().sum() / non_missing)


def _detect_date_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"--date_col '{override}' not found in CSV columns.")
        return override

    lower_map = {c.lower(): c for c in df.columns}
    for name in KNOWN_DATE_COLS:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    rates: list[Tuple[str, float]] = []
    for col in df.columns:
        rate = _datetime_success_rate(df[col])
        rates.append((col, rate))

    rates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_rate = rates[0]
    if best_rate >= 0.80:
        return best_col
    if best_rate >= 0.50:
        return best_col
    raise ValueError(
        "Could not detect a date column. Provide --date_col explicitly. "
        f"Best parse success was {best_rate:.2%} for '{best_col}'."
    )


def _detect_tn_col(df: pd.DataFrame, date_col: str, override: Optional[str] = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"--tn_col '{override}' not found in CSV columns.")
        return override

    candidates = [c for c in df.columns if c != date_col]
    lower_map = {c.lower(): c for c in candidates}

    for name in KNOWN_TN_COLS:
        hit = lower_map.get(name.lower())
        if hit is not None and _numeric_success_rate(df[hit]) > 0:
            return hit

    scored: list[Tuple[float, str]] = []
    for col in candidates:
        col_l = col.lower()
        num_rate = _numeric_success_rate(df[col])
        if num_rate <= 0:
            continue

        score = 0.0
        if "tn" in col_l and "out" in col_l:
            score += 3.0
        elif "tn" in col_l and "effluent" in col_l:
            score += 2.5
        elif "tn" in col_l:
            score += 1.5
        score += num_rate
        scored.append((score, col))

    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    raise ValueError(
        "Could not detect TNout column. Provide --tn_col explicitly. "
        "Expected known names or a numeric column containing TN/TNout cues."
    )


def load_data(
    csv_path: Path | str,
    date_col: Optional[str] = None,
    tn_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, int, int]:
    """Load CSV and return cleaned DataFrame with datetime index and TNout column.

    Returns:
        df_clean: DataFrame indexed by datetime with one column 'TNout' and
            rows where TNout is non-missing.
        detected_date_col: detected/selected date column name in raw CSV.
        detected_tn_col: detected/selected TNout column name in raw CSV.
        n_rows_raw: number of rows in raw CSV.
        n_non_missing_tnout: number of non-missing TNout points in cleaned data.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    if df_raw.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    detected_date_col = _detect_date_col(df_raw, date_col)
    detected_tn_col = _detect_tn_col(df_raw, detected_date_col, tn_col)

    date_parsed = pd.to_datetime(
        df_raw[detected_date_col], errors="coerce", infer_datetime_format=True
    )
    tn_numeric = pd.to_numeric(df_raw[detected_tn_col], errors="coerce")

    df = pd.DataFrame({"date": date_parsed, "TNout": tn_numeric})
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    df_clean = df.dropna(subset=["TNout"]).copy()

    if df_clean.empty:
        raise ValueError("No valid non-missing TNout values after parsing and cleaning.")

    return (
        df_clean,
        detected_date_col,
        detected_tn_col,
        int(len(df_raw)),
        int(df_clean["TNout"].notna().sum()),
    )


def compute_base_rates(series: pd.Series, taus: Iterable[float]) -> Dict[float, float]:
    """Compute exceedance base rates (%), i.e., mean(TNout >= tau)*100."""
    clean = series.dropna()
    return {float(tau): float((clean >= tau).mean() * 100.0) for tau in taus}


def make_figure(df: pd.DataFrame, taus: Iterable[float], out_pdf: Path, out_png: Path) -> None:
    """Create and save 2-panel TNout reality-check figure (PDF + PNG)."""
    taus = list(taus)
    tn = df["TNout"].dropna()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    line_main, = ax1.plot(
        tn.index,
        tn.values,
        lw=0.9,
        color="0.25",
        alpha=0.95,
        label="Daily TNout",
        zorder=1,
    )

    threshold_styles = {
        15: {"ls": "-", "marker": "o"},
        16: {"ls": "--", "marker": "s"},
        17: {"ls": ":", "marker": "^"},
    }

    marker_handles = []
    threshold_handles = []
    for i, tau in enumerate(taus):
        style = threshold_styles.get(int(tau), {"ls": "-.", "marker": "D"})
        col = f"C{i + 1}"
        ax1.axhline(
            tau,
            color=col,
            lw=1.0,
            ls=style["ls"],
            alpha=0.9,
            zorder=0,
        )
        threshold_handles.append(
            Line2D([0], [0], color=col, lw=1.0, ls=style["ls"], label=f"Threshold {int(tau)}")
        )

        mask = tn >= tau
        ax1.scatter(
            tn.index[mask],
            tn[mask],
            s=14,
            marker=style["marker"],
            facecolors="none",
            edgecolors=col,
            linewidths=0.7,
            alpha=0.85,
            zorder=2,
        )
        marker_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markeredgecolor=col,
                markerfacecolor="none",
                markeredgewidth=0.8,
                markersize=5,
                label=f"Exceedance ≥ {int(tau)}",
            )
        )

    ax1.set_ylabel("TNout (mg L$^{-1}$)")
    ax1.grid(True, color="0.9", lw=0.6, alpha=0.9)
    ax1.text(0.01, 0.97, "(A)", transform=ax1.transAxes, va="top", ha="left", fontsize=11)

    handles = [line_main] + threshold_handles + marker_handles
    ax1.legend(handles=handles, loc="upper right", frameon=False, ncol=2, handlelength=2.2)

    n_bins = min(40, max(18, int(np.sqrt(len(tn)))))
    counts, bins, _ = ax2.hist(
        tn.values,
        bins=n_bins,
        color="0.75",
        edgecolor="0.35",
        lw=0.6,
        alpha=0.9,
    )

    ax2_t = ax2.twinx()
    x_sorted = np.sort(tn.values)
    y_ecdf = np.arange(1, len(x_sorted) + 1, dtype=float) / len(x_sorted)
    ax2_t.plot(x_sorted, y_ecdf, lw=1.1, color="0.1", label="ECDF")
    ax2_t.set_ylabel("ECDF")
    ax2_t.set_ylim(0, 1.02)

    base_rates = compute_base_rates(tn, taus)
    y_top = ax2.get_ylim()[1]
    for i, tau in enumerate(taus):
        col = f"C{i + 1}"
        ls = threshold_styles.get(int(tau), {"ls": "-."})["ls"]
        ax2.axvline(tau, color=col, lw=1.0, ls=ls, alpha=0.95)

        x_offset = 0.05 + 0.35 * i
        text_y = y_top * (0.92 - i * 0.12)
        ax2.text(
            tau + x_offset,
            text_y,
            f"≥{int(tau)}: {base_rates[float(tau)]:.1f}%",
            fontsize=8.5,
            color=col,
            ha="left",
            va="center",
        )

    ax2.set_xlabel("TNout (mg L$^{-1}$)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, color="0.9", lw=0.6, alpha=0.9)
    ax2.text(0.01, 0.97, "(B)", transform=ax2.transAxes, va="top", ha="left", fontsize=11)

    ecdf_handle = Line2D([0], [0], color="0.1", lw=1.1, label="ECDF")
    hist_handle = Line2D([0], [0], color="0.5", lw=6, alpha=0.7, label="Histogram")
    ax2.legend(handles=[hist_handle, ecdf_handle], loc="upper left", frameon=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 2 TNout reality check.")
    parser.add_argument("--csv", required=True, help="Path to input daily WWTP CSV file.")
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for fig2_tnout_reality_check.{pdf,png}",
    )
    parser.add_argument("--date_col", default=None, help="Optional explicit date column name.")
    parser.add_argument("--tn_col", default=None, help="Optional explicit TNout column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taus = [15, 16, 17]

    outdir = Path(args.outdir)
    out_pdf = outdir / "fig2_tnout_reality_check.pdf"
    out_png = outdir / "fig2_tnout_reality_check.png"

    df, detected_date_col, detected_tn_col, n_rows, n_non_missing = load_data(
        args.csv,
        date_col=args.date_col,
        tn_col=args.tn_col,
    )

    series = df["TNout"]
    base_rates = compute_base_rates(series, taus)

    print(f"Detected date column: {detected_date_col}")
    print(f"Detected TN column:   {detected_tn_col}")
    print(f"Rows in CSV:          {n_rows}")
    print(f"Non-missing TNout:    {n_non_missing}")
    for tau in taus:
        print(f"Base rate (>= {tau} mg/L): {base_rates[float(tau)]:.1f}%")
    print(
        "TNout summary (mg/L): "
        f"min={series.min():.3f}, median={series.median():.3f}, max={series.max():.3f}"
    )

    make_figure(df, taus, out_pdf=out_pdf, out_png=out_png)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
