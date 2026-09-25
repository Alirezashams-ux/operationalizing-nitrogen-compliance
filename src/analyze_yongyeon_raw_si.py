#!/usr/bin/env python3
"""Analyze the raw Yongyeon WWTP operator dataset.

Outputs
- An ACS-style descriptive statistics table (CSV)
- A data-quality + descriptive + inference statistics report (Markdown)

This script is designed to be reproducible and conservative: it does not
"clean"/alter the raw data beyond type coercion needed for analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover
    multipletests = None  # type: ignore
    adfuller = None  # type: ignore
    kpss = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "Ulsan_Yongyeon WWTP .csv"
DEFAULT_OUT_TABLE = ROOT / "results" / "tables" / "Ulsan_Yongyeon_WWTP_descriptive_stats_ACS.csv"
DEFAULT_OUT_REPORT = ROOT / "reports" / "Ulsan_Yongyeon_WWTP_raw_data_quality_SI_report.md"


UNIT_MAP: Dict[str, str] = {
    "day": "day-of-month",
    "Inflow": "m^3/day",
    "Outflow": "m^3/day",
    "BODin": "mg/L",
    "BODout": "mg/L",
    "TOCin": "mg/L",
    "TOCout": "mg/L",
    "SSin": "mg/L",
    "SSout": "mg/L",
    "TNin": "mg/L",
    "TNout": "mg/L",
    "TPin": "mg/L",
    "TPout": "mg/L",
    "Coliformin": "MPN/100 mL",
    "Coliformout": "MPN/100 mL",
}


SITE_DESCRIPTION_MD = (
    "The Yongyeon Wastewater Treatment Plant (WWTP) is a large municipal facility located in Ulsan, South Korea, "
    "in the Seosaeng-myeon area of Ulju-gun (approximate coordinates: 35.56233°N, 129.1269°E). "
    "It serves a substantial portion of Ulsan’s urban and industrial wastewater demand, with typical daily inflows "
    "in the range of approximately 200,000–250,000 m^3/day. The plant integrates co-treatment of municipal wastewater "
    "and food waste and is monitored under South Korea’s Water Quality Tele-Monitoring System (TMS), which enables "
    "real-time transmission of key effluent parameters (e.g., TOC, TN, TP) to K-eco for compliance monitoring." 
)


@dataclass
class DatasetOverview:
    n_rows: int
    n_cols: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    n_unique_dates: Optional[int]
    n_duplicate_dates: Optional[int]
    n_missing_dates_expected_daily: Optional[int]
    median_step_days: Optional[float]
    date_parse_failures: int


def _format_pct(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{100.0 * x:.2f}%"


def _safe_float(x: float, ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return ""
    if isinstance(x, (np.floating, float)):
        if abs(float(x)) >= 1e6:
            return f"{float(x):.3g}"
        return f"{float(x):.{ndigits}f}"
    return str(x)


def _robust_zscore_outlier_fraction(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 10:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return 0.0
    robust_z = 0.6745 * (values - median) / mad
    return float(np.mean(np.abs(robust_z) > 3.5))


def _iqr_outlier_fraction(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 10:
        return float("nan")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return float(np.mean((values < lo) | (values > hi)))


def _normality_tests(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    out: Dict[str, float] = {}
    if values.size < 8:
        return out

    # D'Agostino K^2 is the most standard for n>=20.
    if values.size >= 20:
        try:
            k2, p = stats.normaltest(values)
            out["normaltest_p"] = float(p)
        except Exception:
            pass

    # Shapiro is okay up to a few thousand; for large n it becomes overpowered.
    if values.size <= 5000:
        try:
            w, p = stats.shapiro(values)
            out["shapiro_p"] = float(p)
        except Exception:
            pass

    return out


def _stationarity_tests(values: np.ndarray) -> Dict[str, float]:
    """ADF and KPSS p-values (if statsmodels is available)."""
    values = values[np.isfinite(values)]
    out: Dict[str, float] = {}
    if values.size < 50:
        return out
    if adfuller is not None:
        try:
            res = adfuller(values, autolag="AIC", regression="c")
            out["adf_p"] = float(res[1])
        except Exception:
            pass
    if kpss is not None:
        try:
            stat, p, lags, crit = kpss(values, regression="c", nlags="auto")
            out["kpss_p"] = float(p)
        except Exception:
            pass
    return out


def _trend_tests(values: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(values) & np.isfinite(t)
    y = values[mask]
    x = t[mask]
    out: Dict[str, float] = {}
    if y.size < 30:
        return out

    try:
        lr = stats.linregress(x, y)
        out["trend_slope_per_day"] = float(lr.slope)
        out["trend_p"] = float(lr.pvalue)
        out["trend_r2"] = float(lr.rvalue**2)
    except Exception:
        pass

    try:
        tau, p = stats.kendalltau(x, y, nan_policy="omit")
        out["kendall_tau"] = float(tau)
        out["kendall_p"] = float(p)
    except Exception:
        pass

    return out


def _monthly_group_test(series: pd.Series, month: pd.Series) -> Dict[str, float]:
    """Kruskal-Wallis across months (robust) with epsilon-squared effect size."""
    out: Dict[str, float] = {}
    df = pd.DataFrame({"y": series, "m": month}).dropna()
    if df.shape[0] < 60:
        return out

    groups: List[np.ndarray] = []
    for m in range(1, 13):
        g = df.loc[df["m"] == m, "y"].to_numpy(dtype=float)
        if g.size >= 5:
            groups.append(g)

    if len(groups) < 3:
        return out

    try:
        h, p = stats.kruskal(*groups)
        out["kruskal_month_p"] = float(p)
        # epsilon-squared for Kruskal: (H - k + 1) / (n - k)
        n = df.shape[0]
        k = len(groups)
        eps2 = (h - k + 1.0) / (n - k) if (n - k) > 0 else float("nan")
        out["kruskal_month_eps2"] = float(eps2)
    except Exception:
        pass

    return out


def load_raw(path: Path) -> Tuple[pd.DataFrame, DatasetOverview]:
    df_raw = pd.read_csv(path)

    # Date parsing
    date_parse_failures = 0
    if "Date" in df_raw.columns:
        dt = pd.to_datetime(df_raw["Date"], format="%m/%d/%Y", errors="coerce")
        date_parse_failures = int(dt.isna().sum())
        df_raw["Date"] = dt
    else:
        dt = None

    # Overview metrics
    date_min = df_raw["Date"].min() if "Date" in df_raw.columns else None
    date_max = df_raw["Date"].max() if "Date" in df_raw.columns else None

    if "Date" in df_raw.columns:
        dates = df_raw["Date"].dropna()
        n_unique = int(dates.nunique())
        n_dupe_dates = int(dates.shape[0] - n_unique)

        dates_sorted = np.sort(dates.unique())
        if dates_sorted.size >= 2:
            steps = np.diff(dates_sorted).astype("timedelta64[D]").astype(float)
            median_step = float(np.median(steps))
        else:
            median_step = None

        # Expected daily calendar between min/max (compare at day resolution)
        if date_min is not None and date_max is not None:
            expected = pd.date_range(date_min, date_max, freq="D")
            expected_days = expected.values.astype("datetime64[D]")
            observed_days = pd.to_datetime(dates_sorted).values.astype("datetime64[D]")
            missing_expected = int(len(set(expected_days) - set(observed_days)))
        else:
            missing_expected = None
    else:
        n_unique = None
        n_dupe_dates = None
        missing_expected = None
        median_step = None

    overview = DatasetOverview(
        n_rows=int(df_raw.shape[0]),
        n_cols=int(df_raw.shape[1]),
        date_min=date_min if isinstance(date_min, pd.Timestamp) else None,
        date_max=date_max if isinstance(date_max, pd.Timestamp) else None,
        n_unique_dates=n_unique,
        n_duplicate_dates=n_dupe_dates,
        n_missing_dates_expected_daily=missing_expected,
        median_step_days=median_step,
        date_parse_failures=date_parse_failures,
    )

    return df_raw, overview


def build_descriptive_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Coerce non-date columns to numeric where possible
    for col in df.columns:
        if col == "Date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: List[Dict[str, object]] = []
    for col in df.columns:
        if col == "Date":
            continue

        s = df[col]
        values = s.to_numpy(dtype=float)
        mask = np.isfinite(values)
        n = int(mask.sum())
        missing_n = int((~mask).sum())
        missing_pct = float(missing_n / len(values)) if len(values) else float("nan")

        if n == 0:
            row = {
                "variable": col,
                "unit": "",
                "n": 0,
                "missing_n": missing_n,
                "missing_pct": 100.0 * missing_pct,
            }
            rows.append(row)
            continue

        v = values[mask]
        q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size >= 2 else float("nan")
        mn = float(np.min(v))
        mx = float(np.max(v))
        skew = float(stats.skew(v, bias=False)) if v.size >= 3 else float("nan")
        kurt = float(stats.kurtosis(v, fisher=True, bias=False)) if v.size >= 4 else float("nan")
        zeros_pct = float(np.mean(v == 0.0))
        neg_n = int(np.sum(v < 0.0))

        row = {
            "variable": col,
            "unit": UNIT_MAP.get(col, ""),
            "n": n,
            "missing_n": missing_n,
            "missing_pct": 100.0 * missing_pct,
            "mean": mean,
            "std": std,
            "mean_sd": f"{mean:.3f} ± {std:.3f}" if np.isfinite(std) else f"{mean:.3f}",
            "median": float(q[2]),
            "q25": float(q[1]),
            "q75": float(q[3]),
            "median_iqr": f"{q[2]:.3f} [{q[1]:.3f}–{q[3]:.3f}]",
            "min": mn,
            "max": mx,
            "min_max": f"{mn:.3f}–{mx:.3f}",
            "p01": float(q[0]),
            "p99": float(q[4]),
            "skew": skew,
            "kurtosis_excess": kurt,
            "zeros_pct": 100.0 * zeros_pct,
            "negatives_n": neg_n,
            "outliers_iqr_pct": 100.0 * _iqr_outlier_fraction(v),
            "outliers_robustz_pct": 100.0 * _robust_zscore_outlier_fraction(v),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # Put common ACS columns first
    preferred = [
        "variable",
        "unit",
        "n",
        "missing_n",
        "missing_pct",
        "mean",
        "std",
        "mean_sd",
        "median",
        "q25",
        "q75",
        "median_iqr",
        "min",
        "max",
        "min_max",
    ]
    remaining = [c for c in out.columns if c not in preferred]
    out = out[preferred + remaining]

    # Clean rounding for publication-ready CSV (avoid float artifacts)
    pct_cols = {"missing_pct", "zeros_pct", "outliers_iqr_pct", "outliers_robustz_pct"}
    int_cols = {"n", "missing_n", "negatives_n"}
    for c in out.columns:
        if c in int_cols or c in {"variable", "unit", "mean_sd", "median_iqr", "min_max"}:
            continue
        if c in pct_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    return out


def detect_likely_interpolated_points(
    df_sorted: pd.DataFrame,
    numeric_cols: Iterable[str],
    tol: float = 1e-12,
) -> Dict[str, List[str]]:
    """Heuristic: flag dates where y[t] == (y[t-1] + y[t+1]) / 2 exactly (within tol).

    This is a conservative indicator of linear interpolation in daily series.
    Returns ISO date strings per variable.
    """
    out: Dict[str, List[str]] = {}
    if "Date" not in df_sorted.columns:
        return out
    if df_sorted.shape[0] < 3:
        return out

    dates = df_sorted["Date"].to_numpy()
    for col in numeric_cols:
        y = pd.to_numeric(df_sorted[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue

        flagged: List[str] = []
        for i in range(1, len(y) - 1):
            if not (np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1])):
                continue
            mid = 0.5 * (y[i - 1] + y[i + 1])
            if abs(y[i] - mid) <= tol * max(1.0, abs(mid)):
                dt = pd.to_datetime(dates[i])
                if pd.notna(dt):
                    flagged.append(str(dt.date()))

        if flagged:
            out[col] = flagged
    return out


def analyze_quality_and_inference(df_raw: pd.DataFrame, overview: DatasetOverview) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Return a dict for report + a per-variable inference table."""
    df = df_raw.copy()
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # numeric coercion (keep Date)
    numeric_cols: List[str] = []
    for col in df.columns:
        if col == "Date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_cols.append(col)

    # Basic data-quality checks
    quality: Dict[str, object] = {}

    # Date integrity
    if "Date" in df.columns:
        dt = df["Date"]
        quality["date_parse_failures"] = overview.date_parse_failures
        quality["date_min"] = str(overview.date_min.date()) if overview.date_min is not None else ""
        quality["date_max"] = str(overview.date_max.date()) if overview.date_max is not None else ""
        quality["n_unique_dates"] = overview.n_unique_dates
        quality["n_duplicate_dates"] = overview.n_duplicate_dates
        quality["missing_dates_expected_daily"] = overview.n_missing_dates_expected_daily
        quality["median_step_days"] = overview.median_step_days

        # day column consistency (if present)
        if "day" in df.columns:
            day_mismatch = int((df["day"] != df["Date"].dt.day).sum())
            quality["day_col_mismatch_count"] = day_mismatch
    else:
        quality["date_parse_failures"] = None

    # Duplicated rows
    quality["duplicate_rows"] = int(df.duplicated().sum())

    # Missingness summary
    miss = df[numeric_cols].isna().mean().sort_values(ascending=False)
    quality["missingness_top"] = miss.head(10).to_dict()

    # Suspicious repeated values (high mode share)
    mode_share: Dict[str, float] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        vc = s.value_counts(dropna=True)
        share = float(vc.iloc[0] / s.shape[0])
        mode_share[col] = share
    quality["high_mode_share"] = {k: float(v) for k, v in sorted(mode_share.items(), key=lambda kv: kv[1], reverse=True)[:10]}

    # Flow consistency (if present)
    if "Inflow" in df.columns and "Outflow" in df.columns:
        inflow = df["Inflow"].to_numpy(dtype=float)
        outflow = df["Outflow"].to_numpy(dtype=float)
        mask = np.isfinite(inflow) & np.isfinite(outflow) & (inflow != 0)
        if mask.sum() > 0:
            ratio = outflow[mask] / inflow[mask]
            quality["outflow_inflow_ratio_median"] = float(np.median(ratio))
            quality["outflow_inflow_ratio_p01"] = float(np.quantile(ratio, 0.01))
            quality["outflow_inflow_ratio_p99"] = float(np.quantile(ratio, 0.99))
            quality["outflow_inflow_ratio_extreme_frac"] = float(np.mean((ratio < 0.5) | (ratio > 1.5)))

    # Negative values by column
    neg_counts = {c: int((df[c] < 0).sum()) for c in numeric_cols}
    quality["negative_values_by_col"] = {k: v for k, v in neg_counts.items() if v > 0}

    # Per-variable inference table
    if "Date" in df.columns:
        t = (df["Date"] - df["Date"].min()).dt.days.to_numpy(dtype=float)
        month = df["Date"].dt.month
    else:
        t = np.arange(df.shape[0], dtype=float)
        month = pd.Series([np.nan] * df.shape[0])

    inference_rows: List[Dict[str, object]] = []
    for col in numeric_cols:
        y = df[col].to_numpy(dtype=float)
        vmask = np.isfinite(y)
        if vmask.sum() == 0:
            continue
        v = y[vmask]

        row: Dict[str, object] = {"variable": col, "n": int(vmask.sum())}
        row.update(_normality_tests(v))
        row.update(_stationarity_tests(v))
        row.update(_trend_tests(y, t))
        row.update(_monthly_group_test(df[col], month))

        inference_rows.append(row)

    inference = pd.DataFrame(inference_rows).sort_values("variable")

    # Correlation + significance (Pearson & Spearman)
    corr_numeric = df[numeric_cols].copy()
    corr_numeric = corr_numeric.dropna(axis=1, how="all")

    corr_info: Dict[str, object] = {}
    if corr_numeric.shape[1] >= 2:
        cols = corr_numeric.columns.tolist()
        pearson_pairs: List[Tuple[str, str, float, float, int]] = []
        spearman_pairs: List[Tuple[str, str, float, float, int]] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = corr_numeric[cols[i]]
                b = corr_numeric[cols[j]]
                pair = pd.concat([a, b], axis=1).dropna()
                n_pair = int(pair.shape[0])
                if n_pair < 30:
                    continue
                try:
                    r, p = stats.pearsonr(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())
                    pearson_pairs.append((cols[i], cols[j], float(r), float(p), n_pair))
                except Exception:
                    pass
                try:
                    r, p = stats.spearmanr(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())
                    spearman_pairs.append((cols[i], cols[j], float(r), float(p), n_pair))
                except Exception:
                    pass

        def _top_with_fdr(pairs: List[Tuple[str, str, float, float, int]], method_name: str) -> List[Dict[str, object]]:
            if not pairs:
                return []
            ps = np.array([pval for _, _, _, pval, _ in pairs], dtype=float)
            if multipletests is None:
                order = np.argsort(ps)
                chosen = [pairs[i] for i in order[:15]]
                return [
                    {"x": a, "y": b, "corr": r, "p": p, "n": n}
                    for a, b, r, p, n in chosen
                ]

            reject, p_adj, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
            enriched = []
            for (a, b, r, p, n), padj, rej in zip(pairs, p_adj, reject):
                enriched.append((a, b, r, p, float(padj), bool(rej), n))
            enriched.sort(key=lambda t: (t[4], -abs(t[2])))
            top = enriched[:15]
            return [
                {"x": a, "y": b, "corr": r, "p": p, "p_fdr": padj, "reject_fdr_0p05": rej, "n": n}
                for a, b, r, p, padj, rej, n in top
            ]

        corr_info["pearson_top"] = _top_with_fdr(pearson_pairs, "pearson")
        corr_info["spearman_top"] = _top_with_fdr(spearman_pairs, "spearman")

    quality["correlations"] = corr_info

    # Likely interpolation flags (simple linear mid-point check)
    try:
        df_sorted = df.sort_values("Date") if "Date" in df.columns else df
        quality["likely_interpolated_midpoint_dates"] = detect_likely_interpolated_points(df_sorted, numeric_cols)
    except Exception:
        quality["likely_interpolated_midpoint_dates"] = {}

    return quality, inference


def render_report(
    input_path: Path,
    overview: DatasetOverview,
    descriptive: pd.DataFrame,
    quality: Dict[str, object],
    inference: pd.DataFrame,
) -> str:
    lines: List[str] = []

    lines.append("# Raw Dataset Data Quality + Descriptive & Inference Statistics (SI)")
    lines.append("")
    lines.append(f"**Dataset**: `{input_path}`  ")
    lines.append(f"**Generated**: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.append("## 0) Site and data provenance")
    lines.append("")
    lines.append(SITE_DESCRIPTION_MD)
    lines.append("")
    lines.append(
        "The dataset used here was provided directly by WWTP operators and consists of daily aggregated measurements "
        "from 2021-01-04 to 2023-10-31 (n ≈ 1031). Variables include inflow/outflow (m^3/day), BOD/TOC/SS/TN/TP (mg/L), "
        "and coliform bacteria (MPN/100 mL), measured for both influent and effluent as available."
    )
    lines.append("")

    lines.append("## 1) Dataset overview")
    lines.append("")
    lines.append(f"- Rows: {overview.n_rows}")
    lines.append(f"- Columns: {overview.n_cols}")
    if overview.date_min is not None and overview.date_max is not None:
        lines.append(f"- Date range (parsed from `Date`): {overview.date_min.date()} to {overview.date_max.date()}")
        if overview.n_unique_dates is not None:
            lines.append(f"- Unique dates: {overview.n_unique_dates} (duplicate-date rows: {overview.n_duplicate_dates})")
        if overview.median_step_days is not None:
            lines.append(f"- Median step between unique dates: {overview.median_step_days:.3f} days")
        if overview.n_missing_dates_expected_daily is not None:
            lines.append(f"- Missing calendar dates (expected daily between min/max): {overview.n_missing_dates_expected_daily}")
    lines.append(f"- Date parse failures: {overview.date_parse_failures}")
    lines.append("")

    lines.append("## 2) Variables")
    lines.append("")
    vars_list = [c for c in descriptive["variable"].tolist()]
    lines.append("Variables in file (excluding `Date`):")
    lines.append("")
    lines.append(", ".join(vars_list))
    lines.append("")

    lines.append("## 3) Data quality diagnostics")
    lines.append("")
    lines.append(f"- Duplicate full rows: {quality.get('duplicate_rows')}")
    if "day_col_mismatch_count" in quality:
        lines.append(f"- `day` column mismatches vs parsed Date day-of-month: {quality.get('day_col_mismatch_count')}")

    lines.append("")
    lines.append("### 3.1 Missingness (top 10 by fraction missing)")
    lines.append("")
    miss_top = quality.get("missingness_top", {}) or {}
    if isinstance(miss_top, dict) and miss_top:
        for k, v in miss_top.items():
            lines.append(f"- {k}: {_format_pct(float(v))}")
    else:
        lines.append("- (No missingness summary available.)")

    lines.append("")
    lines.append("### 3.2 Potential measurement plateaus / repeated values (top 10 mode shares)")
    lines.append("")
    mode = quality.get("high_mode_share", {}) or {}
    if isinstance(mode, dict) and mode:
        for k, v in mode.items():
            lines.append(f"- {k}: mode share ≈ {_format_pct(float(v))}")
    else:
        lines.append("- (No mode-share summary available.)")

    if "negative_values_by_col" in quality and quality["negative_values_by_col"]:
        lines.append("")
        lines.append("### 3.3 Negative values")
        lines.append("")
        for k, v in (quality["negative_values_by_col"] or {}).items():
            lines.append(f"- {k}: {v}")

    if "outflow_inflow_ratio_median" in quality:
        lines.append("")
        lines.append("### 3.4 Flow consistency (Outflow/Inflow)")
        lines.append("")
        lines.append(
            f"- Median ratio: {_safe_float(float(quality['outflow_inflow_ratio_median']), 3)}"
        )
        lines.append(
            f"- 1st–99th percentiles: {_safe_float(float(quality['outflow_inflow_ratio_p01']), 3)} to {_safe_float(float(quality['outflow_inflow_ratio_p99']), 3)}"
        )
        lines.append(
            f"- Extreme fraction (ratio < 0.5 or > 1.5): {_format_pct(float(quality['outflow_inflow_ratio_extreme_frac']))}"
        )

    lines.append("")
    lines.append("### 3.5 Likely interpolated entries (heuristic)")
    lines.append("")
    lines.append(
        "Flagged dates where a value equals the exact midpoint of adjacent days (a conservative indicator of linear interpolation). "
        "This does not prove interpolation, but highlights candidates for SI disclosure."
    )
    interp = quality.get("likely_interpolated_midpoint_dates", {})
    if isinstance(interp, dict) and interp:
        # Summarize counts and show any flags in the July 2023 window
        counts = {k: len(v) for k, v in interp.items() if isinstance(v, list)}
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if top:
            lines.append("- Top variables by count of flagged midpoint dates:")
            for k, v in top:
                lines.append(f"  - {k}: {v}")

        july_window = {"2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28"}
        window_hits: List[str] = []
        for k, dates in interp.items():
            if not isinstance(dates, list):
                continue
            hit = sorted(list(july_window.intersection(set(dates))))
            if hit:
                window_hits.append(f"- {k}: {', '.join(hit)}")
        if window_hits:
            lines.append("- Hits in 2023-07-25 to 2023-07-28 window:")
            lines.extend(window_hits)
        else:
            lines.append("- No midpoint flags were detected in 2023-07-25 to 2023-07-28 for this heuristic.")
    else:
        lines.append("- No midpoint-like interpolations were detected by this heuristic.")

    lines.append("")
    lines.append("## 4) Descriptive statistics (ACS-ready)")
    lines.append("")
    lines.append(
        "The full ACS-style descriptive statistics table is written as CSV. Units were not present in the raw file; the `unit` column is left blank for you to fill with the study’s reporting convention."
    )
    lines.append("")

    # Provide a compact view (mean±sd + median[IQR])
    compact = descriptive[["variable", "n", "missing_pct", "mean_sd", "median_iqr", "min_max"]].copy()
    compact["missing_pct"] = compact["missing_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    lines.append("Preview (compact):")
    lines.append("")
    lines.append(compact.to_markdown(index=False))

    lines.append("")
    lines.append("## 5) Inference statistics")
    lines.append("")
    lines.append("Inference suite includes:")
    lines.append("- Normality tests (D’Agostino K² and/or Shapiro) per variable")
    lines.append("- Stationarity tests (ADF and KPSS) where applicable")
    lines.append("- Trend tests (linear slope per day + Kendall tau)")
    lines.append("- Month-of-year distribution differences (Kruskal–Wallis)")
    lines.append("")

    if not inference.empty:
        cols_show = [c for c in [
            "variable",
            "n",
            "normaltest_p",
            "shapiro_p",
            "adf_p",
            "kpss_p",
            "trend_slope_per_day",
            "trend_p",
            "kendall_tau",
            "kendall_p",
            "kruskal_month_p",
            "kruskal_month_eps2",
        ] if c in inference.columns]

        view = inference[cols_show].copy()
        # format p-values
        for c in view.columns:
            if c.endswith("_p"):
                view[c] = view[c].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
        if "trend_slope_per_day" in view.columns:
            view["trend_slope_per_day"] = view["trend_slope_per_day"].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")
        if "kendall_tau" in view.columns:
            view["kendall_tau"] = view["kendall_tau"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        if "kruskal_month_eps2" in view.columns:
            view["kruskal_month_eps2"] = view["kruskal_month_eps2"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        lines.append("Per-variable inference results (all variables shown):")
        lines.append("")
        lines.append(view.to_markdown(index=False))
    else:
        lines.append("- (Inference table empty; no numeric variables were analyzable.)")

    lines.append("")
    lines.append("## 6) Correlation analysis (multiple-testing controlled)")
    lines.append("")
    corr = (quality.get("correlations") or {}) if isinstance(quality.get("correlations"), dict) else {}

    def _corr_table(items: List[Dict[str, object]]) -> str:
        if not items:
            return "(No eligible pairs with n>=30.)"
        dfc = pd.DataFrame(items)
        if "corr" in dfc.columns:
            dfc["corr"] = dfc["corr"].map(lambda x: f"{float(x):.3f}")
        for pcol in ["p", "p_fdr"]:
            if pcol in dfc.columns:
                dfc[pcol] = dfc[pcol].map(lambda x: f"{float(x):.3g}")
        return dfc.to_markdown(index=False)

    lines.append("### 6.1 Top Pearson correlations")
    lines.append("")
    lines.append(_corr_table(corr.get("pearson_top", [])))
    lines.append("")
    lines.append("### 6.2 Top Spearman correlations")
    lines.append("")
    lines.append(_corr_table(corr.get("spearman_top", [])))

    lines.append("")
    lines.append("## 7) Notes for SI / reproducibility")
    lines.append("")
    lines.append("- This report analyzes the raw operator-provided file with conservative type coercion only.")
    lines.append("- If you want me to align units/variable definitions to the manuscript text, point me to the exact SI section or table template (ACS SI varies by journal/format).")
    lines.append(f"- Script: `{(ROOT / 'src' / 'analyze_yongyeon_raw_si.py').relative_to(ROOT)}`")

    return "\n".join(lines) + "\n"


def main(
    input_path: Path = DEFAULT_INPUT,
    out_table: Path = DEFAULT_OUT_TABLE,
    out_report: Path = DEFAULT_OUT_REPORT,
) -> None:
    df_raw, overview = load_raw(input_path)

    descriptive = build_descriptive_table(df_raw)
    quality, inference = analyze_quality_and_inference(df_raw, overview)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    descriptive.to_csv(out_table, index=False)

    report_md = render_report(
        input_path=input_path,
        overview=overview,
        descriptive=descriptive,
        quality=quality,
        inference=inference,
    )
    out_report.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
