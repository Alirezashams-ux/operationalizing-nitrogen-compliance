#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SI_TABLES = ROOT / "reports" / "main_manuscript_assets" / "SI" / "tables"
SI_FIGS = ROOT / "reports" / "main_manuscript_assets" / "SI" / "figures"


MODEL_KEY_MAP = {
    "Persistence": "persistence",
    "ElasticNet": "enet_a0.1_l0.5",
    "HGBR": "hgbr",
}

MODEL_LABEL = {
    "persistence": "Persistence",
    "enet_a0.1_l0.5": "ElasticNet",
    "hgbr": "HGBR",
}


def _standardize_columns(df: pd.DataFrame, site: str) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if str(c).startswith("Unnamed")], errors="ignore")

    if site == "ulsan":
        pass
    elif site == "seoul":
        rename_map = {
            "Average Temperature": "temp_mean_c",
            "Precipitation": "precip_total_mm",
            "Average Wind Speed": "wind_mean_ms",
            "Highest Wind Speed": "wind_max_ms",
            "Average Humidity": "rh_mean_pct",
            "Sunshine Duration": "sunshine_total_hr",
            "Total Solar Radiation": "solar_rad_total_mj_m2",
            "Gap of daily temperature": "temp_range_c",
        }
        out = out.rename(columns=rename_map)
    else:
        raise ValueError(f"Unknown site: {site}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for c in ["TNout", "Inflow", "TNin", "TOCin", "BODin", "temp_mean_c", "precip_total_mm"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    a = np.sort(a)
    b = np.sort(b)
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, pooled, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, pooled, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _safe_stats(x: pd.Series) -> dict:
    n_total = int(len(x))
    xf = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xf)
    n_valid = int(mask.sum())
    missing_rate = float(1.0 - n_valid / n_total) if n_total > 0 else np.nan

    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "missing_rate": missing_rate,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }

    vals = xf[mask]
    q25 = float(np.quantile(vals, 0.25))
    q75 = float(np.quantile(vals, 0.75))
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "missing_rate": missing_rate,
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if n_valid > 1 else 0.0,
        "median": float(np.median(vals)),
        "iqr": float(q75 - q25),
    }


def _build_ulsan_point_baseline() -> pd.DataFrame:
    path = ROOT / "results" / "tables" / "leaderboard_point_by_horizon.csv"
    d = pd.read_csv(path)

    keep = (
        ((d["model"] == "persistence") & (d["source_group"] == "main_linear"))
        | ((d["model"] == "enet_a0.1_l0.5") & (d["source_group"] == "main_linear"))
        | ((d["model"] == "hgbr") & (d["source_group"] == "hybrid_rank_v2_guarded"))
    )
    d = d.loc[keep, ["model", "source_group", "horizon", "MAE", "RMSE"]].copy()
    d = d.drop_duplicates(subset=["model", "horizon"], keep="first")
    d = d.rename(columns={"model": "model_key", "source_group": "ulsan_source_group"})
    return d


def _build_ulsan_alarm_baseline() -> pd.DataFrame:
    path = ROOT / "results" / "tables" / "model_alarm_performance_full.csv"
    d = pd.read_csv(path)

    keep = (
        ((d["model"] == "persistence") & (d["source_group"] == "main_alarm"))
        | ((d["model"] == "enet_a0.1_l0.5") & (d["source_group"] == "main_alarm"))
        | ((d["model"] == "hgbr") & (d["source_group"] == "hybrid_rank_v2_guarded"))
    )
    cols = ["model", "source_group", "horizon", "tau_mgL", "budget_r", "precision", "recall"]
    d = d.loc[keep, cols].copy()
    d["enrichment"] = d["recall"] / d["budget_r"]
    d = d.drop_duplicates(subset=["model", "horizon", "tau_mgL", "budget_r"], keep="first")
    d = d.rename(columns={"model": "model_key", "source_group": "ulsan_source_group"})
    return d


def make_transportability_gap_table(seoul_summary: pd.DataFrame) -> pd.DataFrame:
    uls_point = _build_ulsan_point_baseline()
    uls_alarm = _build_ulsan_alarm_baseline()

    point = seoul_summary[seoul_summary["metric_type"] == "point"].copy()
    point["model_key"] = point["model"].map(MODEL_KEY_MAP)
    point = point[["dataset", "horizon", "model", "model_key", "MAE", "RMSE"]]

    point_long = point.melt(
        id_vars=["dataset", "horizon", "model", "model_key"],
        value_vars=["MAE", "RMSE"],
        var_name="metric",
        value_name="seoul_value",
    )

    up_long = uls_point.melt(
        id_vars=["model_key", "ulsan_source_group", "horizon"],
        value_vars=["MAE", "RMSE"],
        var_name="metric",
        value_name="ulsan_value",
    )

    point_gap = point_long.merge(up_long, on=["model_key", "horizon", "metric"], how="left")
    point_gap["metric_type"] = "point"
    point_gap["tau_mgL"] = np.nan
    point_gap["budget_r"] = np.nan
    point_gap["direction"] = "lower_is_better"

    alarm = seoul_summary[seoul_summary["metric_type"] == "alarm"].copy()
    alarm["model_key"] = alarm["model"].map(MODEL_KEY_MAP)
    alarm["tau_mgL"] = pd.to_numeric(alarm["tau_mgL"], errors="coerce")
    alarm["budget_r"] = pd.to_numeric(alarm["budget_r"], errors="coerce")
    alarm = alarm[
        [
            "dataset",
            "horizon",
            "model",
            "model_key",
            "tau_mgL",
            "budget_r",
            "precision",
            "recall",
            "enrichment",
        ]
    ]

    alarm_long = alarm.melt(
        id_vars=["dataset", "horizon", "model", "model_key", "tau_mgL", "budget_r"],
        value_vars=["precision", "recall", "enrichment"],
        var_name="metric",
        value_name="seoul_value",
    )

    ua_long = uls_alarm.melt(
        id_vars=["model_key", "ulsan_source_group", "horizon", "tau_mgL", "budget_r"],
        value_vars=["precision", "recall", "enrichment"],
        var_name="metric",
        value_name="ulsan_value",
    )

    alarm_gap = alarm_long.merge(
        ua_long,
        on=["model_key", "horizon", "tau_mgL", "budget_r", "metric"],
        how="left",
    )
    alarm_gap["metric_type"] = "alarm"
    alarm_gap["direction"] = "higher_is_better"

    gap = pd.concat([point_gap, alarm_gap], ignore_index=True)
    gap["delta_seoul_minus_ulsan"] = gap["seoul_value"] - gap["ulsan_value"]

    denom = gap["ulsan_value"].abs().replace({0.0: np.nan})
    gap["relative_change_pct"] = 100.0 * gap["delta_seoul_minus_ulsan"] / denom
    gap["baseline_available"] = gap["ulsan_value"].notna().astype(int)

    # Improvement logic depends on metric direction.
    gap["improved_on_seoul"] = np.where(
        gap["ulsan_value"].notna(),
        np.where(
            gap["direction"] == "lower_is_better",
            gap["delta_seoul_minus_ulsan"] < 0,
            gap["delta_seoul_minus_ulsan"] > 0,
        ),
        np.nan,
    )

    gap["model_label"] = gap["model_key"].map(MODEL_LABEL).fillna(gap["model"].astype(str))
    gap = gap[
        [
            "dataset",
            "metric_type",
            "metric",
            "horizon",
            "tau_mgL",
            "budget_r",
            "model",
            "model_key",
            "model_label",
            "ulsan_source_group",
            "ulsan_value",
            "seoul_value",
            "delta_seoul_minus_ulsan",
            "relative_change_pct",
            "direction",
            "baseline_available",
            "improved_on_seoul",
        ]
    ].sort_values(["metric_type", "metric", "horizon", "tau_mgL", "budget_r", "model_label"]).reset_index(drop=True)

    return gap


def make_domain_shift_table() -> pd.DataFrame:
    u_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Ulsan_Yongsan.csv"), "ulsan")
    s_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Seoul_Tancheon_1.csv"), "seoul")

    variables = [
        "TNout",
        "Inflow",
        "TNin",
        "TOCin",
        "BODin",
        "temp_mean_c",
        "precip_total_mm",
    ]

    rows = []
    for v in variables:
        us = _safe_stats(u_raw[v])
        ss = _safe_stats(s_raw[v])

        uvals = pd.to_numeric(u_raw[v], errors="coerce").to_numpy(dtype=float)
        svals = pd.to_numeric(s_raw[v], errors="coerce").to_numpy(dtype=float)
        uvals = uvals[np.isfinite(uvals)]
        svals = svals[np.isfinite(svals)]

        pooled_sd = np.sqrt((us["sd"] ** 2 + ss["sd"] ** 2) / 2.0) if np.isfinite(us["sd"]) and np.isfinite(ss["sd"]) else np.nan
        smd = (ss["mean"] - us["mean"]) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else np.nan

        rows.append(
            {
                "variable": v,
                "n_total_ulsan": us["n_total"],
                "n_valid_ulsan": us["n_valid"],
                "missing_rate_ulsan": us["missing_rate"],
                "mean_ulsan": us["mean"],
                "sd_ulsan": us["sd"],
                "median_ulsan": us["median"],
                "iqr_ulsan": us["iqr"],
                "n_total_seoul": ss["n_total"],
                "n_valid_seoul": ss["n_valid"],
                "missing_rate_seoul": ss["missing_rate"],
                "mean_seoul": ss["mean"],
                "sd_seoul": ss["sd"],
                "median_seoul": ss["median"],
                "iqr_seoul": ss["iqr"],
                "delta_mean_seoul_minus_ulsan": ss["mean"] - us["mean"],
                "delta_median_seoul_minus_ulsan": ss["median"] - us["median"],
                "smd": smd,
                "ks_statistic": _ks_statistic(uvals, svals),
            }
        )

    out = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    return out


def _draw_heatmap(ax, mat: pd.DataFrame, title: str, cmap: str = "coolwarm") -> None:
    arr = mat.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    vmax = np.nanmax(np.abs(arr[finite])) if finite.any() else 1.0
    vmax = max(vmax, 1e-6)

    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=0)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(i) for i in mat.index])
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = "NA" if not np.isfinite(val) else f"{val:+.3f}"
            color = "black" if not np.isfinite(val) else ("white" if abs(val) > vmax * 0.45 else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    return im


def make_figures(gap: pd.DataFrame, shift: pd.DataFrame) -> None:
    SI_FIGS.mkdir(parents=True, exist_ok=True)

    # Figure 1: transportability gap heatmaps.
    point_rmse = gap[
        (gap["metric_type"] == "point")
        & (gap["metric"] == "RMSE")
        & (gap["baseline_available"] == 1)
    ].copy()
    point_rmse = point_rmse.pivot_table(
        index="model_label",
        columns="horizon",
        values="delta_seoul_minus_ulsan",
        aggfunc="mean",
    )

    alarm_recall = gap[
        (gap["metric_type"] == "alarm")
        & (gap["metric"] == "recall")
        & (gap["baseline_available"] == 1)
    ].copy()
    alarm_recall = (
        alarm_recall.groupby(["model_label", "horizon"], as_index=False)["delta_seoul_minus_ulsan"]
        .mean()
        .pivot(index="model_label", columns="horizon", values="delta_seoul_minus_ulsan")
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    im0 = _draw_heatmap(axes[0], point_rmse, "Point RMSE delta (Seoul - Ulsan)")
    im1 = _draw_heatmap(axes[1], alarm_recall, "Alarm recall delta mean (Seoul - Ulsan)")
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Delta value")

    out_png = SI_FIGS / "FigureS_Seoul_transportability_gap_heatmap.png"
    out_pdf = SI_FIGS / "FigureS_Seoul_transportability_gap_heatmap.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: absolute SMD bars by variable.
    s = shift.copy()
    s = s[np.isfinite(s["smd"])].copy()
    s = s.sort_values("smd", key=lambda x: x.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    colors = ["#C9485B" if v >= 0 else "#0B4F6C" for v in s["smd"].to_numpy(dtype=float)]
    ax.barh(s["variable"], s["smd"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.2, color="0.5", linestyle="--", linewidth=0.8)
    ax.axvline(-0.2, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Standardized mean difference (Seoul vs Ulsan)")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.2)

    out_png = SI_FIGS / "FigureS_Seoul_domain_shift_smd.png"
    out_pdf = SI_FIGS / "FigureS_Seoul_domain_shift_smd.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SI_TABLES.mkdir(parents=True, exist_ok=True)
    SI_FIGS.mkdir(parents=True, exist_ok=True)

    seoul_path = SI_TABLES / "TableS15_external_validation_summary_metrics_seoul.csv"
    if not seoul_path.exists():
        raise FileNotFoundError(f"Missing required file: {seoul_path}")

    seoul_summary = pd.read_csv(seoul_path)
    gap = make_transportability_gap_table(seoul_summary)
    shift = make_domain_shift_table()

    gap_path = SI_TABLES / "TableS15b_transportability_gap_vs_ulsan.csv"
    shift_path = SI_TABLES / "TableS15c_domain_shift_diagnostics_ulsan_vs_seoul.csv"
    gap.to_csv(gap_path, index=False)
    shift.to_csv(shift_path, index=False)

    make_figures(gap, shift)

    print("Saved:")
    print("-", gap_path.relative_to(ROOT))
    print("-", shift_path.relative_to(ROOT))
    print("-", (SI_FIGS / "FigureS_Seoul_transportability_gap_heatmap.png").relative_to(ROOT))
    print("-", (SI_FIGS / "FigureS_Seoul_transportability_gap_heatmap.pdf").relative_to(ROOT))
    print("-", (SI_FIGS / "FigureS_Seoul_domain_shift_smd.png").relative_to(ROOT))
    print("-", (SI_FIGS / "FigureS_Seoul_domain_shift_smd.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
