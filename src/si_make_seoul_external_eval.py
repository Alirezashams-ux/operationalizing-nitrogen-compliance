#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SI_TABLES = ROOT / "reports" / "main_manuscript_assets" / "SI" / "tables"
SI_FIGS = ROOT / "reports" / "main_manuscript_assets" / "SI" / "figures"

TAUS = [15.0, 16.0, 17.0]
BUDGETS = [0.05, 0.10]
HORIZONS = [1, 3, 5]


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

	must_have = ["TNout", "Inflow", "TNin", "TOCin", "Date"]
	for c in must_have:
		if c not in out.columns:
			raise ValueError(f"{site} missing required column: {c}")

	if "BODin" not in out.columns:
		out["BODin"] = np.nan
	if "temp_mean_c" not in out.columns:
		out["temp_mean_c"] = np.nan
	if "precip_total_mm" not in out.columns:
		out["precip_total_mm"] = 0.0

	num_cols = ["TNout", "Inflow", "TNin", "TOCin", "BODin", "temp_mean_c", "precip_total_mm"]
	for c in num_cols:
		out[c] = pd.to_numeric(out[c], errors="coerce")

	return out


def _engineer_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
	d = df.copy()
	d["C_N"] = d["TOCin"] / (d["TNin"] + 1e-6)
	d["y"] = d["TNout"].shift(-h)

	d["TNout_lag1"] = d["TNout"].shift(1)
	d["TNout_lag3"] = d["TNout"].shift(3)
	d["TNout_lag5"] = d["TNout"].shift(5)
	d["TNout_lag7"] = d["TNout"].shift(7)

	d["TNout_roll7"] = d["TNout"].shift(1).rolling(7).mean()
	d["TNout_roll14"] = d["TNout"].shift(1).rolling(14).mean()
	d["TNout_roll30"] = d["TNout"].shift(1).rolling(30).mean()

	for col in ["Inflow", "TNin", "TOCin", "BODin"]:
		d[f"{col}_roll7"] = d[col].rolling(7).mean()
		d[f"{col}_roll14"] = d[col].rolling(14).mean()

	d["temp_roll7"] = d["temp_mean_c"].rolling(7).mean()
	d["temp_roll14"] = d["temp_mean_c"].rolling(14).mean()
	d["temp_roll30"] = d["temp_mean_c"].rolling(30).mean()
	d["precip_sum3"] = d["precip_total_mm"].rolling(3).sum()
	d["precip_sum7"] = d["precip_total_mm"].rolling(7).sum()
	d["precip_sum14"] = d["precip_total_mm"].rolling(14).sum()

	d["C_N_roll14"] = d["C_N"].rolling(14).mean()
	d["Inflow_x_precip3"] = d["Inflow"] * d["precip_sum3"]
	d["temp_x_CN"] = d["temp_mean_c"] * d["C_N"]
	d["Inflow_x_TNin"] = d["Inflow"] * d["TNin"]

	doy = d["Date"].dt.dayofyear.values
	d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
	d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

	feature_cols = [
		"TNout_lag1", "TNout_lag3", "TNout_lag5", "TNout_lag7",
		"TNout_roll7", "TNout_roll14", "TNout_roll30",
		"Inflow", "Inflow_roll7", "Inflow_roll14",
		"TNin", "TNin_roll7", "TNin_roll14",
		"TOCin", "TOCin_roll7", "TOCin_roll14",
		"BODin", "BODin_roll7", "BODin_roll14",
		"C_N", "C_N_roll14",
		"temp_mean_c", "temp_roll7", "temp_roll14", "temp_roll30",
		"precip_total_mm", "precip_sum3", "precip_sum7", "precip_sum14",
		"Inflow_x_precip3", "temp_x_CN", "Inflow_x_TNin",
		"sin_doy", "cos_doy",
	]

	out = d.dropna(subset=feature_cols + ["y", "TNout"]).copy()
	out = out.reset_index(drop=True)
	return out[["Date", "TNout", "y"] + feature_cols]


def _alarm_metrics(y_true: np.ndarray, score: np.ndarray, tau: float, r: float) -> dict:
	n = len(y_true)
	k = max(1, int(np.ceil(r * n)))
	event = (y_true >= tau).astype(int)
	order = np.argsort(-score)
	top = order[:k]
	tp = int(event[top].sum())
	events = int(event.sum())
	fp = int(k - tp)
	recall = float(tp / events) if events > 0 else np.nan
	precision = float(tp / k) if k > 0 else np.nan
	return {
		"n": int(n), "k": int(k), "events": int(events), "TP": int(tp), "FP": int(fp),
		"precision": precision, "recall": recall, "enrichment": float(recall / r) if np.isfinite(recall) else np.nan,
	}


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
	err = y_true - y_pred
	return {
		"MAE": float(mean_absolute_error(y_true, y_pred)),
		"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
		"bias": float(np.mean(err)),
	}


def _build_s14_mapping(ulsan_df: pd.DataFrame, seoul_df: pd.DataFrame) -> pd.DataFrame:
	seoul_cols = set(seoul_df.columns)
	rows = []
	for c in ulsan_df.columns:
		if c in seoul_cols:
			rows.append({"ulsan_variable": c, "seoul_counterpart": c, "unit_check": "check_in_raw_metadata", "transformation": "none", "retained": 1})
		else:
			mapped = {
				"temp_mean_c": "Average Temperature",
				"precip_total_mm": "Precipitation",
				"wind_mean_ms": "Average Wind Speed",
				"wind_max_ms": "Highest Wind Speed",
				"rh_mean_pct": "Average Humidity",
				"sunshine_total_hr": "Sunshine Duration",
				"solar_rad_total_mj_m2": "Total Solar Radiation",
				"temp_range_c": "Gap of daily temperature",
			}.get(c, "")
			rows.append({
				"ulsan_variable": c,
				"seoul_counterpart": mapped,
				"unit_check": "check_in_raw_metadata" if mapped else "not_available",
				"transformation": "rename" if mapped else "dropped",
				"retained": int(bool(mapped)),
			})
	return pd.DataFrame(rows)


def _compute_block_summary(df_pred: pd.DataFrame) -> pd.DataFrame:
	# Create 3 contiguous temporal blocks as pseudo-folds for uncertainty reporting.
	d = df_pred.sort_values("Date").reset_index(drop=True).copy()
	n = len(d)
	block_edges = np.linspace(0, n, 4, dtype=int)
	blocks = []
	for i in range(3):
		s, e = block_edges[i], block_edges[i + 1]
		if e > s:
			b = d.iloc[s:e].copy()
			b["block"] = i + 1
			blocks.append(b)
	return pd.concat(blocks, ignore_index=True)


def run_external_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
	u_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Ulsan_Yongsan.csv"), "ulsan")
	s_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Seoul_Tancheon_1.csv"), "seoul")

	summary_rows = []
	curve_rows = []

	for h in HORIZONS:
		u = _engineer_features(u_raw, h)
		s = _engineer_features(s_raw, h)

		feature_cols = [c for c in u.columns if c not in ["Date", "TNout", "y"]]
		common = [c for c in feature_cols if c in s.columns]
		u_train = u.dropna(subset=common + ["y"])
		s_test = s.dropna(subset=common + ["y", "TNout"])

		Xtr = u_train[common].to_numpy(float)
		ytr = u_train["y"].to_numpy(float)
		Xte = s_test[common].to_numpy(float)
		yte = s_test["y"].to_numpy(float)

		models = {
			"Persistence": None,
			"ElasticNet": Pipeline([
				("scaler", StandardScaler()),
				("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=20000)),
			]),
			"HGBR": None,
		}

		hg_path = ROOT / "results" / "hgbr" / f"hgbr_optuna_H{h}_v2.json"
		if hg_path.exists():
			hp = json.loads(hg_path.read_text()).get("best_params", {})
		else:
			hp = {
				"learning_rate": 0.05,
				"max_iter": 400,
				"max_leaf_nodes": 63,
				"min_samples_leaf": 20,
				"l2_regularization": 1e-4,
				"max_bins": 255,
			}
		hp.update({"early_stopping": True, "validation_fraction": 0.1, "random_state": 42})
		models["HGBR"] = HistGradientBoostingRegressor(**hp)

		preds = {}
		for name, model in models.items():
			if name == "Persistence":
				y_pred = s_test["TNout"].to_numpy(float)
			else:
				model.fit(Xtr, ytr)
				y_pred = model.predict(Xte)
			preds[name] = y_pred

			pm = _point_metrics(yte, y_pred)
			summary_rows.append({
				"dataset": "Seoul_Tancheon", "horizon": h, "model": name,
				"metric_type": "point", "tau_mgL": np.nan, "budget_r": np.nan,
				"n_test": len(yte), "k": np.nan, "events": np.nan,
				"TP": np.nan, "FP": np.nan,
				"precision": np.nan, "recall": np.nan, "enrichment": np.nan,
				"MAE": pm["MAE"], "RMSE": pm["RMSE"], "bias": pm["bias"],
			})

		date_vals = pd.to_datetime(s_test["Date"])
		for model_name, y_pred in preds.items():
			dfp = pd.DataFrame({"Date": date_vals, "y_true": yte, "y_pred": y_pred})
			blocks = _compute_block_summary(dfp)

			for tau in TAUS:
				for r in BUDGETS:
					# overall
					m = _alarm_metrics(dfp["y_true"].to_numpy(float), dfp["y_pred"].to_numpy(float), tau=tau, r=r)

					# pseudo-fold uncertainty from 3 temporal blocks
					block_vals = []
					for b, g in blocks.groupby("block"):
						mb = _alarm_metrics(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float), tau=tau, r=r)
						block_vals.append({"block": int(b), **mb})
					bdf = pd.DataFrame(block_vals)
					rec_mean = float(bdf["recall"].mean())
					rec_sd = float(bdf["recall"].std(ddof=1)) if len(bdf) > 1 else 0.0
					rec_ci95 = float(1.96 * rec_sd / np.sqrt(len(bdf))) if len(bdf) > 1 else 0.0
					pre_mean = float(bdf["precision"].mean())
					pre_sd = float(bdf["precision"].std(ddof=1)) if len(bdf) > 1 else 0.0
					pre_ci95 = float(1.96 * pre_sd / np.sqrt(len(bdf))) if len(bdf) > 1 else 0.0

					summary_rows.append({
						"dataset": "Seoul_Tancheon", "horizon": h, "model": model_name,
						"metric_type": "alarm", "tau_mgL": tau, "budget_r": r,
						"n_test": m["n"], "k": m["k"], "events": m["events"],
						"TP": m["TP"], "FP": m["FP"],
						"precision": m["precision"], "recall": m["recall"], "enrichment": m["enrichment"],
						"precision_mean_block": pre_mean, "precision_ci95_block": pre_ci95,
						"recall_mean_block": rec_mean, "recall_ci95_block": rec_ci95,
						"MAE": np.nan, "RMSE": np.nan, "bias": np.nan,
					})

					curve_rows.append({
						"horizon": h, "tau_mgL": tau, "budget_r": r, "model": model_name,
						"recall_mean": rec_mean, "recall_ci95": rec_ci95,
						"recall_overall": m["recall"],
					})

	return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def plot_seoul_budget_recall(curve: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
	key_models = ["Persistence", "ElasticNet", "HGBR"]
	taus = [15.0, 16.0, 17.0]
	horizons = [1, 3, 5]

	plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9})
	fig, axes = plt.subplots(len(horizons), len(taus), figsize=(9.2, 7.8), sharex=True, sharey=True, constrained_layout=True)

	for i, h in enumerate(horizons):
		for j, tau in enumerate(taus):
			ax = axes[i, j]
			panel = curve[(curve["horizon"] == h) & (curve["tau_mgL"] == tau)]

			for model in key_models:
				d = panel[panel["model"] == model].sort_values("budget_r")
				if d.empty:
					continue
				x = d["budget_r"].to_numpy(float)
				y = d["recall_mean"].to_numpy(float)
				ci = d["recall_ci95"].to_numpy(float)
				line, = ax.plot(x, y, marker="o", linewidth=1.4, label=model)
				ax.fill_between(x, np.maximum(0, y - ci), np.minimum(1, y + ci), alpha=0.18, color=line.get_color(), linewidth=0)

			xx = np.array(sorted(panel["budget_r"].unique().tolist()))
			if len(xx):
				ax.plot(xx, xx, linestyle="--", color="0.5", linewidth=0.9)

			if i == 0:
				ax.set_title(f"τ={int(tau)} mg L$^{{-1}}$")
			if j == 0:
				ax.set_ylabel(f"H={h} d\nRecall")
			if i == len(horizons) - 1:
				ax.set_xlabel("Alarm budget r")
			ax.set_ylim(0, 1)
			ax.grid(True, alpha=0.2)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=600, bbox_inches="tight")
	fig.savefig(out_pdf, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	SI_TABLES.mkdir(parents=True, exist_ok=True)
	SI_FIGS.mkdir(parents=True, exist_ok=True)

	u_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Ulsan_Yongsan.csv"), "ulsan")
	s_raw = _standardize_columns(pd.read_csv(ROOT / "data" / "raw" / "Seoul_Tancheon_1.csv"), "seoul")
	s14 = _build_s14_mapping(u_raw, s_raw)
	s14_path = SI_TABLES / "TableS14_cross_site_variable_mapping_ulsan_to_seoul.csv"
	s14.to_csv(s14_path, index=False)

	s15, curve = run_external_eval()
	s15_path = SI_TABLES / "TableS15_external_validation_summary_metrics_seoul.csv"
	s15.to_csv(s15_path, index=False)

	curve_path = SI_TABLES / "TableS15a_seoul_budget_recall_curve_data.csv"
	curve.to_csv(curve_path, index=False)

	fig_png = SI_FIGS / "FigureS_Seoul_budget_recall_transferability.png"
	fig_pdf = SI_FIGS / "FigureS_Seoul_budget_recall_transferability.pdf"
	plot_seoul_budget_recall(curve, fig_png, fig_pdf)

	print("Saved:")
	print("-", s14_path.relative_to(ROOT))
	print("-", s15_path.relative_to(ROOT))
	print("-", curve_path.relative_to(ROOT))
	print("-", fig_png.relative_to(ROOT))
	print("-", fig_pdf.relative_to(ROOT))


if __name__ == "__main__":
	main()
