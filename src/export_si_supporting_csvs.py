#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "results" / "hybrid_rank_v2_runs" / "20260220_131820_H5_hybrid_rank_v2"
OUT_DIR = REPO / "reports" / "main_manuscript_assets" / "SI" / "data_exports"


def export_ulsan_daily_aligned() -> Path:
	src = REPO / "data" / "Ulsan_Yongsan.csv"
	df = pd.read_csv(src)
	if "Date" in df.columns:
		df = df.rename(columns={"Date": "date"})
	drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
	if drop_cols:
		df = df.drop(columns=drop_cols)
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.sort_values("date").reset_index(drop=True)
	df["date"] = df["date"].dt.strftime("%Y-%m-%d")

	out = OUT_DIR / "ulsan_daily_aligned.csv"
	df.to_csv(out, index=False)
	return out


def export_cv_folds() -> Path:
	rank_scores = pd.read_csv(RUN_DIR / "predictions" / "rank_scores_tauwise_H5.csv")
	rank_scores["date"] = pd.to_datetime(rank_scores["date"], errors="coerce")
	cv = rank_scores[["date", "fold"]].drop_duplicates().copy()
	cv["horizon"] = 5
	cv["split"] = "test"
	cv = cv.sort_values(["fold", "date"]).reset_index(drop=True)
	cv["date"] = cv["date"].dt.strftime("%Y-%m-%d")

	out = OUT_DIR / "cv_folds.csv"
	cv.to_csv(out, index=False)
	return out


def export_predictions_long() -> Path:
	base = pd.read_csv(RUN_DIR / "predictions" / "rank_scores_tauwise_H5.csv")
	tcn = pd.read_csv(REPO / "results" / "predictions" / "bcr_tcn_v11_H5_v2_preds.csv")
	enet = pd.read_csv(RUN_DIR / "predictions" / "enet_H5_preds.csv")
	persist = pd.read_csv(RUN_DIR / "predictions" / "persistence_H5_preds.csv")
	hgbr = pd.read_csv(RUN_DIR / "predictions" / "hgbr_H5_preds.csv")
	hpoint = pd.read_csv(RUN_DIR / "predictions" / "hybrid_point_H5_preds.csv")

	key = ["date", "fold", "y_true"]
	for df in [base, tcn, enet, persist, hgbr, hpoint]:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")

	rows = []
	tau_list = [15, 16, 17]

	tcn_keep = tcn[["date", "fold", "y_true", "y_pred", "p_tau15", "p_tau16", "p_tau17"]].copy()
	for tau in tau_list:
		tmp = tcn_keep[["date", "fold", "y_true", "y_pred", f"p_tau{tau}"]].copy()
		tmp["horizon"] = 5
		tmp["tau_mgL"] = tau
		tmp["model"] = "tcn"
		tmp = tmp.rename(columns={f"p_tau{tau}": "score"})
		tmp["score_source"] = f"p_tau{tau}"
		rows.append(tmp)

	hybrid_keep = base[["date", "fold", "y_true", "s_tau15_hybrid", "s_tau16_hybrid", "s_tau17_hybrid"]].copy()
	hybrid_point_lookup = hpoint.rename(columns={"y_pred": "y_pred_hybrid_point"})[["date", "fold", "y_pred_hybrid_point"]]
	hybrid_keep = hybrid_keep.merge(hybrid_point_lookup, on=["date", "fold"], how="left")
	for tau in tau_list:
		tmp = hybrid_keep[["date", "fold", "y_true", "y_pred_hybrid_point", f"s_tau{tau}_hybrid"]].copy()
		tmp["horizon"] = 5
		tmp["tau_mgL"] = tau
		tmp["model"] = "hybrid_rank_v2"
		tmp = tmp.rename(columns={"y_pred_hybrid_point": "y_pred", f"s_tau{tau}_hybrid": "score"})
		tmp["score_source"] = f"s_tau{tau}_hybrid"
		rows.append(tmp)

	for model_name, df in [("enet", enet), ("persistence", persist), ("hgbr", hgbr), ("hybrid_point", hpoint)]:
		keep = df[["date", "fold", "y_true", "y_pred"]].copy()
		for tau in tau_list:
			tmp = keep.copy()
			tmp["horizon"] = 5
			tmp["tau_mgL"] = tau
			tmp["model"] = model_name
			tmp["score"] = tmp["y_pred"]
			tmp["score_source"] = "y_pred"
			rows.append(tmp)

	out_df = pd.concat(rows, ignore_index=True)
	out_df = out_df[["date", "fold", "horizon", "tau_mgL", "model", "y_true", "y_pred", "score", "score_source"]]
	out_df = out_df.sort_values(["fold", "date", "tau_mgL", "model"]).reset_index(drop=True)
	out_df["date"] = pd.to_datetime(out_df["date"]).dt.strftime("%Y-%m-%d")

	out = OUT_DIR / "predictions_long.csv"
	out_df.to_csv(out, index=False)
	return out


def export_tcn_training_log() -> Path:
	meta = json.loads((REPO / "results" / "metrics" / "bcr_tcn_v11_H5_v2_meta.json").read_text())
	fold_summary = pd.DataFrame(meta.get("fold_summary", []))
	fold_summary["horizon"] = meta.get("h", 5)
	fold_summary["sequence_length_L"] = meta.get("L", None)
	fold_summary["n_predictions_total"] = meta.get("n_preds", None)
	fold_summary["note"] = "Per-epoch logs were not saved; this is fold-level best validation summary."

	out = OUT_DIR / "tcn_training_log.csv"
	fold_summary.to_csv(out, index=False)
	return out


def export_hybridrank_weights() -> Path:
	src = RUN_DIR / "models" / "rank_hybrid_weights_H5.csv"
	df = pd.read_csv(src)
	out = OUT_DIR / "hybridrank_weights.csv"
	df.to_csv(out, index=False)
	return out


def sanity_check(paths: list[Path]) -> pd.DataFrame:
	rows = []
	for p in paths:
		df = pd.read_csv(p)
		info = {
			"file": str(p.relative_to(REPO)),
			"rows": int(len(df)),
			"cols": int(df.shape[1]),
			"has_null": bool(df.isna().any().any()),
		}
		if "date" in df.columns:
			info["date_min"] = str(df["date"].min())
			info["date_max"] = str(df["date"].max())
			info["date_unique"] = int(df["date"].nunique())
		if {"date", "fold"}.issubset(df.columns):
			info["dup_date_fold"] = int(df.duplicated(["date", "fold"]).sum())
		if {"date", "fold", "tau_mgL", "model"}.issubset(df.columns):
			info["dup_key_predlong"] = int(df.duplicated(["date", "fold", "tau_mgL", "model"]).sum())
		if "fold" in df.columns:
			info["folds"] = ",".join(map(str, sorted(pd.Series(df["fold"]).dropna().unique().tolist())))
		rows.append(info)

	out = pd.DataFrame(rows)
	out_path = OUT_DIR / "si_export_sanity_check.csv"
	out.to_csv(out_path, index=False)
	return out


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	outputs = [
		export_ulsan_daily_aligned(),
		export_cv_folds(),
		export_predictions_long(),
		export_tcn_training_log(),
		export_hybridrank_weights(),
	]
	sanity = sanity_check(outputs)

	print("Exported files:")
	for p in outputs:
		print("-", p.relative_to(REPO))
	print("Sanity report:")
	print((OUT_DIR / "si_export_sanity_check.csv").relative_to(REPO))
	print(sanity.to_string(index=False))


if __name__ == "__main__":
	main()
