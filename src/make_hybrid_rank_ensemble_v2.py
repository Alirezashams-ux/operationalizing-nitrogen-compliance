#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rank01(x: pd.Series) -> np.ndarray:
    r = x.rank(method="average", ascending=True).to_numpy(dtype=float)
    if len(r) <= 1:
        return np.zeros_like(r)
    return (r - 1.0) / (len(r) - 1.0)


def budget_token(r: float) -> str:
    return f"{r:.3f}".replace(".", "p")


def normalize_pred_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"Date": "date"})
    if "y" in out.columns and "y_true" not in out.columns:
        out = out.rename(columns={"y": "y_true"})
    if "yhat" in out.columns and "y_pred" not in out.columns:
        out = out.rename(columns={"yhat": "y_pred"})
    if "pred" in out.columns and "y_pred" not in out.columns:
        out = out.rename(columns={"pred": "y_pred"})

    required = ["date", "y_true"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"{name}: missing required column '{col}'. Columns={list(out.columns)}")

    if "fold" not in out.columns:
        out["fold"] = 0

    out["date"] = out["date"].astype(str)
    out["fold"] = out["fold"].astype(int)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    return out


def mase(y_true: np.ndarray, y_pred: np.ndarray, m: int = 1) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) <= m:
        return float("nan")
    denom = np.mean(np.abs(y_true[m:] - y_true[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / (denom + 1e-9))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MASE1": mase(y_true, y_pred, m=1),
        "MASE7": mase(y_true, y_pred, m=7),
        "n": int(len(y_true)),
    }


def alarm_metrics_from_scores(y_true: np.ndarray, score: np.ndarray, tau: float, budget_r: float) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(budget_r * n)))
    event = (y_true >= tau).astype(int)
    order = np.argsort(-score)
    alarm_idx = order[:k]
    tp = int(event[alarm_idx].sum())
    events = int(event.sum())
    return {
        "n": int(n),
        "k": int(k),
        "events": events,
        "TP": tp,
        "precision": float(tp / k if k > 0 else np.nan),
        "recall": float(tp / events if events > 0 else np.nan),
    }


def build_weight_grid(n_models: int, step: float) -> List[Tuple[float, ...]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    combos = []
    for tup in product(vals, repeat=n_models):
        s = float(np.sum(tup))
        if s <= 0:
            continue
        w = tuple(float(v / s) for v in tup)
        combos.append(w)
    return combos


def choose_output_dir(root: Path, horizon: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{ts}_H{horizon}_hybrid_rank_v2"
    out = base
    i = 1
    while out.exists():
        out = Path(f"{base}_{i}")
        i += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def make_point_learning_curve(df: pd.DataFrame, model_cols: Dict[str, str], min_points: int = 120, step: int = 20) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True)
    rows = []
    for end in range(min_points, len(d) + 1, step):
        y = d.loc[: end - 1, "y_true"].to_numpy(float)
        for model_name, col in model_cols.items():
            yp = d.loc[: end - 1, col].to_numpy(float)
            m = point_metrics(y, yp)
            rows.append({"n_points": end, "model": model_name, **m})
    return pd.DataFrame(rows)


def make_alarm_budget_curves(df: pd.DataFrame, score_cols: Dict[str, Dict[int, str]], taus: List[int], budget_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    y = df["y_true"].to_numpy(float)
    for model_name, tau_map in score_cols.items():
        for tau in taus:
            score = df[tau_map[tau]].to_numpy(float)
            for r in budget_grid:
                m = alarm_metrics_from_scores(y, score, tau, float(r))
                rows.append({"model": model_name, "tau": tau, "budget_r": float(r), **m})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe hybrid rank ensemble v2 with timestamped outputs.")
    ap.add_argument("--h", type=int, required=True, help="Forecast horizon (e.g., 1, 3, 5)")
    ap.add_argument("--tcn", default=None, help="Path to TCN prediction CSV")
    ap.add_argument("--enet", default=None, help="Path to ENet prediction CSV")
    ap.add_argument("--persist", default=None, help="Path to persistence prediction CSV")
    ap.add_argument("--hgbr", default=None, help="Path to HGBR prediction CSV (optional but recommended)")
    ap.add_argument("--taus", default="15,16,17", help="Comma-separated alarm thresholds")
    ap.add_argument("--budgets", default="0.05,0.10", help="Comma-separated alarm budgets")
    ap.add_argument("--weight_step", type=float, default=0.1, help="Grid step for weight search")
    ap.add_argument(
        "--rank_objective",
        default="paper_keypoints",
        choices=["paper_keypoints", "mean_recall", "policy_specific", "policy_guarded"],
        help="Rank-weight tuning objective",
    )
    ap.add_argument("--guard_budget_max", type=float, default=0.05, help="For policy_guarded: use paper-fixed scores for budgets <= this value")
    ap.add_argument("--output_root", default="results/hybrid_rank_v2_runs", help="Root folder for timestamped run outputs")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_dir = root / "results" / "predictions"

    tcn_path = Path(args.tcn) if args.tcn else pred_dir / f"bcr_tcn_v11_H{args.h}_v2_preds.csv"
    enet_path = Path(args.enet) if args.enet else pred_dir / f"enet_a0.1_l0.5_v2_H{args.h}_preds.csv"
    per_path = Path(args.persist) if args.persist else pred_dir / f"persistence_v2_H{args.h}_preds.csv"
    hg_path = Path(args.hgbr) if args.hgbr else pred_dir / f"hgbr_optuna_H{args.h}_v2_preds.csv"

    taus = [int(x.strip()) for x in args.taus.split(",") if x.strip()]
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]

    for p in [tcn_path, enet_path, per_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    out_dir = choose_output_dir(root / args.output_root, args.h)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    models_dir = out_dir / "models"
    preds_dir = out_dir / "predictions"
    for d in [tables_dir, figs_dir, models_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tcn = normalize_pred_df(pd.read_csv(tcn_path), "TCN")
    enet = normalize_pred_df(pd.read_csv(enet_path), "ENet")
    per = normalize_pred_df(pd.read_csv(per_path), "Persistence")

    # Prepare main merge frame
    key_cols = ["fold", "date"]
    use_hg = hg_path.exists()

    tcn_keep = ["fold", "date", "y_true"]
    for tau in taus:
        pcol = f"p_tau{tau}"
        if pcol in tcn.columns:
            tcn_keep.append(pcol)
    if "y_pred" in tcn.columns:
        tcn_keep.append("y_pred")

    df = tcn[tcn_keep].copy()
    df = df.merge(enet[["fold", "date", "y_pred"]].rename(columns={"y_pred": "y_pred_enet"}), on=key_cols, how="inner")
    df = df.merge(per[["fold", "date", "y_pred"]].rename(columns={"y_pred": "y_pred_persist"}), on=key_cols, how="inner")

    if use_hg:
        hg = normalize_pred_df(pd.read_csv(hg_path), "HGBR")
        df = df.merge(hg[["fold", "date", "y_pred"]].rename(columns={"y_pred": "y_pred_hgbr"}), on=key_cols, how="inner")

    # Fill missing TCN tau-probability by y_pred rank if needed
    if "y_pred" in df.columns:
        for tau in taus:
            col = f"p_tau{tau}"
            if col not in df.columns:
                df[col] = np.nan

    base_reg_cols = ["y_pred_enet", "y_pred_persist"]
    if use_hg:
        base_reg_cols.append("y_pred_hgbr")

    # --------------------
    # 1) Point prediction hybrid (TNout): minimize MAE on train folds, apply to held-out fold
    # --------------------
    grid_point = build_weight_grid(len(base_reg_cols), step=args.weight_step)
    point_weight_rows = []
    df["y_pred_hybrid_point"] = np.nan

    folds = sorted(df["fold"].unique())
    for test_fold in folds:
        tr = df[df["fold"] != test_fold].copy()
        te = df[df["fold"] == test_fold].copy()

        best_w = None
        best_mae = float("inf")

        for w in grid_point:
            pred = np.zeros(len(tr), dtype=float)
            for wi, col in zip(w, base_reg_cols):
                pred += wi * tr[col].to_numpy(float)
            mae_val = np.mean(np.abs(tr["y_true"].to_numpy(float) - pred))
            if mae_val < best_mae:
                best_mae = float(mae_val)
                best_w = w

        pred_te = np.zeros(len(te), dtype=float)
        for wi, col in zip(best_w, base_reg_cols):
            pred_te += wi * te[col].to_numpy(float)

        df.loc[df["fold"] == test_fold, "y_pred_hybrid_point"] = pred_te
        point_weight_rows.append({
            "fold": int(test_fold),
            **{f"w_{col}": float(wi) for wi, col in zip(best_w, base_reg_cols)},
            "train_mae": float(best_mae),
        })

    point_weights = pd.DataFrame(point_weight_rows)
    point_weights.to_csv(models_dir / f"point_hybrid_weights_H{args.h}.csv", index=False)

    # --------------------
    # 2) Alarm rank hybrid: tau-specific, fold-wise leakage-safe tuning
    # --------------------
    rank_sources = [f"p_tau{tau}" for tau in taus]
    rank_reg_sources = ["y_pred_enet", "y_pred_persist"]
    if use_hg:
        rank_reg_sources.append("y_pred_hgbr")

    rank_weight_rows = []
    for tau in taus:
        df[f"s_tau{tau}_hybrid"] = np.nan
        if args.rank_objective in ("policy_specific", "policy_guarded"):
            for r in budgets:
                df[f"s_tau{tau}_r{budget_token(r)}_hybrid"] = np.nan

    # Weight dims: [tcn_tau, enet, persist, (hgbr)]
    n_rank_models = 1 + len(rank_reg_sources)
    grid_rank = build_weight_grid(n_rank_models, step=args.weight_step)

    for test_fold in folds:
        tr = df[df["fold"] != test_fold].copy()
        te = df[df["fold"] == test_fold].copy()

        baseline_w = [0.5, 0.25, 0.25] + ([0.0] if use_hg else [])
        baseline_floor: Dict[int, Dict[float, Dict[str, float]]] = {}

        for tau_ref in taus:
            tr_parts_ref = []
            for _, g in tr.groupby("fold", sort=True):
                rt_ref = rank01(g[f"p_tau{tau_ref}"] if f"p_tau{tau_ref}" in g.columns else g["y_pred_enet"])
                parts_ref = [rt_ref] + [rank01(g[col]) for col in rank_reg_sources]
                score_ref = np.zeros(len(g), dtype=float)
                for wi, part in zip(baseline_w, parts_ref):
                    score_ref += wi * part
                tr_parts_ref.append(pd.DataFrame({"y_true": g["y_true"].to_numpy(float), "score": score_ref}))
            all_tr_ref = pd.concat(tr_parts_ref, ignore_index=True)
            baseline_floor[tau_ref] = {}
            for r_ref in budgets:
                baseline_floor[tau_ref][float(r_ref)] = alarm_metrics_from_scores(
                    all_tr_ref["y_true"].to_numpy(float),
                    all_tr_ref["score"].to_numpy(float),
                    tau=tau_ref,
                    budget_r=float(r_ref),
                )

        for tau in taus:
            best_w = None
            best_obj = (-1.0, -1.0, -1.0)
            best_key_metrics: Dict[str, float] = {}

            if args.rank_objective in ("policy_specific", "policy_guarded"):
                tau_default_set = False
                for r_target in budgets:
                    best_w_local = None
                    best_obj_local = (-1.0, -1.0, -1.0)
                    best_key_local: Dict[str, float] = {}

                    for w in grid_rank:
                        tr_parts = []
                        for _, g in tr.groupby("fold", sort=True):
                            rt = rank01(g[f"p_tau{tau}"] if f"p_tau{tau}" in g.columns else g["y_pred_enet"])
                            parts = [rt]
                            for col in rank_reg_sources:
                                parts.append(rank01(g[col]))
                            score = np.zeros(len(g), dtype=float)
                            for wi, part in zip(w, parts):
                                score += wi * part
                            tr_parts.append(pd.DataFrame({"y_true": g["y_true"].to_numpy(float), "score": score}))

                        all_tr = pd.concat(tr_parts, ignore_index=True)
                        m_target = alarm_metrics_from_scores(
                            all_tr["y_true"].to_numpy(float),
                            all_tr["score"].to_numpy(float),
                            tau=tau,
                            budget_r=float(r_target),
                        )
                        floor_prec = baseline_floor[tau][float(r_target)]["precision"]
                        if m_target["precision"] + 1e-12 < floor_prec:
                            continue
                        obj = (float(m_target["recall"]), float(m_target["precision"]), float(m_target["TP"]))
                        if obj > best_obj_local:
                            best_obj_local = obj
                            best_w_local = w
                            best_key_local = {
                                "train_prec_r_target": float(m_target["precision"]),
                                "train_rec_r_target": float(m_target["recall"]),
                                "train_tp_r_target": float(m_target["TP"]),
                            }

                    if best_w_local is None:
                        best_w_local = tuple(float(x) for x in baseline_w)
                        best_obj_local = (0.0, 0.0, 0.0)
                        best_key_local = {
                            "train_prec_r_target": np.nan,
                            "train_rec_r_target": np.nan,
                            "train_tp_r_target": np.nan,
                        }

                    rt_te = rank01(te[f"p_tau{tau}"] if f"p_tau{tau}" in te.columns else te["y_pred_enet"])
                    parts_te = [rt_te] + [rank01(te[col]) for col in rank_reg_sources]
                    score_te = np.zeros(len(te), dtype=float)
                    for wi, part in zip(best_w_local, parts_te):
                        score_te += wi * part

                    score_col = f"s_tau{tau}_r{budget_token(r_target)}_hybrid"
                    df.loc[df["fold"] == test_fold, score_col] = score_te
                    if not tau_default_set:
                        df.loc[df["fold"] == test_fold, f"s_tau{tau}_hybrid"] = score_te
                        tau_default_set = True

                    rank_weight_rows.append({
                        "fold": int(test_fold),
                        "tau": int(tau),
                        "budget_r": float(r_target),
                        "rank_objective": args.rank_objective,
                        "objective_1": float(best_obj_local[0]),
                        "objective_2": float(best_obj_local[1]),
                        "objective_3": float(best_obj_local[2]),
                        "baseline_prec_floor": float(baseline_floor[tau][float(r_target)]["precision"]),
                        "w_tcn": float(best_w_local[0]),
                        **{f"w_{col}": float(wi) for wi, col in zip(best_w_local[1:], rank_reg_sources)},
                        **best_key_local,
                    })
            else:
                for w in grid_rank:
                    tr_parts = []
                    for _, g in tr.groupby("fold", sort=True):
                        rt = rank01(g[f"p_tau{tau}"] if f"p_tau{tau}" in g.columns else g["y_pred_enet"])
                        parts = [rt]
                        for col in rank_reg_sources:
                            parts.append(rank01(g[col]))
                        score = np.zeros(len(g), dtype=float)
                        for wi, part in zip(w, parts):
                            score += wi * part
                        tr_parts.append(pd.DataFrame({"y_true": g["y_true"].to_numpy(float), "score": score}))

                    all_tr = pd.concat(tr_parts, ignore_index=True)
                    train_alarm_by_budget: Dict[float, Dict[str, float]] = {}
                    for r in budgets:
                        m = alarm_metrics_from_scores(all_tr["y_true"].to_numpy(float), all_tr["score"].to_numpy(float), tau=tau, budget_r=r)
                        train_alarm_by_budget[float(r)] = m

                    recalls = [train_alarm_by_budget[float(r)]["recall"] for r in budgets]
                    precs = [train_alarm_by_budget[float(r)]["precision"] for r in budgets]

                    if args.rank_objective == "paper_keypoints":
                        if tau == 15:
                            m05 = train_alarm_by_budget[0.05]
                            floor_prec = baseline_floor[15][0.05]["precision"]
                            if m05["precision"] + 1e-12 < floor_prec:
                                continue
                            obj = (float(m05["precision"]), float(m05["recall"]), float(np.mean(recalls)))
                        elif tau == 16:
                            m05 = train_alarm_by_budget[0.05]
                            m10 = train_alarm_by_budget[0.10] if 0.10 in train_alarm_by_budget else m05
                            floor_prec = baseline_floor[16][0.05]["precision"]
                            if m05["precision"] + 1e-12 < floor_prec:
                                continue
                            obj = (float(np.mean([m05["recall"], m10["recall"]])), float(m05["precision"]), float(m10["precision"]))
                        else:
                            obj = (float(np.mean(recalls)), float(np.mean(precs)), 0.0)
                    else:
                        obj = (float(np.mean(recalls)), float(np.mean(precs)), 0.0)

                    if obj > best_obj:
                        best_obj = obj
                        best_w = w
                        best_key_metrics = {
                            "train_prec_r005": float(train_alarm_by_budget[0.05]["precision"]) if 0.05 in train_alarm_by_budget else np.nan,
                            "train_rec_r005": float(train_alarm_by_budget[0.05]["recall"]) if 0.05 in train_alarm_by_budget else np.nan,
                            "train_prec_r010": float(train_alarm_by_budget[0.10]["precision"]) if 0.10 in train_alarm_by_budget else np.nan,
                            "train_rec_r010": float(train_alarm_by_budget[0.10]["recall"]) if 0.10 in train_alarm_by_budget else np.nan,
                        }

                if best_w is None:
                    best_w = tuple(float(x) for x in baseline_w)
                    best_obj = (0.0, 0.0, 0.0)
                    best_key_metrics = {
                        "train_prec_r005": np.nan,
                        "train_rec_r005": np.nan,
                        "train_prec_r010": np.nan,
                        "train_rec_r010": np.nan,
                    }

                # Apply best weights on test fold (rank computed within test fold)
                rt_te = rank01(te[f"p_tau{tau}"] if f"p_tau{tau}" in te.columns else te["y_pred_enet"])
                parts_te = [rt_te] + [rank01(te[col]) for col in rank_reg_sources]
                score_te = np.zeros(len(te), dtype=float)
                for wi, part in zip(best_w, parts_te):
                    score_te += wi * part

                df.loc[df["fold"] == test_fold, f"s_tau{tau}_hybrid"] = score_te
                rank_weight_rows.append({
                    "fold": int(test_fold),
                    "tau": int(tau),
                    "rank_objective": args.rank_objective,
                    "objective_1": float(best_obj[0]),
                    "objective_2": float(best_obj[1]),
                    "objective_3": float(best_obj[2]),
                    "baseline_prec_floor_r005": float(baseline_floor[tau][0.05]["precision"]) if 0.05 in baseline_floor[tau] else np.nan,
                    "w_tcn": float(best_w[0]),
                    **{f"w_{col}": float(wi) for wi, col in zip(best_w[1:], rank_reg_sources)},
                    **best_key_metrics,
                })

    rank_weights = pd.DataFrame(rank_weight_rows)
    rank_weights.to_csv(models_dir / f"rank_hybrid_weights_H{args.h}.csv", index=False)

    # Save explicit static paper-fixed rank weights for traceability
    paper_fixed_weights = pd.DataFrame(
        {
            "component": ["tcn", "y_pred_enet", "y_pred_persist"] + (["y_pred_hgbr"] if use_hg else []),
            "weight": [0.5, 0.25, 0.25] + ([0.0] if use_hg else []),
        }
    )
    paper_fixed_weights.to_csv(models_dir / f"paper_fixed_rank_weights_H{args.h}.csv", index=False)

    # Fixed paper hybrid score (for direct comparison)
    fixed_w = [0.5, 0.25, 0.25] + ([0.0] if use_hg else [])
    for tau in taus:
        s_all = []
        for _, g in df.groupby("fold", sort=True):
            rt = rank01(g[f"p_tau{tau}"] if f"p_tau{tau}" in g.columns else g["y_pred_enet"])
            parts = [rt] + [rank01(g[col]) for col in rank_reg_sources]
            s = np.zeros(len(g), dtype=float)
            for wi, part in zip(fixed_w, parts):
                s += wi * part
            s_all.append(pd.Series(s, index=g.index))
        df[f"s_tau{tau}_paper_fixed"] = pd.concat(s_all).sort_index()

    # --------------------
    # 3) Save hybrid predictions (TNout + tau-scores)
    # --------------------
    pred_cols = ["date", "fold", "y_true", "y_pred_hybrid_point", "y_pred_enet", "y_pred_persist"]
    if use_hg:
        pred_cols.append("y_pred_hgbr")
    for tau in taus:
        pred_cols.extend([f"p_tau{tau}", f"s_tau{tau}_hybrid"])

    pred_out = df[pred_cols].sort_values(["fold", "date"]).reset_index(drop=True)
    pred_out.to_csv(out_dir / f"hybrid_rank_v2_H{args.h}_preds.csv", index=False)

    # Save per-model prediction/score files separately
    pred_out[["date", "fold", "y_true", "y_pred_hybrid_point"]].rename(
        columns={"y_pred_hybrid_point": "y_pred"}
    ).to_csv(preds_dir / f"hybrid_point_H{args.h}_preds.csv", index=False)
    pred_out[["date", "fold", "y_true", "y_pred_enet"]].rename(
        columns={"y_pred_enet": "y_pred"}
    ).to_csv(preds_dir / f"enet_H{args.h}_preds.csv", index=False)
    pred_out[["date", "fold", "y_true", "y_pred_persist"]].rename(
        columns={"y_pred_persist": "y_pred"}
    ).to_csv(preds_dir / f"persistence_H{args.h}_preds.csv", index=False)
    if use_hg:
        pred_out[["date", "fold", "y_true", "y_pred_hgbr"]].rename(
            columns={"y_pred_hgbr": "y_pred"}
        ).to_csv(preds_dir / f"hgbr_H{args.h}_preds.csv", index=False)

    tau_score_cols = [f"p_tau{tau}" for tau in taus] + [f"s_tau{tau}_hybrid" for tau in taus] + [f"s_tau{tau}_paper_fixed" for tau in taus]
    available_tau_score_cols = [c for c in tau_score_cols if c in pred_out.columns]
    pred_out[["date", "fold", "y_true"] + available_tau_score_cols].to_csv(
        preds_dir / f"rank_scores_tauwise_H{args.h}.csv", index=False
    )
    if args.rank_objective in ("policy_specific", "policy_guarded"):
        budget_cols = []
        for tau in taus:
            for r in budgets:
                c = f"s_tau{tau}_r{budget_token(r)}_hybrid"
                if c in pred_out.columns:
                    budget_cols.append(c)
        if budget_cols:
            pred_out[["date", "fold", "y_true"] + budget_cols].to_csv(
                preds_dir / f"rank_scores_budget_specific_H{args.h}.csv", index=False
            )

    # --------------------
    # 4) Point metrics (TNout prediction): MAE/RMSE/MASE
    # --------------------
    point_rows = []
    model_cols = {
        "hybrid_point": "y_pred_hybrid_point",
        "enet": "y_pred_enet",
        "persistence": "y_pred_persist",
    }
    if use_hg:
        model_cols["hgbr"] = "y_pred_hgbr"

    for model_name, col in model_cols.items():
        # overall
        m = point_metrics(df["y_true"].to_numpy(float), df[col].to_numpy(float))
        point_rows.append({"model": model_name, "scope": "overall", **m})
        # fold-wise
        for fold, g in df.groupby("fold", sort=True):
            mf = point_metrics(g["y_true"].to_numpy(float), g[col].to_numpy(float))
            point_rows.append({"model": model_name, "scope": f"fold{int(fold)}", **mf})

    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(tables_dir / f"point_metrics_tnout_H{args.h}.csv", index=False)

    # --------------------
    # 5) Alarm-budget metrics (overall + fold)
    # --------------------
    alarm_rows = []
    score_models = {
        "hybrid_rank_v2": {tau: f"s_tau{tau}_hybrid" for tau in taus},
        "hybrid_paper_fixed": {tau: f"s_tau{tau}_paper_fixed" for tau in taus},
        "tcn": {tau: f"p_tau{tau}" for tau in taus},
        "enet": {tau: "y_pred_enet" for tau in taus},
        "persistence": {tau: "y_pred_persist" for tau in taus},
    }
    if use_hg:
        score_models["hgbr"] = {tau: "y_pred_hgbr" for tau in taus}

    scopes = [("overall", df)] + [(f"fold{int(f)}", g) for f, g in df.groupby("fold", sort=True)]
    for model_name, tau_cols in score_models.items():
        for scope_name, dfx in scopes:
            yv = dfx["y_true"].to_numpy(float)
            for tau in taus:
                for r in budgets:
                    if model_name == "hybrid_rank_v2" and args.rank_objective in ("policy_specific", "policy_guarded"):
                        if args.rank_objective == "policy_guarded" and float(r) <= float(args.guard_budget_max):
                            sv = dfx[f"s_tau{tau}_paper_fixed"].to_numpy(float)
                        else:
                            bcol = f"s_tau{tau}_r{budget_token(r)}_hybrid"
                            sv = dfx[bcol].to_numpy(float)
                    else:
                        sv = dfx[tau_cols[tau]].to_numpy(float)
                    m = alarm_metrics_from_scores(yv, sv, tau=tau, budget_r=r)
                    alarm_rows.append({
                        "model": model_name,
                        "scope": scope_name,
                        "tau": int(tau),
                        "budget_r": float(r),
                        **m,
                    })

    alarm_df = pd.DataFrame(alarm_rows)
    alarm_df.to_csv(tables_dir / f"alarm_budget_metrics_H{args.h}.csv", index=False)

    key_cmp = alarm_df[
        (alarm_df["scope"] == "overall")
        & (alarm_df["tau"].isin([15, 16]))
        & (alarm_df["budget_r"].isin(budgets))
        & (alarm_df["model"].isin(["hybrid_rank_v2", "hybrid_paper_fixed", "tcn", "enet", "persistence", "hgbr"]))
    ].copy()
    key_cmp = key_cmp.sort_values(["tau", "budget_r", "model"]).reset_index(drop=True)
    key_cmp.to_csv(tables_dir / f"key_operating_points_compare_H{args.h}.csv", index=False)

    # --------------------
    # 6) Learning curves + visualization CSVs
    # --------------------
    lc_point = make_point_learning_curve(df, model_cols=model_cols, min_points=max(120, len(df) // 8), step=max(10, len(df) // 40))
    lc_point.to_csv(tables_dir / f"learning_curve_point_metrics_H{args.h}.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    for model_name in lc_point["model"].unique():
        d = lc_point[lc_point["model"] == model_name]
        plt.plot(d["n_points"], d["MAE"], marker="o", linewidth=1.8, label=model_name)
    plt.xlabel("Number of chronological points")
    plt.ylabel("MAE (TNout)")
    plt.title(f"Learning curve (MAE), H={args.h}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figs_dir / f"learning_curve_mae_H{args.h}.png", dpi=300)
    plt.savefig(figs_dir / f"learning_curve_mae_H{args.h}.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for model_name in lc_point["model"].unique():
        d = lc_point[lc_point["model"] == model_name]
        plt.plot(d["n_points"], d["RMSE"], marker="o", linewidth=1.8, label=model_name)
    plt.xlabel("Number of chronological points")
    plt.ylabel("RMSE (TNout)")
    plt.title(f"Learning curve (RMSE), H={args.h}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figs_dir / f"learning_curve_rmse_H{args.h}.png", dpi=300)
    plt.savefig(figs_dir / f"learning_curve_rmse_H{args.h}.pdf")
    plt.close()

    budget_grid = np.round(np.arange(0.01, 0.201, 0.01), 4)
    alarm_curve_models = {
        "hybrid_rank_v2": {tau: f"s_tau{tau}_hybrid" for tau in taus},
        "hybrid_paper_fixed": {tau: f"s_tau{tau}_paper_fixed" for tau in taus},
        "tcn": {tau: f"p_tau{tau}" for tau in taus},
        "persistence": {tau: "y_pred_persist" for tau in taus},
        "enet": {tau: "y_pred_enet" for tau in taus},
    }
    if use_hg:
        alarm_curve_models["hgbr"] = {tau: "y_pred_hgbr" for tau in taus}

    alarm_curve_df = make_alarm_budget_curves(df, alarm_curve_models, taus=taus, budget_grid=budget_grid)

    # For guarded/policy-specific runs, overwrite hybrid curve values with budget-specific/guarded policy scores
    if args.rank_objective in ("policy_specific", "policy_guarded"):
        y_all = df["y_true"].to_numpy(float)
        for tau in taus:
            for r in budget_grid:
                if args.rank_objective == "policy_guarded" and float(r) <= float(args.guard_budget_max):
                    s = df[f"s_tau{tau}_paper_fixed"].to_numpy(float)
                else:
                    # choose closest configured budget weight
                    r_ref = min(budgets, key=lambda b: abs(float(b) - float(r)))
                    c = f"s_tau{tau}_r{budget_token(r_ref)}_hybrid"
                    if c in df.columns:
                        s = df[c].to_numpy(float)
                    else:
                        s = df[f"s_tau{tau}_hybrid"].to_numpy(float)
                m = alarm_metrics_from_scores(y_all, s, tau=tau, budget_r=float(r))
                mask = (
                    (alarm_curve_df["model"] == "hybrid_rank_v2")
                    & (alarm_curve_df["tau"] == tau)
                    & (np.isclose(alarm_curve_df["budget_r"], float(r)))
                )
                for k in ["n", "k", "events", "TP", "precision", "recall"]:
                    alarm_curve_df.loc[mask, k] = m[k]

    alarm_curve_df.to_csv(tables_dir / f"budget_recall_curves_long_H{args.h}.csv", index=False)

    for tau in taus:
        plt.figure(figsize=(7.2, 4.8))
        for model_name in alarm_curve_df["model"].unique():
            d = alarm_curve_df[(alarm_curve_df["model"] == model_name) & (alarm_curve_df["tau"] == tau)]
            if d.empty:
                continue
            plt.plot(d["budget_r"], d["recall"], linewidth=2.0, label=model_name)
        plt.plot(budget_grid, budget_grid, "--", color="gray", linewidth=1.5, label="random baseline")
        plt.xlabel("Alarm budget r")
        plt.ylabel("Recall (TP / events)")
        plt.title(f"Budget–recall curve, TNout threshold τ={tau}, H={args.h}")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figs_dir / f"budget_recall_curve_tau{tau}_H{args.h}.png", dpi=300)
        plt.savefig(figs_dir / f"budget_recall_curve_tau{tau}_H{args.h}.pdf")
        plt.close()

    # Save run metadata for reproducibility
    registry_rows = [
        {
            "model": "hybrid_point",
            "type": "regression",
            "prediction_file": str(preds_dir / f"hybrid_point_H{args.h}_preds.csv"),
            "weights_file": str(models_dir / f"point_hybrid_weights_H{args.h}.csv"),
        },
        {
            "model": "hybrid_rank_v2",
            "type": "ranking",
            "prediction_file": str(preds_dir / f"rank_scores_tauwise_H{args.h}.csv"),
            "weights_file": str(models_dir / f"rank_hybrid_weights_H{args.h}.csv"),
        },
        {
            "model": "hybrid_paper_fixed",
            "type": "ranking",
            "prediction_file": str(preds_dir / f"rank_scores_tauwise_H{args.h}.csv"),
            "weights_file": str(models_dir / f"paper_fixed_rank_weights_H{args.h}.csv"),
        },
        {
            "model": "enet",
            "type": "regression",
            "prediction_file": str(preds_dir / f"enet_H{args.h}_preds.csv"),
            "weights_file": "n/a (external pretrained model)",
        },
        {
            "model": "persistence",
            "type": "baseline",
            "prediction_file": str(preds_dir / f"persistence_H{args.h}_preds.csv"),
            "weights_file": "n/a (rule-based)",
        },
    ]
    if use_hg:
        registry_rows.append(
            {
                "model": "hgbr",
                "type": "regression",
                "prediction_file": str(preds_dir / f"hgbr_H{args.h}_preds.csv"),
                "weights_file": "n/a (external pretrained model)",
            }
        )
    pd.DataFrame(registry_rows).to_csv(models_dir / f"model_registry_H{args.h}.csv", index=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "horizon": args.h,
        "inputs": {
            "tcn": str(tcn_path),
            "enet": str(enet_path),
            "persistence": str(per_path),
            "hgbr": str(hg_path) if use_hg else None,
        },
        "taus": taus,
        "budgets": budgets,
        "weight_step": args.weight_step,
        "rank_objective": args.rank_objective,
        "guard_budget_max": args.guard_budget_max,
        "base_regression_models": base_reg_cols,
        "output_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved run folder:", out_dir)
    print("Predictions:", out_dir / f"hybrid_rank_v2_H{args.h}_preds.csv")
    print("Point metrics (TNout MAE/RMSE/MASE):", tables_dir / f"point_metrics_tnout_H{args.h}.csv")
    print("Alarm metrics:", tables_dir / f"alarm_budget_metrics_H{args.h}.csv")
    print("Point learning curve CSV:", tables_dir / f"learning_curve_point_metrics_H{args.h}.csv")
    print("Alarm curve CSV:", tables_dir / f"budget_recall_curves_long_H{args.h}.csv")
    print("Model registry:", models_dir / f"model_registry_H{args.h}.csv")


if __name__ == "__main__":
    main()
