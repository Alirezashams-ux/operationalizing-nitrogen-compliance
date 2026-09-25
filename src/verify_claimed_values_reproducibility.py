#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


@dataclass
class Finding:
    status: str
    check: str
    detail: str


def _alarm_from_score(y_true: np.ndarray, score: np.ndarray, tau: float, budget_r: float) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(math.ceil(budget_r * n)))
    event = (y_true >= tau).astype(int)
    order = np.argsort(-score)
    top = event[order[:k]]
    tp = int(top.sum())
    events = int(event.sum())
    precision = tp / k if k else float("nan")
    recall = tp / events if events else float("nan")
    return {"n": n, "k": k, "events": events, "tp": tp, "precision": precision, "recall": recall}


def verify_table1() -> List[Finding]:
    findings: List[Finding] = []
    claim = pd.read_csv(REPO / "results" / "final_tables" / "results_discussion_csvs" / "Table1_point_forecast_summary_H1_H3_H5_main_linear.csv")

    src = {
        1: json.loads((REPO / "results" / "metrics" / "main_linear_metrics_H1.json").read_text()),
        3: json.loads((REPO / "results" / "metrics" / "main_linear_metrics_H3.json").read_text()),
        5: json.loads((REPO / "results" / "metrics" / "main_linear_metrics_H5.json").read_text()),
    }

    for _, r in claim.iterrows():
        h = int(r["horizon"])
        best_model = str(r["best_model"])
        pm = src[h]["point_metrics"][best_model]
        df = pd.DataFrame(pm)
        mae = float(df["MAE"].mean())
        rmse = float(df["RMSE"].mean())
        mase1 = float(df["MASE1"].mean())
        mase7 = float(df["MASE7"].mean())

        ok = (
            abs(float(r["MAE_mgL"]) - round(mae, 4)) <= 1e-4
            and abs(float(r["RMSE_mgL"]) - round(rmse, 4)) <= 1e-4
            and abs(float(r["MASE1"]) - round(mase1, 4)) <= 1e-4
            and abs(float(r["MASE7"]) - round(mase7, 4)) <= 1e-4
        )
        findings.append(
            Finding(
                "PASS" if ok else "FAIL",
                f"Table1 H={h}",
                (
                    f"claim mae/rmse/mase1/mase7={r['MAE_mgL']}/{r['RMSE_mgL']}/{r['MASE1']}/{r['MASE7']}"
                    f" vs recomputed {round(mae,4)}/{round(rmse,4)}/{round(mase1,4)}/{round(mase7,4)}"
                ),
            )
        )
    return findings


def verify_table2() -> List[Finding]:
    findings: List[Finding] = []
    claim = pd.read_csv(REPO / "results" / "final_tables" / "results_discussion_csvs" / "Table2_exceedance_prevalence_random_baseline.csv")
    pred_files = {
        1: REPO / "results" / "predictions" / "persistence_H1_preds.csv",
        3: REPO / "results" / "predictions" / "persistence_H3_preds.csv",
        5: REPO / "results" / "predictions" / "persistence_H5_preds.csv",
    }

    for _, r in claim.iterrows():
        h = int(r["horizon"])
        tau = float(r["tau"])
        df = pd.read_csv(pred_files[h])
        y = df["y_true"].to_numpy(float)
        n = len(y)
        events = int((y >= tau).sum())
        prevalence = events / n

        ok = (
            int(r["n_test"]) == n
            and int(r["events"]) == events
            and abs(float(r["prevalence"]) - prevalence) <= 1e-12
            and abs(float(r["random_recall_at_r005"]) - 0.05) <= 1e-12
            and abs(float(r["random_recall_at_r010"]) - 0.10) <= 1e-12
        )
        findings.append(
            Finding(
                "PASS" if ok else "FAIL",
                f"Table2 H={h} tau={int(tau)}",
                f"claim n/events/prevalence={int(r['n_test'])}/{int(r['events'])}/{float(r['prevalence']):.12f} vs recomputed {n}/{events}/{prevalence:.12f}",
            )
        )
    return findings


def _score_col_for_model(model: str, tau: int) -> str:
    if model in {"tcn", "hybrid_paper_fixed"}:
        return f"p_tau{tau}"
    if model == "hybrid_rank_v2":
        return f"s_tau{tau}_hybrid"
    return "y_pred"


def verify_table3() -> List[Finding]:
    findings: List[Finding] = []
    claim = pd.read_csv(REPO / "results" / "final_tables" / "results_discussion_csvs" / "Table3_representative_operating_points_main.csv")

    file_map = {
        (1, "persistence"): REPO / "results" / "predictions" / "persistence_H1_preds.csv",
        (1, "enet_a0.1_l0.5"): REPO / "results" / "predictions" / "enet_a0.1_l0.5_H1_preds.csv",
        (3, "persistence"): REPO / "results" / "predictions" / "persistence_H3_preds.csv",
        (3, "enet_a1.0_l0.2"): REPO / "results" / "predictions" / "enet_a1.0_l0.2_H3_preds.csv",
        (3, "enet_a0.01_l0.2"): REPO / "results" / "predictions" / "enet_a0.01_l0.2_H3_preds.csv",
        (5, "persistence"): REPO / "results" / "predictions" / "persistence_v2_H5_preds.csv",
        (5, "enet"): REPO / "results" / "predictions" / "enet_a0.1_l0.5_v2_H5_preds.csv",
        (5, "tcn"): REPO / "results" / "predictions" / "bcr_tcn_v11_H5_v2_preds.csv",
        (5, "hybrid_paper_fixed"): REPO / "results" / "predictions" / "hybrid_rank_ens_H5_v2_preds.csv",
        (5, "hybrid_rank_v2"): REPO / "results" / "hybrid_rank_v2_runs" / "20260219_195429_H5_hybrid_rank_v2" / "hybrid_rank_v2_H5_preds.csv",
    }
    h5_guarded_metrics = pd.read_csv(
        REPO
        / "results"
        / "hybrid_rank_v2_runs"
        / "20260219_195429_H5_hybrid_rank_v2"
        / "tables"
        / "alarm_budget_metrics_H5.csv"
    )

    for _, r in claim.iterrows():
        h = int(r["horizon"])
        tau = int(r["tau_mgL"])
        budget = float(r["budget_r"])
        model = str(r["model"])

        # HybridRank v2 uses guarded policy/budget-specific scores; the canonical source is alarm_budget_metrics_H5.csv.
        if h == 5 and model == "hybrid_rank_v2":
            sub = h5_guarded_metrics[
                (h5_guarded_metrics["model"] == "hybrid_rank_v2")
                & (h5_guarded_metrics["scope"] == "overall")
                & (h5_guarded_metrics["tau"].astype(float) == float(tau))
                & (h5_guarded_metrics["budget_r"].astype(float) == float(budget))
            ]
            if sub.empty:
                findings.append(
                    Finding(
                        "FAIL",
                        f"Table3 H={h} tau={tau} r={budget} {model}",
                        "No matching row in guarded alarm metrics source table.",
                    )
                )
                continue
            srow = sub.iloc[0]
            ok = (
                int(r["k"]) == int(srow["k"])
                and int(r["event_count"]) == int(srow["events"])
                and int(r["tp"]) == int(srow["TP"])
                and abs(float(r["precision"]) - round(float(srow["precision"]), 4)) <= 1e-4
                and abs(float(r["recall"]) - round(float(srow["recall"]), 4)) <= 1e-4
            )
            findings.append(
                Finding(
                    "PASS" if ok else "FAIL",
                    f"Table3 H={h} tau={tau} r={budget} {model}",
                    (
                        f"claim k/events/tp/prec/rec={int(r['k'])}/{int(r['event_count'])}/{int(r['tp'])}/{float(r['precision'])}/{float(r['recall'])}"
                        f" vs guarded-metrics {int(srow['k'])}/{int(srow['events'])}/{int(srow['TP'])}/{round(float(srow['precision']),4)}/{round(float(srow['recall']),4)}"
                    ),
                )
            )
            continue

        src_file = file_map[(h, model)]
        df = pd.read_csv(src_file)

        score_col = _score_col_for_model(model, tau)
        if score_col not in df.columns:
            findings.append(Finding("FAIL", f"Table3 H={h} tau={tau} r={budget} {model}", f"score column {score_col} not found in {src_file.relative_to(REPO)}"))
            continue

        y = df["y_true"].to_numpy(float)
        s = df[score_col].to_numpy(float)
        m = _alarm_from_score(y, s, tau=tau, budget_r=budget)

        ok = (
            int(r["k"]) == m["k"]
            and int(r["event_count"]) == m["events"]
            and int(r["tp"]) == m["tp"]
            and abs(float(r["precision"]) - round(m["precision"], 4)) <= 1e-4
            and abs(float(r["recall"]) - round(m["recall"], 4)) <= 1e-4
        )
        findings.append(
            Finding(
                "PASS" if ok else "FAIL",
                f"Table3 H={h} tau={tau} r={budget} {model}",
                (
                    f"claim k/events/tp/prec/rec={int(r['k'])}/{int(r['event_count'])}/{int(r['tp'])}/{float(r['precision'])}/{float(r['recall'])}"
                    f" vs recomputed {m['k']}/{m['events']}/{m['tp']}/{round(m['precision'],4)}/{round(m['recall'],4)}"
                ),
            )
        )
    return findings


def write_report(findings: List[Finding]) -> Path:
    out = REPO / "reports" / "claimed_values_reproducibility_audit_20260406.md"
    n_fail = sum(1 for f in findings if f.status == "FAIL")
    n_pass = sum(1 for f in findings if f.status == "PASS")
    n_warn = sum(1 for f in findings if f.status == "WARN")

    lines: List[str] = []
    lines.append("# Claimed Values Reproducibility Audit")
    lines.append("")
    lines.append(f"- PASS: {n_pass}")
    lines.append(f"- WARN: {n_warn}")
    lines.append(f"- FAIL: {n_fail}")
    lines.append("")
    lines.append("## Detailed Results")
    for f in findings:
        lines.append(f"- [{f.status}] {f.check}: {f.detail}")
    lines.append("")
    lines.append("## Conclusion")
    if n_fail == 0:
        lines.append("All audited claimed values are reproducible from repository source artifacts.")
    else:
        lines.append("Some claimed values are not reproducible from audited source artifacts; review FAIL items.")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    findings: List[Finding] = []
    findings.extend(verify_table1())
    findings.extend(verify_table2())
    findings.extend(verify_table3())
    out = write_report(findings)
    print(f"Wrote: {out}")
    for f in findings:
        print(f"[{f.status}] {f.check}: {f.detail}")


if __name__ == "__main__":
    main()
