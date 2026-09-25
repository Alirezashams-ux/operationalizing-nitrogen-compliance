import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

TAUS = [15.0, 16.0, 17.0]
BUDGETS = [0.05, 0.10]

def mase(y_true, y_pred, y_train, m=1):
    if len(y_train) <= m:
        return np.nan
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y_true - y_pred)) / (denom + 1e-9)

def point_metrics(y_train, y_test, y_pred):
    return dict(
        MAE=float(mean_absolute_error(y_test, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        MASE1=float(mase(y_test, y_pred, y_train, m=1)),
        MASE7=float(mase(y_test, y_pred, y_train, m=7)),
        n_test=int(len(y_test)),
    )

def alarm_metrics(df, tau, r):
    df = df.copy()
    df["event"] = (df["y_true"] >= tau).astype(int)
    n = len(df)
    k = max(1, int(np.ceil(r * n)))
    df = df.sort_values("y_pred", ascending=False)
    alarm = df.head(k)
    tp = int(alarm["event"].sum())
    total_events = int(df["event"].sum())
    precision = tp / k if k > 0 else np.nan
    recall = tp / total_events if total_events > 0 else np.nan
    return dict(n=n, k=k, events=total_events, TP=tp, precision=precision, recall=recall)

def persistence_pred_from_y(y_all, test_idx, lag):
    # For horizon-H target y = TNout(t+H), persistence baseline is y_hat(t+H)=TNout(t)
    # and TNout(t) corresponds to y shifted by lag=H: y[i-lag] = TNout(t).
    y_pred = []
    for i in test_idx:
        j = i - lag
        y_pred.append(y_all[j] if j >= 0 else np.nan)
    return np.array(y_pred, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True, help="forecast horizon (e.g., 1,3,5)")
    ap.add_argument("--n_splits", type=int, default=3)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    NPZ  = ROOT / "features" / f"ulsan_H{args.h}_features.npz"

    pred_dir = ROOT / "results" / "predictions"
    met_dir  = ROOT / "results" / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    dates = z["dates"].astype(str)
    feat_names = z["feature_names"].tolist()

    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    # Candidate ElasticNet configs (small, transparent)
    enet_grid = [
        (0.01, 0.2), (0.1, 0.2), (1.0, 0.2),
        (0.01, 0.5), (0.1, 0.5), (1.0, 0.5),
    ]

    model_specs = [("persistence", None)]
    for a, l1 in enet_grid:
        model_specs.append((f"enet_a{a}_l{l1}", (a, l1)))

    # ---- Run CV, save fold predictions, compute point metrics ----
    point = {}
    for name, spec in model_specs:
        rows = []
        fold_metrics = []
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            y_tr, y_te = y[tr], y[te]
            d_te = dates[te]

            if name == "persistence":
                y_pred = persistence_pred_from_y(y, te, lag=args.h)
                ok = ~np.isnan(y_pred)
                y_te = y_te[ok]
                d_te = d_te[ok]
                y_pred = y_pred[ok]
            else:
                alpha, l1_ratio = spec
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=20000))
                ])
                model.fit(X[tr], y_tr)
                y_pred = model.predict(X[te])

            rows.append(pd.DataFrame({
                "date": d_te,
                "fold": fold,
                "y_true": y_te,
                "y_pred": y_pred
            }))

            m = point_metrics(y_tr, y_te, y_pred)
            m["fold"] = fold
            fold_metrics.append(m)

        df_pred = pd.concat(rows, ignore_index=True)
        out_csv = pred_dir / f"{name}_H{args.h}_preds.csv"
        df_pred.to_csv(out_csv, index=False)

        point[name] = fold_metrics

    # Save point metrics JSON
    out_point = met_dir / f"main_linear_metrics_H{args.h}.json"
    with open(out_point, "w") as f:
        json.dump({"npz": str(NPZ), "features": feat_names, "point_metrics": point}, f, indent=2)

    # ---- Alarm budget metrics (overall + fold-wise) ----
    alarm_rows = []
    for name, _ in model_specs:
        df = pd.read_csv(pred_dir / f"{name}_H{args.h}_preds.csv")
        scopes = [("overall", df)] + [(f"fold{f}", df[df["fold"] == f]) for f in sorted(df["fold"].unique())]
        for scope, dfx in scopes:
            for tau in TAUS:
                for r in BUDGETS:
                    m = alarm_metrics(dfx, tau, r)
                    alarm_rows.append({"model": name, "scope": scope, "tau": tau, "budget_r": r, **m})

    df_alarm = pd.DataFrame(alarm_rows)
    out_alarm = met_dir / f"alarm_budget_H{args.h}.csv"
    df_alarm.to_csv(out_alarm, index=False)

    # ---- Print a small leaderboard (overall MAE + τ=15 budget results) ----
    def avg_metric(model, key):
        vals = [d[key] for d in point[model]]
        return float(np.mean(vals))

    models = [m[0] for m in model_specs]
    leaderboard = sorted([(m, avg_metric(m, "MAE")) for m in models], key=lambda x: x[1])[:8]

    print("\nSaved:")
    print(" ", out_point)
    print(" ", out_alarm)
    print("\nTop models by avg MAE:")
    for m, v in leaderboard:
        print(f"  {m:18s}  MAE={v:.4f}")

    print("\nOverall alarm summary (tau=15):")
    print(df_alarm[(df_alarm.scope=="overall") & (df_alarm.tau==15.0)][["model","budget_r","precision","recall","TP","events","k"]]
          .sort_values(["budget_r","recall"], ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()
