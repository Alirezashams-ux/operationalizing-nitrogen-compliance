import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

TAUS = [15.0,16.0,17.0]
BUDGETS = [0.05,0.10]

def alarm_metrics(df, tau, r):
    df = df.copy()
    df["event"] = (df["y_true"] >= tau).astype(int)
    n = len(df)
    k = max(1, int(np.ceil(r*n)))
    df = df.sort_values("y_pred", ascending=False)
    alarm = df.head(k)
    tp = int(alarm["event"].sum())
    total = int(df["event"].sum())
    return dict(n=n,k=k,events=total,TP=tp,
                precision=tp/k if k else np.nan,
                recall=tp/total if total else np.nan)

def persistence_pred_from_y(y_all, test_idx, lag):
    # for horizon-H target y=TNout(t+H), persistence is TNout(t)=y[i-lag]
    out=[]
    for i in test_idx:
        j=i-lag
        out.append(y_all[j] if j>=0 else np.nan)
    return np.array(out, float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    NPZ = ROOT / "features" / f"ulsan_H{args.h}_features_v2.npz"
    pred_dir = ROOT / "results" / "predictions"
    met_dir  = ROOT / "results" / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=True)
    X,y = z["X"], z["y"]
    dates = z["dates"].astype(str)

    tscv = TimeSeriesSplit(n_splits=args.splits)

    models = {
        "persistence_v2": None,
        "enet_a0.1_l0.5_v2": Pipeline([("scaler", StandardScaler()),
                                       ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=20000))]),
    }

    # save preds
    for name, mdl in models.items():
        rows=[]
        for fold,(tr,te) in enumerate(tscv.split(X),1):
            if name.startswith("persistence"):
                yp = persistence_pred_from_y(y, te, lag=args.h)
                ok = ~np.isnan(yp)
                df = pd.DataFrame({"date":dates[te][ok],"fold":fold,"y_true":y[te][ok],"y_pred":yp[ok]})
            else:
                mdl.fit(X[tr], y[tr])
                yp = mdl.predict(X[te])
                df = pd.DataFrame({"date":dates[te],"fold":fold,"y_true":y[te],"y_pred":yp})
            rows.append(df)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(pred_dir / f"{name}_H{args.h}_preds.csv", index=False)

    # alarms
    alarm_rows=[]
    for name in models:
        df = pd.read_csv(pred_dir / f"{name}_H{args.h}_preds.csv")
        for scope, dfx in [("overall", df)] + [(f"fold{f}", df[df.fold==f]) for f in sorted(df.fold.unique())]:
            for tau in TAUS:
                for r in BUDGETS:
                    m = alarm_metrics(dfx, tau, r)
                    alarm_rows.append({"model":name,"scope":scope,"tau":tau,"budget_r":r,**m})
    out_alarm = pd.DataFrame(alarm_rows)
    out_path = met_dir / f"alarm_budget_v2_H{args.h}.csv"
    out_alarm.to_csv(out_path, index=False)
    print("Saved:", out_path)

    key = out_alarm[(out_alarm.scope=="overall") & (out_alarm.budget_r==0.05) & (out_alarm.tau.isin([15.0,16.0]))]
    print("\nKey rows (overall, r=5%, tau=15/16):")
    print(key[["model","tau","precision","recall","TP","events","k"]].sort_values(["tau","recall"], ascending=[True,False]).to_string(index=False))

if __name__=="__main__":
    main()
