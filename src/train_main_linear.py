import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
NPZ  = ROOT / "features" / "ulsan_H1_features.npz"
OUT  = ROOT / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

def mase(y_true, y_pred, y_train, m=1):
    # MASE denominator: mean abs naive seasonal difference on training
    # for m=1, naive is y_{t-1}; for m=7 weekly naive, set m=7
    if len(y_train) <= m:
        return np.nan
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y_true - y_pred)) / (denom + 1e-9)

def eval_fold(y_train, y_test, y_pred):
    return dict(
        MAE=float(mean_absolute_error(y_test, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        MASE1=float(mase(y_test, y_pred, y_train, m=1)),
        MASE7=float(mase(y_test, y_pred, y_train, m=7)),
        n_test=int(len(y_test)),
    )

def persistence_pred(y_all, test_idx, lag=1):
    # predict y_t = y_{t-lag}
    y_pred = []
    for i in test_idx:
        j = i - lag
        y_pred.append(y_all[j] if j >= 0 else np.nan)
    return np.array(y_pred, dtype=float)

def run():
    z = np.load(NPZ, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    dates = z["dates"]
    feat_names = z["feature_names"].tolist()

    tscv = TimeSeriesSplit(n_splits=3)

    results = {}

    # ---- Baselines ----
    for name, lag in [("persistence_1", 1), ("seasonal_7", 7)]:
        fold_metrics = []
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            y_pred = persistence_pred(y, te, lag=lag)
            # drop first few if nan (shouldn't happen with our feature build, but safe)
            ok = ~np.isnan(y_pred)
            m = eval_fold(y[tr], y[te][ok], y_pred[ok])
            m["fold"] = fold
            fold_metrics.append(m)
        results[name] = fold_metrics

    # ---- Ridge grid (transparent) ----
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    for a in ridge_alphas:
        name = f"ridge_alpha_{a}"
        fold_metrics = []
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=a, random_state=42))
        ])
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            m = eval_fold(y[tr], y[te], pred)
            m["fold"] = fold
            fold_metrics.append(m)
        results[name] = fold_metrics

    # ---- ElasticNet small grid ----
    enet_grid = [
        (0.01, 0.2), (0.1, 0.2), (1.0, 0.2),
        (0.01, 0.5), (0.1, 0.5), (1.0, 0.5),
    ]
    for alpha, l1_ratio in enet_grid:
        name = f"enet_a{alpha}_l{l1_ratio}"
        fold_metrics = []
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=20000))
        ])
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            m = eval_fold(y[tr], y[te], pred)
            m["fold"] = fold
            fold_metrics.append(m)
        results[name] = fold_metrics

    out_path = OUT / "main_linear_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"npz": str(NPZ), "features": feat_names, "results": results}, f, indent=2)

    print("Saved metrics:", out_path)

if __name__ == "__main__":
    run()
