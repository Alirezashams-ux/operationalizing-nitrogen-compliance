import argparse, json
import numpy as np
from pathlib import Path
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    NPZ  = ROOT / "features" / f"ulsan_H{args.h}_features_v2.npz"
    OUTD = ROOT / "results" / "hgbr"
    OUTD.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=True)
    X,y = z["X"], z["y"]
    tscv = TimeSeriesSplit(n_splits=args.splits)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 200, 1800),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 30.0, log=True),
            max_bins=trial.suggest_int("max_bins", 64, 255),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        maes=[]
        for tr,te in tscv.split(X):
            m = HistGradientBoostingRegressor(**params)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            maes.append(mean_absolute_error(y[te], pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)

    out = {"h":args.h,"npz":str(NPZ),"best_mae":study.best_value,"best_params":study.best_params}
    path = OUTD / f"hgbr_optuna_H{args.h}_v2.json"
    path.write_text(json.dumps(out, indent=2))
    print("Saved:", path)
    print("Best MAE:", study.best_value)
    print("Best params:", study.best_params)

if __name__=="__main__":
    main()
