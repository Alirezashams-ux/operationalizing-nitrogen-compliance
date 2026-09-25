from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repository root
ULSAN_CSV = ROOT / "data" / "raw" / "Ulsan_Yongsan.csv"
OUT_NPZ   = ROOT / "features" / "ulsan_H1_features.npz"

def pick_col(df, candidates, required=True):
    # case-insensitive exact match
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise ValueError(
            f"Missing required column. Tried: {candidates}\nAvailable: {list(df.columns)}"
        )
    return None

def add_time_features(df, date_col):
    doy = df[date_col].dt.dayofyear.values
    df["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    df["cos_doy"] = np.cos(2*np.pi*doy/365.25)
    return df

def make_features(df, horizon=1):
    # drop junk unnamed cols
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    date_col  = pick_col(df, ["Date", "date"])
    tnout_col = pick_col(df, ["TNout", "tnout", "TN_out", "T-Nout", "Tn_out"])
    inflow_col= pick_col(df, ["Inflow", "inflow", "Flow", "Qin", "InfluentFlow"])
    tnin_col  = pick_col(df, ["TNin", "tnin", "TN_in", "T-Nin", "Tn_in"])
    tocin_col = pick_col(df, ["TOCin", "tocin", "TOC_in", "TOC"])
    temp_col  = pick_col(df, ["temp_mean_c", "Temp", "Temperature", "temp"], required=True)
    precip_col= pick_col(df, ["precip_total_mm", "Precipitation", "Rainfall", "rain"], required=True)
    bodin_col = pick_col(df, ["BODin", "bodin", "BOD_in", "BOD"], required=False)

    # parse date + sort
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # numeric coercion
    for c in [tnout_col, inflow_col, tnin_col, tocin_col, temp_col, precip_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if bodin_col is not None:
        df[bodin_col] = pd.to_numeric(df[bodin_col], errors="coerce")

    # carbon proxy
    df["C_N"] = df[tocin_col] / (df[tnin_col] + 1e-6)

    # target: TNout(t+h)
    df["y"] = df[tnout_col].shift(-horizon)

    # memory features
    df["TNout_lag1"]   = df[tnout_col].shift(1)
    df["TNout_roll7"]  = df[tnout_col].shift(1).rolling(7).mean()
    df["TNout_roll14"] = df[tnout_col].shift(1).rolling(14).mean()

    # influent/load smoothers (use day t and earlier only)
    df["Inflow"]       = df[inflow_col]
    df["TNin"]         = df[tnin_col]
    df["TOCin"]        = df[tocin_col]
    df["temp_mean_c"]  = df[temp_col]
    df["precip_total_mm"] = df[precip_col]

    df["Inflow_roll7"] = df["Inflow"].rolling(7).mean()
    df["TNin_roll7"]   = df["TNin"].rolling(7).mean()
    df["TOCin_roll7"]  = df["TOCin"].rolling(7).mean()

    if bodin_col is not None:
        df["BODin"] = df[bodin_col]
        df["BODin_roll7"] = df["BODin"].rolling(7).mean()
    else:
        df["BODin"] = np.nan
        df["BODin_roll7"] = np.nan

    # weather aggregates
    df["temp_roll7"]  = df["temp_mean_c"].rolling(7).mean()
    df["precip_sum3"] = df["precip_total_mm"].rolling(3).sum()

    # seasonality
    df = add_time_features(df, date_col)

    feature_cols = [
        "TNout_lag1","TNout_roll7","TNout_roll14",
        "Inflow","Inflow_roll7",
        "TNin","TNin_roll7",
        "TOCin","TOCin_roll7",
        "BODin","BODin_roll7",
        "C_N",
        "temp_mean_c","temp_roll7",
        "precip_total_mm","precip_sum3",
        "sin_doy","cos_doy"
    ]

    out = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    X = out[feature_cols].to_numpy(np.float32)
    y = out["y"].to_numpy(np.float32)
    dates = out[date_col].dt.strftime("%Y-%m-%d").to_numpy()

    used = dict(date=date_col, tnout=tnout_col, inflow=inflow_col, tnin=tnin_col,
                tocin=tocin_col, temp=temp_col, precip=precip_col, bodin=bodin_col)

    return X, y, dates, feature_cols, used

def main():
    if not ULSAN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {ULSAN_CSV}")

    df = pd.read_csv(ULSAN_CSV)
    X, y, dates, feature_names, used = make_features(df, horizon=1)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        X=X, y=y, dates=dates,
        feature_names=np.array(feature_names, dtype=object)
    )

    print("Saved:", OUT_NPZ)
    print("X:", X.shape, "y:", y.shape)
    print("Dates:", dates[0], "->", dates[-1])
    print("Used columns mapping:", used)

if __name__ == "__main__":
    main()

