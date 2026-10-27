"""Airline delay risk: predict whether a carrier-airport-month will be a high-delay period.

Data: BTS "Airline On-Time Statistics and Delay Causes" monthly summary (Airline_Delay_Cause.csv),
one row per carrier x airport x month, 2013-2023.

Target: high_delay = 1 when at least 20% of arriving flights that month were delayed 15+ minutes.

Features are built only from traffic volume, market structure, seasonality and disruptions. The
delay-cause columns (carrier_ct, weather_ct, nas_ct, ..., arr_delay minutes) are deliberately
excluded: they are components of the target and would leak it.

Usage: python train.py [--data data/Airline_Delay_Cause.csv] [--out results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

LEAK_COLS = [
    "arr_del15", "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
    "arr_delay", "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=["arr_flights", "arr_del15"]).copy()
    df = df[df["arr_flights"] >= 1]
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"]
    y = (df["delay_rate"] >= 0.20).astype(int).rename("high_delay")

    f = pd.DataFrame(index=df.index)
    # traffic volume
    f["arr_flights"] = df["arr_flights"]
    f["log_arr_flights"] = np.log1p(df["arr_flights"])
    # airport congestion: all arrivals at the airport that month, and how many carriers serve it
    ap_month = df.groupby(["year", "month", "airport"])["arr_flights"]
    f["airport_month_flights"] = ap_month.transform("sum")
    f["airport_month_carriers"] = ap_month.transform("count")
    # carrier scale that month, and the carrier's share of the airport (market share / competition)
    f["carrier_month_flights"] = df.groupby(["year", "month", "carrier"])["arr_flights"].transform("sum")
    f["carrier_share_at_airport"] = df["arr_flights"] / f["airport_month_flights"]
    f["airport_share_of_national"] = f["airport_month_flights"] / df.groupby(["year", "month"])["arr_flights"].transform("sum")
    f["carrier_airports_served"] = df.groupby(["year", "month", "carrier"])["airport"].transform("count")
    # operational disruption that is not part of the target
    f["arr_cancelled"] = df["arr_cancelled"].fillna(0)
    f["arr_diverted"] = df["arr_diverted"].fillna(0)
    f["cancel_rate"] = f["arr_cancelled"] / df["arr_flights"]
    f["divert_rate"] = f["arr_diverted"] / df["arr_flights"]
    # seasonality
    f["year"] = df["year"]
    f["month"] = df["month"]
    f["quarter"] = (df["month"] - 1) // 3 + 1
    f["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    f["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    f["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    # identity of carrier and airport, one-hot encoded
    cats = pd.get_dummies(df[["carrier", "airport"]], prefix=["carrier", "airport"], dtype=np.uint8)
    f = pd.concat([f, cats], axis=1)
    return f, y


def evaluate(name: str, model, X_test, y_test, out: Path) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_high_delay": round(precision_score(y_test, pred), 4),
        "recall_high_delay": round(recall_score(y_test, pred), 4),
        "f1_high_delay": round(f1_score(y_test, pred), 4),
        "recall_low_delay": round(recall_score(y_test, pred, pos_label=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n=== {name} ===")
    for k, v in m.items():
        if k != "confusion_matrix": print(f"{k:>22}: {v}")
    print(classification_report(y_test, pred, target_names=["low delay", "high delay"]))
    fpr, tpr, _ = roc_curve(y_test, proba)
    m["_roc"] = (fpr, tpr)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Airline_Delay_Cause.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print("raw rows:", len(df), "columns:", len(df.columns))
    X, y = build_features(df)
    print("rows after cleaning:", len(X), "| features:", X.shape[1],
          f"({X.shape[1] - X.columns.str.startswith(('carrier_', 'airport_')).sum() + 6} engineered + one-hot ids)")
    print("high-delay share: %.3f" % y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("train:", X_train.shape, "test:", X_test.shape)

    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_m = evaluate("Random Forest", rf, X_test, y_test, out)

    import lightgbm as lgb
    gbm = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                             random_state=42, verbose=-1)
    gbm.fit(X_train, y_train)
    gbm_m = evaluate("LightGBM", gbm, X_test, y_test, out)

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 5))
    for m in (rf_m, gbm_m):
        fpr, tpr = m["_roc"]; plt.plot(fpr, tpr, label=f"{m['model']} (AUC {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("High-delay month: ROC on the held-out 20%"); plt.legend(); plt.tight_layout(); plt.savefig(out / "roc.png", dpi=150)

    imp = pd.Series(gbm.feature_importances_, index=X.columns)
    eng = imp[~imp.index.str.startswith(("carrier_", "airport_")) | imp.index.isin(
        ["carrier_month_flights", "carrier_share_at_airport", "carrier_airports_served", "airport_month_flights", "airport_month_carriers", "airport_share_of_national"])]
    top = eng.sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 5)); top[::-1].plot.barh(); plt.title("LightGBM split importance, engineered features")
    plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=150)

    for m in (rf_m, gbm_m): m.pop("_roc")
    (out / "metrics.json").write_text(json.dumps({"n_train": len(X_train), "n_test": len(X_test),
        "n_features": int(X.shape[1]), "high_delay_share": round(float(y.mean()), 4),
        "models": [rf_m, gbm_m], "top_engineered_features": top.round(1).to_dict()}, indent=2))
    print("\nwrote", out / "metrics.json")


if __name__ == "__main__":
    main()
