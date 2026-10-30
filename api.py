import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

model = joblib.load("artifacts/xgb.joblib")
meta = json.load(open("artifacts/meta.json"))
FEATURES = meta["features"]
THRESHOLD = meta["threshold"]
CATEGORIES = [f[4:] for f in FEATURES if f.startswith("cat_")]
explainer = shap.TreeExplainer(model)
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

app = FastAPI(title="API de détection de fraude explicable", version="1.0")


class Transaction(BaseModel):
    amt: float = Field(..., gt=0, description="Montant en $")
    hour: float = Field(..., ge=0, lt=24, description="Heure décimale (14.5 = 14h30)")
    dayofweek: int = Field(..., ge=0, le=6, description="0 = lundi")
    category: str
    tx_1h: int = Field(0, ge=0, description="Transactions de la carte sur la dernière heure")
    tx_24h: int = Field(0, ge=0, description="Transactions de la carte sur les dernières 24 h")
    amt_ratio_30d: float = Field(1.0, ge=0, description="Montant / moyenne des 30 derniers jours")
    hours_since_prev: Optional[float] = Field(None, ge=0, description="Heures depuis la transaction précédente")
    dist_km: float = Field(0.0, ge=0)
    age: float = Field(..., gt=0)
    city_pop: int = Field(..., ge=0)


def to_row(t: Transaction) -> pd.DataFrame:
    if t.category not in CATEGORIES:
        raise HTTPException(422, f"Catégorie inconnue. Valeurs possibles : {CATEGORIES}")
    gap_s = 30 * 86400 if t.hours_since_prev is None else min(t.hours_since_prev * 3600, 30 * 86400)
    row = {
        "log_amt": np.log1p(t.amt),
        "hour_sin": np.sin(2 * np.pi * t.hour / 24),
        "hour_cos": np.cos(2 * np.pi * t.hour / 24),
        "dayofweek": t.dayofweek,
        "tx_1h": t.tx_1h,
        "tx_24h": t.tx_24h,
        "amt_ratio_30d": t.amt_ratio_30d,
        "gap_prev_s": np.log1p(gap_s),
        "dist_km": t.dist_km,
        "age": t.age,
        "log_city_pop": np.log1p(t.city_pop),
    }
    for c in CATEGORIES:
        row[f"cat_{c}"] = float(c == t.category)
    return pd.DataFrame([row])[FEATURES]


def describe(name, v, hour):
    if name == "log_amt": return f"Montant élevé : {np.expm1(v):,.0f} $"
    if name == "amt_ratio_30d": return f"Montant = {v:.1f} × la moyenne des 30 derniers jours de la carte"
    if name in ("hour_sin", "hour_cos"): return f"Heure de la transaction : {hour} h"
    if name == "tx_1h": return f"{int(v)} transaction(s) sur la dernière heure"
    if name == "tx_24h": return f"{int(v)} transaction(s) sur les dernières 24 h"
    if name == "gap_prev_s": return f"Délai depuis la transaction précédente : {np.expm1(v) / 3600:.1f} h"
    if name == "dist_km": return f"Distance client-marchand : {v:.0f} km"
    if name == "age": return f"Âge du titulaire : {v:.0f} ans"
    if name == "log_city_pop": return f"Population de la ville : {np.expm1(v):,.0f} habitants"
    if name == "dayofweek": return f"Jour : {JOURS[int(v)]}"
    if name.startswith("cat_"): return f"Catégorie : {name[4:]}"
    return f"{name} = {v:.2f}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"model": "XGBoost", "threshold": THRESHOLD, "c_alert": meta["c_alert"],
            "features": FEATURES, "categories": CATEGORIES}


@app.post("/predict")
def predict(t: Transaction):
    p = float(model.predict_proba(to_row(t))[0, 1])
    return {"fraud_probability": round(p, 5), "threshold": THRESHOLD, "alert": p >= THRESHOLD}


@app.post("/explain")
def explain(t: Transaction, top: int = 5):
    X = to_row(t)
    p = float(model.predict_proba(X)[0, 1])
    e = explainer(X)[0]

    agg = {}
    for name, v, c in zip(e.feature_names, e.data, e.values):
        if c <= 0 or (name.startswith("cat_") and v < 0.5):
            continue
        lab = describe(name, v, int(t.hour))
        agg[lab] = agg.get(lab, 0.0) + float(c)
    reasons = [{"motif": k, "contribution": round(v, 3)}
               for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:top]]

    return {"fraud_probability": round(p, 5), "threshold": THRESHOLD, "alert": p >= THRESHOLD,
            "reason_codes": reasons,
            "shap_values": {n: round(float(c), 4) for n, c in zip(e.feature_names, e.values)},
            "base_value": round(float(e.base_values), 4)}