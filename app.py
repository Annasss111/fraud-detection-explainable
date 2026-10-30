import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             fbeta_score, matthews_corrcoef)

st.set_page_config(page_title="Détection de fraude explicable", layout="wide")

SCORE_COL = {"XGBoost": "score_xgb", "Isolation Forest": "score_iforest", "Autoencodeur": "score_ae"}
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


# ---------- Chargement ----------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/xgb.joblib")
    meta = json.load(open("artifacts/meta.json"))
    return model, meta, shap.TreeExplainer(model)

@st.cache_data
def load_data():
    df = pd.read_parquet("artifacts/test_scored.parquet")
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    df["category"] = df[cat_cols].idxmax(axis=1).str.replace("cat_", "", regex=False)
    ang = np.arctan2(df["hour_sin"], df["hour_cos"])
    df["hour"] = (np.floor(ang * 24 / (2 * np.pi) + 1e-6) % 24).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)

model, meta, explainer = load_model()
df = load_data()
y = df["is_fraud"].to_numpy()
amt = df["amt"].to_numpy()
default_k = int(meta["k_ratio"] * len(df))


# ---------- Fonctions ----------
def topk_mask(score, k):
    m = np.zeros(len(score), dtype=bool)
    m[np.argsort(-score)[:k]] = True
    return m

def metrics_at(alert, c_alert):
    n = int(alert.sum())
    tp = int((alert & (y == 1)).sum())
    missed = float(amt[(y == 1) & ~alert].sum())
    return {"Alertes": n,
            "Précision": tp / n if n else 0.0,
            "Rappel": tp / y.sum(),
            "F2": fbeta_score(y, alert, beta=2, zero_division=0),
            "MCC": matthews_corrcoef(y, alert),
            "Coût ($)": missed + c_alert * n,
            "Coût sans modèle ($)": float(amt[y == 1].sum())}

@st.cache_data
def compare(_df, k):
    rows, curves = {}, {}
    for name, col in SCORE_COL.items():
        s = _df[col].to_numpy()
        a = topk_mask(s, k)
        tp = (a & (y == 1)).sum()
        rows[name] = {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s),
                      f"Précision@{k}": tp / k, f"Rappel@{k}": tp / y.sum(),
                      "F2@k": fbeta_score(y, a, beta=2), "MCC@k": matthews_corrcoef(y, a)}
        p, r, _ = precision_recall_curve(y, s)
        step = max(1, len(p) // 3000)
        curves[name] = (r[::step], p[::step])
    return pd.DataFrame(rows).T, curves

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

def reason_codes(exp_row, hour, top=5):
    agg = {}
    for name, v, c in zip(exp_row.feature_names, exp_row.data, exp_row.values):
        if c <= 0 or (name.startswith("cat_") and v < 0.5):
            continue
        lab = describe(name, v, hour)
        agg[lab] = agg.get(lab, 0.0) + float(c)
    return sorted(agg.items(), key=lambda t: -t[1])[:top]


# ---------- Barre latérale : modèle, seuil, budget ----------
with st.sidebar:
    st.header("Paramètres d'alerte")
    model_name = st.selectbox("Modèle", list(SCORE_COL))
    score = df[SCORE_COL[model_name]].to_numpy()
    c_alert = st.slider("Coût d'analyse par alerte ($)", 1.0, 50.0, float(meta["c_alert"]), 1.0)
    mode = st.radio("Mode d'alerte", ["Seuil de score", "Budget d'alertes k"])

    if mode == "Seuil de score":
        lo, hi = float(score.min()), float(score.max())
        step = (hi - lo) / 10000
        d = meta["threshold"] if model_name == "XGBoost" else float(np.sort(score)[-default_k])
        d = lo + round((d - lo) / step) * step
        thr = st.slider("Seuil", lo, hi, float(d), step, format="%.4f", key=f"thr_{model_name}")
        alert = score >= thr
    else:
        k = st.slider("Budget k (alertes)", 100, 10000, default_k, 100)
        alert = topk_mask(score, k)

st.title("Détection de fraude explicable")
st.caption("Jeu Sparkov (synthétique) · période de test · séparation chronologique 60/20/20")
tab1, tab2, tab3, tab4 = st.tabs(["1 · Données", "2 · Comparaison des modèles",
                                  "3 · Seuil et budget", "4 · Explication d'une alerte"])

# ---------- Onglet 1 : données ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraudes", f"{int(y.sum()):,}")
    c3.metric("Taux de fraude", f"{y.mean() * 100:.2f} %")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Montants par classe")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = np.linspace(0, np.log10(amt.max()), 60)
        ax.hist(np.log10(amt[y == 0]), bins, density=True, alpha=0.6, label="normale")
        ax.hist(np.log10(amt[y == 1]), bins, density=True, alpha=0.6, label="fraude")
        ax.set_xlabel("log10(montant)"); ax.legend()
        st.pyplot(fig)
    with g2:
        st.subheader("Taux de fraude par mois (%)")
        m = df.groupby(df["date"].dt.to_period("M"))["is_fraud"].mean() * 100
        m.index = m.index.astype(str)
        st.line_chart(m)

    g3, g4 = st.columns(2)
    with g3:
        st.subheader("Par heure (%)")
        st.bar_chart(df.groupby("hour")["is_fraud"].mean() * 100)
    with g4:
        st.subheader("Par catégorie (%)")
        st.bar_chart((df.groupby("category")["is_fraud"].mean() * 100).sort_values())

# ---------- Onglet 2 : comparaison ----------
with tab2:
    table, curves = compare(df, default_k)
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (r, p) in curves.items():
        ax.plot(r, p, label=name)
    ax.axhline(y.mean(), ls="--", c="gray", label="hasard")
    ax.set_xlabel("Rappel"); ax.set_ylabel("Précision"); ax.legend()
    st.pyplot(fig)

# ---------- Onglet 3 : seuil et budget ----------
with tab3:
    m = metrics_at(alert, c_alert)
    cols = st.columns(5)
    cols[0].metric("Alertes", f"{m['Alertes']:,}")
    cols[1].metric("Précision", f"{m['Précision']:.1%}")
    cols[2].metric("Rappel", f"{m['Rappel']:.1%}")
    cols[3].metric("F2", f"{m['F2']:.3f}")
    cols[4].metric("MCC", f"{m['MCC']:.3f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coût total", f"{m['Coût ($)']:,.0f} $")
    c2.metric("Coût sans modèle", f"{m['Coût sans modèle ($)']:,.0f} $")
    c3.metric("Économie", f"{1 - m['Coût ($)'] / m['Coût sans modèle ($)']:.1%}")

    order = np.argsort(-score)
    fa = (amt * y)[order]
    cost = fa.sum() - np.concatenate([[0], np.cumsum(fa)]) + c_alert * np.arange(len(y) + 1)
    N = 15000
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.arange(N), cost[:N])
    ax.axvline(m["Alertes"], c="red", ls="--", label="réglage actuel")
    ax.axvline(int(np.argmin(cost)), c="green", ls=":", label="coût minimal")
    ax.set_xlabel("Nombre d'alertes traitées"); ax.set_ylabel("Coût ($)"); ax.legend()
    st.pyplot(fig)

# ---------- Onglet 4 : explication ----------
with tab4:
    st.caption("Explications SHAP calculées pour XGBoost. Contributions en log-odds : "
               "positif = pousse vers la fraude.")
    cand = np.where(alert)[0]
    cand = cand[np.argsort(-df["score_xgb"].to_numpy()[cand])][:100]
    if len(cand) == 0:
        st.info("Aucune alerte avec ce réglage.")
    else:
        i = st.selectbox(
            "Alerte à expliquer (100 meilleurs scores XGBoost parmi les alertes)", cand,
            format_func=lambda j: (f"{df.loc[j, 'date']:%d/%m %H:%M} | {df.loc[j, 'amt']:.2f} $ | "
                                   f"{df.loc[j, 'category']} | score {df.loc[j, 'score_xgb']:.3f} | "
                                   f"{'FRAUDE' if df.loc[j, 'is_fraud'] else 'normale'}"))
        Xr = df.loc[[i], meta["features"]]
        exp = explainer(Xr)[0]
        left, right = st.columns([1, 1.3])
        with left:
            st.subheader("Codes motifs")
            for lab, c in reason_codes(exp, int(df.loc[i, "hour"])):
                st.write(f"- **{lab}** (+{c:.2f})")
        with right:
            shap.plots.waterfall(exp, max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)