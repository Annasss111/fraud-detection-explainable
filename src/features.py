import numpy as np
import pandas as pd

DAY = 86400


def _count_prev(t, window):
    """Nb de transactions précédentes de la même carte dans la fenêtre (courante exclue)."""
    left = np.searchsorted(t, t - window, side="left")
    return np.arange(len(t)) - left


def _mean_prev(t, x, window):
    """Moyenne des montants précédents dans la fenêtre (NaN s'il n'y a pas d'historique)."""
    left = np.searchsorted(t, t - window, side="left")
    i = np.arange(len(t))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    n = i - left
    s = cs[i] - cs[left]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, s / n, np.nan)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def build_features(df):
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    ts = df["trans_date_trans_time"]
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    x = df["amt"].to_numpy()

    n = len(df)
    tx_1h = np.zeros(n)
    tx_24h = np.zeros(n)
    mean_30d = np.full(n, np.nan)
    gap = np.full(n, np.nan)

    # Une boucle par carte : les positions sont déjà en ordre chronologique
    for _, idx in df.groupby("cc_num").indices.items():
        tt, xx = t[idx], x[idx]
        tx_1h[idx] = _count_prev(tt, 3600)
        tx_24h[idx] = _count_prev(tt, DAY)
        mean_30d[idx] = _mean_prev(tt, xx, 30 * DAY)
        gap[idx] = np.r_[np.nan, np.diff(tt)]

    h = ts.dt.hour + ts.dt.minute / 60
    out = pd.DataFrame({
        "trans_date_trans_time": ts,
        "cc_num": df["cc_num"],
        "category": df["category"],
        "is_fraud": df["is_fraud"],
        "amt": x,
        "log_amt": np.log1p(x),
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dayofweek": ts.dt.dayofweek,
        "tx_1h": tx_1h,
        "tx_24h": tx_24h,
        # Sans historique : ratio neutre (1.0) et écart plafonné à 30 jours
        "amt_ratio_30d": np.nan_to_num(x / mean_30d, nan=1.0),
        "gap_prev_s": np.nan_to_num(gap, nan=30 * DAY).clip(max=30 * DAY),
        "dist_km": _haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"]),
        "age": (ts - pd.to_datetime(df["dob"])).dt.days / 365.25,
        "log_city_pop": np.log1p(df["city_pop"]),
    })
    return out