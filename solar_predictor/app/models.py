"""Model training, backtesting and prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from features import consumption_features, ev_features, solar_features

log = logging.getLogger("solar_predictor.models")

SUN_MIN = -3.0  # degrees; below this solar is forced to zero


def _gbr(**kw) -> HistGradientBoostingRegressor:
    p = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=0.5, random_state=0)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def clean(actuals: pd.DataFrame) -> pd.DataFrame:
    """Drop counter resets / glitches from hourly kWh actuals."""
    a = actuals.copy()
    for c in a.columns:
        s = a[c]
        s = s.where(s >= -0.05)
        hi = s.quantile(0.999)
        s = s.where(s <= max(hi * 1.5, 0.5))
        a[c] = s.clip(lower=0)
    return a


def accuracy(pred: pd.Series, actual: pd.Series) -> float | None:
    """100 * (1 - sum|err| / sum actual), on aligned series (e.g. daily totals)."""
    m = pred.notna() & actual.notna()
    tot = actual[m].sum()
    if m.sum() == 0 or tot <= 0:
        return None
    return round(float(100 * (1 - (pred[m] - actual[m]).abs().sum() / tot)), 1)


@dataclass
class Predictor:
    lat: float
    lon: float
    tz: str
    arrays: list
    holidays_set: object
    free_start: float = 11.0
    free_end: float = 14.0
    solar_cap: float = 0.0
    ev_cap: float = 0.0
    m_solar: object = None
    m_cons: object = None   # base load (house minus EV charging when EV is metered)
    m_ev: object = None
    m_grid: object = None
    solar_cols: list = field(default_factory=list)
    cons_cols: list = field(default_factory=list)
    ev_cols: list = field(default_factory=list)
    grid_cols: list = field(default_factory=lambda: ["cons", "base", "ev", "solar", "deficit", "base_deficit",
                                                     "surplus", "hour", "free"])
    trained_rows: int = 0
    backtest: dict = field(default_factory=dict)

    # ---------- features ----------
    def _sf(self, wx):
        return solar_features(wx, self.lat, self.lon, self.tz, self.arrays)

    def _cf(self, wx, base_hist, solar_pred):
        f = consumption_features(wx, base_hist, self.lat, self.lon, self.tz, self.holidays_set, solar_pred)
        day = pd.DatetimeIndex(wx.index.tz_convert(self.tz).normalize())
        f["solar_day"] = f["solar_pred"].groupby(day.values).transform("sum").values
        return f

    def _ef(self, wx, solar_pred, ev_hist, soc_hist):
        return ev_features(wx, self.tz, self.holidays_set, solar_pred, ev_hist, soc_hist,
                           self.free_start, self.free_end)

    def _gf(self, cons, solar, ev, idx):
        loc = idx.tz_convert(self.tz)
        c, s, e = np.asarray(cons, float), np.asarray(solar, float), np.asarray(ev, float)
        b = c - e
        return pd.DataFrame({"cons": c, "base": b, "ev": e, "solar": s, "deficit": np.maximum(c - s, 0),
                             "base_deficit": np.maximum(b - s, 0), "surplus": np.maximum(s - c, 0),
                             "hour": loc.hour.values, "free": self.in_free(idx).astype(int)}, index=idx)

    # ---------- solar ----------
    def predict_solar(self, wx: pd.DataFrame) -> pd.Series:
        X = self._sf(wx)
        p = self.m_solar.predict(X[self.solar_cols])
        p = np.clip(p, 0, self.solar_cap or None)
        p[X["sun_elev"].values <= SUN_MIN] = 0.0
        return pd.Series(p, index=wx.index, name="solar")

    @staticmethod
    def _split(act: pd.DataFrame, min_rows: int = 0) -> pd.DataFrame:
        """Add 'base' = consumption minus metered EV charging."""
        a = act.copy()
        if "ev" in a and a["ev"].notna().sum() > min_rows:
            ev = a["ev"].clip(lower=0)
            a["ev"] = ev
            a["base"] = (a["cons"] - ev.fillna(0)).clip(lower=0)
            a.loc[ev.isna(), "base"] = np.nan
        else:
            a["ev"] = np.nan
            a["base"] = a["cons"]
        return a

    # ---------- training ----------
    def _fit(self, wx: pd.DataFrame, act: pd.DataFrame) -> None:
        act = self._split(act, min_rows=24 * 30)
        idx = wx.index.intersection(act.index)
        wx, act = wx.loc[idx], act.loc[idx]

        Xs = self._sf(wx)
        self.solar_cols = list(Xs.columns)
        day = (Xs["sun_elev"] > SUN_MIN) & act["solar"].notna() & Xs["shortwave_radiation"].notna()
        self.solar_cap = float(act["solar"].quantile(0.999) * 1.05)
        self.m_solar = _gbr().fit(Xs.loc[day, self.solar_cols], act.loc[day, "solar"])
        solar_hat = self.predict_solar(wx)

        Xc = self._cf(wx, act["base"].dropna(), solar_hat)
        self.cons_cols = list(Xc.columns)
        ok = act["base"].notna() & Xc["lag_daily_7d"].notna()
        self.m_cons = _gbr().fit(Xc.loc[ok, self.cons_cols], act.loc[ok, "base"])

        self.m_ev = None
        if act["ev"].notna().sum() > 24 * 30:
            Xe = self._ef(wx, solar_hat, act["ev"].dropna(), act.get("soc"))
            self.ev_cols = list(Xe.columns)
            oke = act["ev"].notna() & Xe["lag_daily_7d"].notna()
            self.ev_cap = float(act["ev"].max() * 1.05)
            # charging is all-or-nothing: classify "charging this hour", then size it when charging
            y = act.loc[oke, "ev"]
            on = y > 0.5
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=10,
                                                 random_state=0).fit(Xe.loc[oke, self.ev_cols], on)
            reg = _gbr(min_samples_leaf=5).fit(Xe.loc[oke][on.values][self.ev_cols], y[on]) if on.sum() > 20 else None
            self.m_ev = (clf, reg, float(y[on].mean()) if on.any() else 0.0)

        g = act[["cons", "solar", "grid"]].join(act["ev"].fillna(0)).dropna()
        Xg = self._gf(g["cons"], g["solar"], g["ev"], g.index)
        self.m_grid = _gbr(max_iter=300).fit(Xg[self.grid_cols], g["grid"])
        self.trained_rows = int(day.sum())

    def predict(self, wx: pd.DataFrame, hist: pd.DataFrame, live_soc: float | None = None) -> pd.DataFrame:
        """hist: recent hourly actuals with cons (and ev, soc when available)."""
        h = self._split(hist) if self.m_ev is not None else self._split(hist.drop(columns="ev", errors="ignore"))
        solar = self.predict_solar(wx)
        Xc = self._cf(wx, h["base"].dropna(), solar)
        base = np.clip(self.m_cons.predict(Xc[self.cons_cols]), 0, None)
        if self.m_ev is not None:
            Xe = self._ef(wx, solar, h["ev"].dropna(), h.get("soc"))
            if live_soc is not None:  # today: use the car's current charge level
                loc = wx.index.tz_convert(self.tz)
                today = pd.Timestamp.now(tz=self.tz).normalize()
                Xe.loc[(loc >= today) & (loc < today + pd.Timedelta(days=1)), "soc_eve"] = live_soc
            clf, reg, typical = self.m_ev
            p_on = clf.predict_proba(Xe[self.ev_cols])[:, 1]
            size = reg.predict(Xe[self.ev_cols]) if reg is not None else np.full(len(Xe), typical)
            ev = np.where(p_on >= 0.5, np.clip(size, 0, self.ev_cap or None), 0.0)
        else:
            ev = np.zeros(len(wx))
        cons = base + ev
        Xg = self._gf(cons, solar.values, ev, wx.index)
        grid = np.clip(self.m_grid.predict(Xg[self.grid_cols]), 0, None)
        grid = np.minimum(grid, cons)  # can't import more than the house uses (no battery)
        return pd.DataFrame({"solar": solar.values, "base": base, "ev": ev, "cons": cons, "grid": grid},
                            index=wx.index)

    def predict_grid(self, cons, solar, ev, idx) -> np.ndarray:
        Xg = self._gf(cons, solar, ev, idx)
        g = np.clip(self.m_grid.predict(Xg[self.grid_cols]), 0, None)
        return np.minimum(g, np.asarray(cons, float))

    def in_free(self, idx: pd.DatetimeIndex) -> np.ndarray:
        loc = idx.tz_convert(self.tz)
        h = loc.hour.values + loc.minute.values / 60
        return (h >= self.free_start) & (h < self.free_end)

    def train(self, wx: pd.DataFrame, actuals: pd.DataFrame, holdout_days: int = 28) -> dict:
        """Backtest on the last `holdout_days` local days, then refit on everything."""
        act = clean(actuals)
        loc_day = act.index.tz_convert(self.tz).normalize()
        days = sorted(set(loc_day))
        bt = {}
        if len(days) > holdout_days + 45:
            cut = days[-holdout_days - 1]  # exclude today (partial)
            train_mask = loc_day < cut
            test_mask = (loc_day >= cut) & (loc_day < days[-1])
            self._fit(wx, act[train_mask])
            test_idx = act.index[test_mask].intersection(wx.index)
            # lag/state features only look back to the previous evening, so full history mirrors live use
            pred = self.predict(wx.loc[test_idx], act)
            a = self._split(act.loc[test_idx]) if self.m_ev is not None else \
                self._split(act.loc[test_idx].drop(columns="ev", errors="ignore"))
            free = self.in_free(test_idx)
            pred["grid_paid"] = np.where(free, 0.0, pred["grid"])
            a["grid_paid"] = np.where(free, 0.0, a["grid"])
            d = pred.index.tz_convert(self.tz).normalize()
            pdaily = pred.groupby(d).sum()
            adaily = a.groupby(d).sum(min_count=20)
            keys = ["solar", "cons", "base", "ev", "grid", "grid_paid"]
            bt = {k: accuracy(pdaily[k], adaily[k]) for k in keys}
            bt["solar_hourly"] = accuracy(pred["solar"], a["solar"])
            bt["days"] = int(adaily["solar"].notna().sum())
            r = lambda v: None if pd.isna(v) else round(float(v), 1)  # noqa: E731
            bt["daily"] = [{"date": str(k.date()), **{f"{c}_pred": r(pdaily.at[k, c]) for c in keys},
                            **{f"{c}_actual": r(adaily.at[k, c]) for c in keys}} for k in pdaily.index]
            log.info("Backtest (%s days): solar %s%%, consumption %s%% (base %s%%, EV %s%%), "
                     "grid %s%%, paid grid %s%%", bt["days"], bt["solar"], bt["cons"], bt["base"],
                     bt["ev"], bt["grid"], bt["grid_paid"])
        self._fit(wx, act)
        self.backtest = bt
        return bt


def simulate_battery(df: pd.DataFrame, cap_kwh: float, reserve_pct: float, max_kw: float,
                     soc_kwh: float, eff: float = 0.92) -> pd.DataFrame:
    """Hour-by-hour battery: soak up surplus, cover deficit, report grid import."""
    out = df.copy()
    reserve = cap_kwh * reserve_pct / 100
    soc = min(max(soc_kwh, reserve), cap_kwh)
    grid, socs = [], []
    for s, c in zip(df["solar"].values, df["cons"].values):
        net = s - c
        if net >= 0:
            soc = min(cap_kwh, soc + min(net, max_kw) * eff)
            g = 0.0
        else:
            draw = min(-net, max_kw, (soc - reserve) * eff)
            soc -= draw / eff
            g = -net - draw
        grid.append(g)
        socs.append(soc)
    out["grid"] = grid
    out["soc"] = socs
    return out
