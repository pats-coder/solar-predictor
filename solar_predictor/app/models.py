"""Model training, backtesting and prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import consumption_features, solar_features

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
    solar_cap: float = 0.0
    m_solar: object = None
    m_cons: object = None
    m_grid: object = None
    solar_cols: list = field(default_factory=list)
    cons_cols: list = field(default_factory=list)
    grid_cols: list = field(default_factory=lambda: ["cons", "solar", "deficit", "surplus", "hour"])
    trained_rows: int = 0
    backtest: dict = field(default_factory=dict)

    # ---------- features ----------
    def _sf(self, wx):
        return solar_features(wx, self.lat, self.lon, self.tz, self.arrays)

    def _cf(self, wx, cons_hist, solar_pred):
        return consumption_features(wx, cons_hist, self.lat, self.lon, self.tz, self.holidays_set, solar_pred)

    @staticmethod
    def _gf(cons, solar, idx, tz):
        loc = idx.tz_convert(tz)
        c, s = np.asarray(cons, float), np.asarray(solar, float)
        return pd.DataFrame({"cons": c, "solar": s, "deficit": np.maximum(c - s, 0),
                             "surplus": np.maximum(s - c, 0), "hour": loc.hour.values}, index=idx)

    # ---------- solar ----------
    def predict_solar(self, wx: pd.DataFrame) -> pd.Series:
        X = self._sf(wx)
        p = self.m_solar.predict(X[self.solar_cols])
        p = np.clip(p, 0, self.solar_cap or None)
        p[X["sun_elev"].values <= SUN_MIN] = 0.0
        return pd.Series(p, index=wx.index, name="solar")

    # ---------- training ----------
    def _fit(self, wx: pd.DataFrame, act: pd.DataFrame) -> None:
        idx = wx.index.intersection(act.index)
        wx, act = wx.loc[idx], act.loc[idx]

        Xs = self._sf(wx)
        self.solar_cols = [c for c in Xs.columns]
        day = (Xs["sun_elev"] > SUN_MIN) & act["solar"].notna() & Xs["shortwave_radiation"].notna()
        self.solar_cap = float(act["solar"].quantile(0.999) * 1.05)
        self.m_solar = _gbr().fit(Xs.loc[day, self.solar_cols], act.loc[day, "solar"])
        solar_hat = self.predict_solar(wx)

        Xc = self._cf(wx, act["cons"].dropna(), solar_hat)
        self.cons_cols = list(Xc.columns)
        ok = act["cons"].notna() & Xc["lag_daily_7d"].notna()
        self.m_cons = _gbr().fit(Xc.loc[ok, self.cons_cols], act.loc[ok, "cons"])

        g = act[["cons", "solar", "grid"]].dropna()
        Xg = self._gf(g["cons"], g["solar"], g.index, self.tz)
        self.m_grid = _gbr(max_iter=300).fit(Xg[self.grid_cols], g["grid"])
        self.trained_rows = int(day.sum())

    def predict(self, wx: pd.DataFrame, cons_hist: pd.Series) -> pd.DataFrame:
        solar = self.predict_solar(wx)
        Xc = self._cf(wx, cons_hist, solar)
        cons = np.clip(self.m_cons.predict(Xc[self.cons_cols]), 0, None)
        Xg = self._gf(cons, solar.values, wx.index, self.tz)
        grid = np.clip(self.m_grid.predict(Xg[self.grid_cols]), 0, None)
        grid = np.minimum(grid, cons)  # can't import more than the house uses (no battery)
        return pd.DataFrame({"solar": solar.values, "cons": cons, "grid": grid}, index=wx.index)

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
            # lag features only look at D-8..D-2, so passing full history mirrors live use
            pred = self.predict(wx.loc[test_idx], act["cons"].dropna())
            d = pred.index.tz_convert(self.tz).normalize()
            pdaily = pred.groupby(d).sum()
            adaily = act.loc[test_idx].groupby(d).sum(min_count=20)
            bt = {k: accuracy(pdaily[k], adaily[k]) for k in ["solar", "cons", "grid"]}
            bt["solar_hourly"] = accuracy(pred["solar"], act.loc[test_idx, "solar"])
            bt["days"] = int(adaily["solar"].notna().sum())
            bt["daily"] = [
                {"date": str(k.date()), "solar_pred": round(pdaily.at[k, "solar"], 1),
                 "solar_actual": None if pd.isna(adaily.at[k, "solar"]) else round(adaily.at[k, "solar"], 1),
                 "cons_pred": round(pdaily.at[k, "cons"], 1),
                 "cons_actual": None if pd.isna(adaily.at[k, "cons"]) else round(adaily.at[k, "cons"], 1)}
                for k in pdaily.index]
            log.info("Backtest (%s days): solar %s%%, consumption %s%%, grid %s%%",
                     bt["days"], bt["solar"], bt["cons"], bt["grid"])
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
