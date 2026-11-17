"""Feature engineering for the solar, consumption and grid models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sun_position(idx_utc: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Solar elevation/azimuth (degrees) at the middle of each hour (NOAA approximation)."""
    t = idx_utc + pd.Timedelta(minutes=30)
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    zen = np.arccos(cosz)
    elev = 90 - np.degrees(zen)
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180
    return pd.DataFrame({"sun_elev": elev, "sun_az": az % 360}, index=idx_utc)


def clear_sky_ghi(elev_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI (W/m2)."""
    cz = np.sin(np.radians(np.clip(elev_deg, 0, 90)))
    with np.errstate(divide="ignore", over="ignore"):
        g = 1098.0 * cz * np.exp(-0.057 / np.where(cz > 0, cz, np.inf))
    return np.where(cz > 0.01, g, 0.0)


def _calendar(idx_utc: pd.DatetimeIndex, tz: str, holidays_set) -> pd.DataFrame:
    loc = idx_utc.tz_convert(tz)
    doy = loc.dayofyear.values
    hol = np.array([d in holidays_set for d in loc.date])
    return pd.DataFrame({
        "hour": loc.hour.values,
        "dow": loc.dayofweek.values,
        "weekend": (loc.dayofweek.values >= 5) | hol,
        "holiday": hol,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=idx_utc)


def solar_features(wx: pd.DataFrame, lat: float, lon: float, tz: str, arrays: list[dict]) -> pd.DataFrame:
    sp = sun_position(wx.index, lat, lon)
    cs = clear_sky_ghi(sp["sun_elev"].values)
    gti_cols = [f"gti_{a['name']}" for a in arrays]
    f = pd.DataFrame(index=wx.index)
    for c in gti_cols + ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
                         "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                         "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        f[c] = wx[c]
    f["gti_sum"] = wx[gti_cols].sum(axis=1)
    f["sun_elev"] = sp["sun_elev"]
    f["sun_az"] = sp["sun_az"]
    f["clear_sky"] = cs
    f["kt"] = np.where(cs > 20, wx["shortwave_radiation"] / np.maximum(cs, 1), np.nan)
    loc = wx.index.tz_convert(tz)
    f["hour"] = loc.hour.values
    doy = loc.dayofyear.values
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def consumption_features(wx: pd.DataFrame, cons_hist: pd.Series, lat: float, lon: float, tz: str,
                         holidays_set, solar_pred: pd.Series | None = None) -> pd.DataFrame:
    """cons_hist: hourly consumption actuals (may extend past wx). Lag features only
    use days D-8..D-2 so tomorrow's forecast never needs data that isn't known yet."""
    f = _calendar(wx.index, tz, holidays_set)
    f["temp"] = wx["temperature_2m"]
    f["rh"] = wx["relative_humidity_2m"]
    f["ghi"] = wx["shortwave_radiation"]
    loc = wx.index.tz_convert(tz)
    day = pd.DatetimeIndex(loc.normalize())
    # daily max/min temp for the local day (drives aircon / heating)
    tday = f["temp"].groupby(day.values).agg(["max", "min"])
    f["tmax_day"] = tday["max"].reindex(day.values).values
    f["tmin_day"] = tday["min"].reindex(day.values).values
    f["solar_pred"] = solar_pred.reindex(wx.index).values if solar_pred is not None else np.nan
    # lag features from history
    ch = cons_hist.copy()
    ch.index = ch.index.tz_convert(tz)
    tbl = ch.groupby([pd.DatetimeIndex(ch.index.normalize()), ch.index.hour]).sum().unstack()
    lag_same_hour = np.full(len(f), np.nan)
    lag_day = np.full(len(f), np.nan)
    daily = tbl.sum(axis=1, min_count=20)
    cache: dict = {}
    for i, (d, h) in enumerate(zip(day, f["hour"].values)):
        if d not in cache:
            lo, hi = d - pd.Timedelta(days=8), d - pd.Timedelta(days=2)
            win = tbl.loc[(tbl.index >= lo) & (tbl.index <= hi)]
            cache[d] = (win.mean(), daily.loc[win.index].mean()) if len(win) >= 3 else None
        c = cache[d]
        if c is not None:
            lag_same_hour[i] = c[0].get(h, np.nan)
            lag_day[i] = c[1]
    f["lag_same_hour_7d"] = lag_same_hour
    f["lag_daily_7d"] = lag_day
    return f
