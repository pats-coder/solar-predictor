"""Open-Meteo weather: archived past forecasts (training) and live forecasts (prediction).

Radiation values from Open-Meteo are means over the PRECEDING hour, so a value
stamped T describes T-1h..T. Everything returned here is re-indexed to the hour
START (UTC) so it lines up with Home Assistant hourly statistics.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger("solar_predictor.weather")

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FC_URL = "https://api.open-meteo.com/v1/forecast"
BASE_VARS = [
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
]


def om_azimuth(compass_deg: float) -> float:
    """Compass bearing the panels face (0=N, 90=E) -> Open-Meteo convention (0=S, -90=E, 90=W)."""
    a = (compass_deg - 180.0) % 360.0
    return a - 360.0 if a > 180.0 else a


def _get(url: str, params: dict, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # noqa: PERF203
            if i == tries - 1:
                raise
            log.warning("Open-Meteo request failed (%s), retrying", e)
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def _frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    idx = pd.to_datetime(h.pop("time"), utc=True) - pd.Timedelta(hours=1)
    return pd.DataFrame(h, index=idx)


def _fetch(url: str, lat: float, lon: float, arrays: list[dict], extra: dict) -> pd.DataFrame:
    common = {"latitude": lat, "longitude": lon, "timezone": "UTC", **extra}
    df = _frame(_get(url, {**common, "hourly": ",".join(BASE_VARS)}))
    for a in arrays:
        js = _get(url, {**common, "hourly": "global_tilted_irradiance",
                        "tilt": a["tilt"], "azimuth": om_azimuth(a["azimuth"])})
        df[f"gti_{a['name']}"] = _frame(js)["global_tilted_irradiance"]
    return df


def history(lat, lon, arrays, start: date, end: date, cache_path: str | None = None) -> pd.DataFrame:
    """Archived forecasts between start and end (inclusive), cached on disk."""
    cached = None
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        want = {f"gti_{a['name']}" for a in arrays}
        if not want.issubset(cached.columns):
            cached = None
        elif len(cached):
            # refetch the last 3 days in case they were incomplete
            start = max(start, (cached.index.max() - pd.Timedelta(days=3)).date())
    frames = []
    s = start
    while s <= end:
        e = min(s + timedelta(days=89), end)
        log.info("Fetching archived weather %s..%s", s, e)
        frames.append(_fetch(HIST_URL, lat, lon, arrays, {"start_date": s.isoformat(), "end_date": e.isoformat()}))
        s = e + timedelta(days=1)
    df = pd.concat(([cached] if cached is not None else []) + frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if cache_path:
        df.to_pickle(cache_path)
    return df


def forecast(lat, lon, arrays, days: int = 3) -> pd.DataFrame:
    return _fetch(FC_URL, lat, lon, arrays, {"forecast_days": days, "past_days": 1})
