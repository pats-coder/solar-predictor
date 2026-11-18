"""End-to-end test with synthetic HA history and weather (no network).

Run:  python tests/test_pipeline.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

APP = os.path.join(os.path.dirname(__file__), "..", "solar_predictor", "app")
sys.path.insert(0, os.path.abspath(APP))
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp

import features  # noqa: E402
import main  # noqa: E402
import weather  # noqa: E402

LAT, LON, TZ = -32.93, 151.75, "Australia/Sydney"
ARRAYS = [{"name": "north", "azimuth": 0, "tilt": 20}, {"name": "east", "azimuth": 90, "tilt": 20},
          {"name": "west", "azimuth": 270, "tilt": 20}]
rng = np.random.default_rng(1)
END = pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(days=3)
IDX = pd.date_range(pd.Timestamp("2026-02-01", tz="UTC"), END, freq="h")


def make_weather(idx):
    sp = features.sun_position(idx, LAT, LON)
    cs = features.clear_sky_ghi(sp["sun_elev"].values)
    days = idx.normalize()
    cloud_day = pd.Series(rng.uniform(0, 100, len(days.unique())), index=days.unique())
    cloud = np.clip(cloud_day.reindex(days).values + rng.normal(0, 10, len(idx)), 0, 100)
    ghi = cs * (1 - 0.75 * (cloud / 100) ** 2)
    az = sp["sun_az"].values
    wx = pd.DataFrame(index=idx)
    wx["shortwave_radiation"] = ghi
    wx["direct_normal_irradiance"] = ghi * 0.7
    wx["diffuse_radiation"] = ghi * 0.3
    for c in ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]:
        wx[c] = cloud
    loc_h = idx.tz_convert(TZ).hour.values
    wx["temperature_2m"] = 20 + 6 * np.sin((loc_h - 9) / 24 * 2 * np.pi) - 0.05 * cloud
    wx["relative_humidity_2m"] = 60.0
    wx["precipitation"] = (cloud > 85) * 1.0
    wx["wind_speed_10m"] = 10.0
    face = {"north": 0, "east": 90, "west": 270}
    for a in ARRAYS:
        diff = np.radians(az - face[a["name"]])
        wx[f"gti_{a['name']}"] = ghi * np.clip(0.75 + 0.45 * np.cos(diff), 0, None)
    return wx


WX = make_weather(IDX)
TRUE_KWP = {"north": 5.0, "east": 3.5, "west": 4.5}
solar = sum(WX[f"gti_{n}"] / 1000 * k * 0.85 for n, k in TRUE_KWP.items())
solar = (solar * rng.normal(1, 0.05, len(IDX))).clip(lower=0)
loc = IDX.tz_convert(TZ)
base = 1.2 + 0.8 * np.isin(loc.hour, [6, 7, 17, 18, 19, 20]) + 0.6 * (loc.dayofweek >= 5)
cons = base + 0.15 * np.maximum(WX["temperature_2m"] - 24, 0) + 0.25 * solar + rng.normal(0, 0.25, len(IDX))
cons = cons.clip(lower=0.3)
# EV: charges 7 kW in the free window when the evening SoC was below 60%, drains 12%/day
ev = np.zeros(len(IDX))
soc = np.zeros(len(IDX))
s_now, charge_today = 70.0, False
for i, t in enumerate(loc):
    if t.hour == 0:
        s_now -= rng.uniform(5, 20)
        charge_today = s_now < 60
    if charge_today and 11 <= t.hour < 14:
        ev[i] = 7.0
        s_now = min(100, s_now + 9)
    soc[i] = s_now
cons = cons + ev
grid = np.maximum(cons - solar, 0) * 1.08
ACT = pd.DataFrame({"P": solar, "C": cons, "G": grid, "EV": ev, "SOC": soc}, index=IDX)
now_h = pd.Timestamp.now(tz="UTC").floor("h")
ACT = ACT[ACT.index < now_h - pd.Timedelta(hours=1)]


class FakeHA:
    def __init__(self):
        self.published = {}

    def config(self):
        return {"latitude": LAT, "longitude": LON, "time_zone": TZ}

    def state(self, eid):
        return {"state": "20.0"}

    def set_state(self, eid, state, attrs):
        json.dumps(attrs, default=str)
        self.published[eid] = (state, attrs)

    def hourly_change(self, ids, start, end=None):
        out = ACT[ACT.index >= pd.Timestamp(start)][["P", "C", "G"]].copy()
        out.columns = ids
        return out

    def hourly_mean(self, ids, start, end=None):
        out = ACT[ACT.index >= pd.Timestamp(start)][["EV", "SOC"]].copy()
        out.columns = ids
        return out


weather.history = lambda lat, lon, arrays, s, e, cache=None: WX[WX.index.date <= e]
weather.forecast = lambda lat, lon, arrays, days=3: WX[(WX.index >= now_h - pd.Timedelta(days=1))
                                                        & (WX.index < now_h + pd.Timedelta(days=days))]

opt = {"production_stat": "sensor.p", "consumption_stat": "sensor.c", "grid_import_stat": "sensor.g",
       "benchmark_today": "sensor.bt", "benchmark_tomorrow": "sensor.bm", "arrays": ARRAYS,
       "ev_power_stat": "sensor.ev", "ev_soc_stat": "sensor.soc", "ev_soc_entity": "sensor.soc",
       "free_start": "11:00", "free_end": "14:00", "battery_kwh": 0, "battery_reserve_pct": 10,
       "battery_max_kw": 5, "history_start": "2026-02-01", "holiday_subdiv": "NSW", "update_minutes": 15}

fake = FakeHA()
app = main.App(opt, ha=fake)
app.train()
app.update()
bt = app.model.backtest
print("backtest:", {k: v for k, v in bt.items() if k != "daily"})
for k, (s, a) in sorted(fake.published.items()):
    print(f"{k:55s} {s}")
assert bt["solar"] > 85, bt
assert bt["cons"] > 80, bt
assert bt["ev"] > 40, {k: v for k, v in bt.items() if k != "daily"}
assert "sensor.solar_predictor_ev_charging_tomorrow" in fake.published
fc = fake.published["sensor.solar_predictor_solar_today"][1]["forecast"]
assert len(fc) == 48, len(fc)
assert weather.om_azimuth(0) in (180, -180) and weather.om_azimuth(90) == -90 and weather.om_azimuth(270) == 90
# reload saved model path
app2 = main.App(opt, ha=fake)
assert app2.model is not None
print("OK")
