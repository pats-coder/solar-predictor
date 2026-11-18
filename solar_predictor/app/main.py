"""Solar Predictor add-on entry point."""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
import traceback
from datetime import date, datetime, timedelta, timezone

import holidays
import numpy as np
import pandas as pd

import weather
from ha import HA
from models import Predictor, accuracy, clean, simulate_battery

DATA = os.environ.get("DATA_DIR", "/data")
MODEL_PATH = os.path.join(DATA, "model.pkl")
WX_CACHE = os.path.join(DATA, "weather_hist.pkl")
TRACK_PATH = os.path.join(DATA, "tracking.json")
PREFIX = "sensor.solar_predictor_"
VERSION = "0.2.1"

log = logging.getLogger("solar_predictor")


def load_options() -> dict:
    with open(os.path.join(DATA, "options.json")) as f:
        return json.load(f)


def hhmm(s: str) -> float:
    h, m = s.split(":")
    return int(h) + int(m) / 60


class App:
    def __init__(self, opt: dict, ha: HA | None = None):
        self.opt = opt
        self.ha = ha or HA()
        cfg = self.ha.config()
        self.lat, self.lon, self.tz = cfg["latitude"], cfg["longitude"], cfg["time_zone"]
        yrs = range(date.today().year - 2, date.today().year + 3)
        self.hol = holidays.country_holidays("AU", subdiv=opt.get("holiday_subdiv") or None, years=yrs)
        self.stats = [opt["production_stat"], opt["consumption_stat"], opt["grid_import_stat"]]
        self.model: Predictor | None = None
        self.trained_on: date | None = None
        self.last_error: str | None = None
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                if (saved.get("arrays") == opt["arrays"] and saved.get("version") == VERSION
                        and saved.get("ev") == (opt.get("ev_power_stat"), opt.get("ev_soc_stat"))):
                    self.model, self.trained_on = saved["model"], saved["trained_on"]
            except Exception:  # noqa: BLE001
                log.warning("Saved model unreadable; will retrain")

    # ---------- helpers ----------
    def now_local(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.tz)

    def actuals(self, start: datetime) -> pd.DataFrame:
        df = self.ha.hourly_change(self.stats, start)
        df.columns = ["solar", "cons", "grid"]
        extra = {k: self.opt.get(o) for k, o in [("ev", "ev_power_stat"), ("soc", "ev_soc_stat")] if self.opt.get(o)}
        if extra:
            m = self.ha.hourly_mean(list(extra.values()), start)
            m.columns = list(extra.keys())
            df = df.join(m, how="left")
        return df

    def in_free(self, idx: pd.DatetimeIndex) -> np.ndarray:
        loc = idx.tz_convert(self.tz)
        h = loc.hour.values + loc.minute.values / 60
        return (h >= hhmm(self.opt["free_start"])) & (h < hhmm(self.opt["free_end"]))

    def put(self, key: str, state, attrs: dict) -> None:
        base = {"attribution": "Solar Predictor (Open-Meteo + HA history)"}
        try:
            self.ha.set_state(PREFIX + key, state, {**base, **attrs})
        except Exception as e:  # noqa: BLE001
            log.warning("Could not publish %s: %s", key, e)

    # ---------- training ----------
    def train(self) -> None:
        start = datetime.fromisoformat(self.opt["history_start"]).replace(tzinfo=timezone.utc)
        log.info("Training: loading HA statistics since %s", start.date())
        act = self.actuals(start)
        first = act.dropna(how="all").index.min()
        wx = weather.history(self.lat, self.lon, self.opt["arrays"], first.date(),
                             datetime.now(timezone.utc).date(), WX_CACHE)
        m = Predictor(self.lat, self.lon, self.tz, self.opt["arrays"], self.hol,
                      free_start=hhmm(self.opt["free_start"]), free_end=hhmm(self.opt["free_end"]))
        m.train(wx, act)
        self.model, self.trained_on = m, self.now_local().date()
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": m, "trained_on": self.trained_on, "arrays": self.opt["arrays"],
                         "version": VERSION, "ev": (self.opt.get("ev_power_stat"), self.opt.get("ev_soc_stat"))}, f)
        log.info("Model trained on %s daylight hours", m.trained_rows)

    # ---------- tracking of as-issued forecasts ----------
    def _track_load(self) -> dict:
        if os.path.exists(TRACK_PATH):
            with open(TRACK_PATH) as f:
                return json.load(f)
        return {}

    def _bench(self, key: str) -> float | None:
        eid = self.opt.get(key)
        if not eid:
            return None
        st = self.ha.state(eid)
        try:
            return float(st["state"]) if st else None
        except (TypeError, ValueError):
            return None

    def track(self, now: pd.Timestamp, today_tot: dict, tomorrow_tot: dict, daily_act: pd.DataFrame) -> dict:
        tr = self._track_load()
        td, tm = str(now.date()), str((now + pd.Timedelta(days=1)).date())
        if now.hour >= 21 and "dayahead" not in tr.get(tm, {}):
            tr.setdefault(tm, {})["dayahead"] = {**tomorrow_tot, "bench_solar": self._bench("benchmark_tomorrow")}
        if 5 <= now.hour < 9 and "morning" not in tr.get(td, {}):
            tr.setdefault(td, {})["morning"] = {**today_tot, "bench_solar": self._bench("benchmark_today")}
        for d, rec in tr.items():
            if d < td and "actual" not in rec:
                k = pd.Timestamp(d, tz=self.tz)
                if k in daily_act.index and daily_act.loc[k].notna().all():
                    rec["actual"] = {c: round(float(daily_act.at[k, c]), 2) for c in ["solar", "cons", "grid", "grid_paid"]}
        cutoff = str((now - pd.Timedelta(days=60)).date())
        tr = {d: r for d, r in tr.items() if d >= cutoff}
        with open(TRACK_PATH, "w") as f:
            json.dump(tr, f, indent=1)
        return tr

    def live_accuracy(self, tr: dict, now: pd.Timestamp) -> dict:
        start = str((now - pd.Timedelta(days=30)).date())
        res = {}
        for when in ["dayahead", "morning"]:
            rows = [r for d, r in tr.items() if d >= start and when in r and "actual" in r]
            if not rows:
                continue
            a = pd.DataFrame([r["actual"] for r in rows])
            p = pd.DataFrame([r[when] for r in rows])
            res[when] = {"days": len(rows),
                         "solar": accuracy(p["solar"], a["solar"]),
                         "cons": accuracy(p["cons"], a["cons"]),
                         "grid": accuracy(p["grid_paid"], a["grid_paid"]) if "grid_paid" in a else None,
                         "bench_solar": accuracy(p["bench_solar"].astype(float), a["solar"])
                         if p["bench_solar"].notna().any() else None}
        return res

    def _num(self, key: str) -> float | None:
        eid = self.opt.get(key)
        st = self.ha.state(eid) if eid else None
        try:
            return float(st["state"]) if st else None
        except (TypeError, ValueError):
            return None

    def _st(self, key: str) -> str | None:
        eid = self.opt.get(key)
        st = self.ha.state(eid) if eid else None
        return st["state"] if st else None

    def live_ev_plan(self, now: pd.Timestamp, idx: pd.DatetimeIndex) -> np.ndarray | None:
        """Expected EV kWh per remaining hour today, from plug/home/charge-level state.
        Returns None when the car's live state isn't configured or readable (model forecast stands)."""
        if not (self.opt.get("ev_plug_entity") and self.opt.get("ev_soc_entity")):
            return None
        soc, plug = self._num("ev_soc_entity"), self._st("ev_plug_entity")
        if soc is None or plug in (None, "unknown", "unavailable"):
            return None
        home = self._st("ev_home_entity") if self.opt.get("ev_home_entity") else "home"
        limit = max(self._num("ev_limit_entity") or 80.0, 80.0)
        cap = float(self.opt.get("ev_battery_kwh") or 0) or 65.0
        kw = float(self.opt.get("ev_charge_kw") or 11.0)
        plan = np.zeros(len(idx))
        if plug != "on" or home != "home" or soc >= 95:
            return plan
        need = max(0.0, (limit - soc) / 100 * cap) / 0.9  # wall-side energy incl. charging losses
        free = self.in_free(idx)
        for i in np.where(free)[0]:
            if need <= 0:
                break
            mins = 60 - (now.minute if idx[i] == now.tz_convert("UTC").floor("h") else 0)
            e = min(need, kw * mins / 60)
            plan[i] = e
            need -= e
        return plan

    # ---------- forecast cycle ----------
    def update(self) -> None:
        m = self.model
        now = self.now_local()
        today0 = now.normalize()
        wx = weather.forecast(self.lat, self.lon, self.opt["arrays"], days=3)
        act = clean(self.actuals(datetime.now(timezone.utc) - timedelta(days=12)))
        live_soc = None
        if self.opt.get("ev_soc_entity"):
            st = self.ha.state(self.opt["ev_soc_entity"])
            try:
                live_soc = float(st["state"]) if st else None
            except (TypeError, ValueError):
                live_soc = None
        pred = m.predict(wx, act, live_soc)

        # same-day correction of solar from today's actuals so far
        loc_idx = pred.index.tz_convert(self.tz)
        is_today = (loc_idx >= today0) & (loc_idx < today0 + pd.Timedelta(days=1))
        done = act.index[(act.index.tz_convert(self.tz) >= today0) & act["solar"].notna()]
        done = done.intersection(pred.index)
        corr = 1.0
        p_done = pred.loc[done, "solar"].sum()
        p_day = pred.loc[is_today, "solar"].sum()
        if p_done > 0.5 and p_day > 0:
            ratio = float(np.clip(act.loc[done, "solar"].sum() / p_done, 0.5, 1.5))
            w = min(1.0, p_done / (0.3 * p_day))
            corr = 1 + w * (ratio - 1)
        future_today = is_today & ~pred.index.isin(done)
        pred.loc[future_today, "solar"] *= corr

        # same-day correction of base load (EV charging excluded)
        a_base = act["cons"] - (act["ev"].fillna(0) if "ev" in act else 0)
        b_done = pred.loc[done, "base"].sum()
        bcorr = 1.0
        if b_done > 3 and pred.loc[is_today, "base"].sum() > 0:
            ratio = float(np.clip(a_base.loc[done].sum() / b_done, 0.7, 1.3))
            w = min(1.0, b_done / (0.5 * pred.loc[is_today, "base"].sum()))
            bcorr = 1 + w * (ratio - 1)
        pred.loc[future_today, "base"] *= bcorr

        # today's EV charging from the car's live state (mirrors the free-window charging automation)
        ev_plan = self.live_ev_plan(now, pred.index[future_today])
        if ev_plan is not None:
            pred.loc[future_today, "ev"] = ev_plan
        pred["cons"] = pred["base"] + pred["ev"]
        pred["grid"] = m.predict_grid(pred["cons"].values, pred["solar"].values, pred["ev"].values, pred.index)

        # battery (optional)
        if self.opt.get("battery_kwh", 0) > 0:
            fut = pred.index >= pd.Timestamp.now(tz="UTC").floor("h")
            sim = simulate_battery(pred.loc[fut], self.opt["battery_kwh"], self.opt["battery_reserve_pct"],
                                   self.opt["battery_max_kw"], self.opt["battery_kwh"] * 0.5)
            pred.loc[fut, "grid"] = sim["grid"].values

        # merge actuals for completed hours today
        merged = pred.copy()
        merged.loc[done, ["solar", "cons", "grid"]] = act.loc[done, ["solar", "cons", "grid"]].values
        if "ev" in act:
            merged.loc[done, "ev"] = act.loc[done, "ev"].fillna(0).values
        merged.loc[done, "base"] = (merged.loc[done, "cons"] - merged.loc[done, "ev"]).clip(lower=0)
        merged["free"] = self.in_free(merged.index)
        merged["grid_paid"] = np.where(merged["free"], 0.0, merged["grid"])
        merged["actual"] = merged.index.isin(done)

        def totals(day0):
            sel = (loc_idx >= day0) & (loc_idx < day0 + pd.Timedelta(days=1))
            s = merged.loc[sel]
            return {"solar": round(s["solar"].sum(), 2), "cons": round(s["cons"].sum(), 2),
                    "ev": round(s["ev"].sum(), 2), "base": round(s["base"].sum(), 2),
                    "grid": round(s["grid"].sum(), 2), "grid_paid": round(s["grid_paid"].sum(), 2)}

        t_today, t_tom = totals(today0), totals(today0 + pd.Timedelta(days=1))
        hour_now = pd.Timestamp.now(tz="UTC").floor("h")
        remaining = merged.loc[is_today & (merged.index >= hour_now) & ~merged["actual"], "solar"].sum()

        def hourly(day0, days=2):
            sel = (loc_idx >= day0) & (loc_idx < day0 + pd.Timedelta(days=days))
            return [{"start": t.tz_convert(self.tz).isoformat(), "solar": round(r.solar, 3),
                     "consumption": round(r.cons, 3), "ev_charging": round(r.ev, 3), "grid": round(r.grid, 3),
                     "free_window": bool(r.free), "actual": bool(r.actual)}
                    for t, r in merged.loc[sel].iterrows()]

        fc = hourly(today0)
        nxt = merged[(merged.index >= hour_now) & ~merged["free"] & (merged["grid"] > 0.2)]
        next_draw = nxt.index[0].tz_convert(self.tz).isoformat() if len(nxt) else None
        next_hour = merged.loc[merged.index == hour_now + pd.Timedelta(hours=1), "solar"]

        # daily actuals for tracking
        ad = act.copy()
        ad["grid_paid"] = np.where(self.in_free(ad.index), 0.0, ad["grid"])
        dl = ad.index.tz_convert(self.tz).normalize()
        daily_act = ad.groupby(dl).sum(min_count=20)
        tr = self.track(now, t_today, t_tom, daily_act)
        live = self.live_accuracy(tr, now)

        kwh = {"unit_of_measurement": "kWh", "device_class": "energy", "state_class": "total"}
        self.put("solar_today", t_today["solar"], {**kwh, "friendly_name": "Solar Predictor Solar Today",
                 "icon": "mdi:solar-power-variant", "same_day_correction": round(corr, 3), "forecast": fc})
        self.put("solar_remaining_today", round(remaining, 2), {**kwh, "friendly_name": "Solar Predictor Solar Remaining Today",
                 "icon": "mdi:solar-power-variant-outline"})
        self.put("solar_tomorrow", t_tom["solar"], {**kwh, "friendly_name": "Solar Predictor Solar Tomorrow",
                 "icon": "mdi:solar-power-variant"})
        self.put("solar_next_hour", round(float(next_hour.iloc[0]) if len(next_hour) else 0.0, 3),
                 {**kwh, "friendly_name": "Solar Predictor Solar Next Hour", "icon": "mdi:solar-power"})
        self.put("consumption_today", t_today["cons"], {**kwh, "friendly_name": "Solar Predictor Consumption Today",
                 "icon": "mdi:home-lightning-bolt", "base_load": t_today["base"], "ev_charging": t_today["ev"],
                 "same_day_correction": round(bcorr, 3)})
        self.put("consumption_tomorrow", t_tom["cons"], {**kwh, "friendly_name": "Solar Predictor Consumption Tomorrow",
                 "icon": "mdi:home-lightning-bolt", "base_load": t_tom["base"], "ev_charging": t_tom["ev"]})
        self.put("ev_charging_today", t_today["ev"], {**kwh, "friendly_name": "Solar Predictor EV Charging Today",
                 "icon": "mdi:ev-station"})
        self.put("ev_charging_tomorrow", t_tom["ev"], {**kwh, "friendly_name": "Solar Predictor EV Charging Tomorrow",
                 "icon": "mdi:ev-station"})
        self.put("grid_import_today", t_today["grid_paid"], {**kwh, "friendly_name": "Solar Predictor Paid Grid Import Today",
                 "icon": "mdi:transmission-tower-import", "including_free_window": t_today["grid"]})
        self.put("grid_import_tomorrow", t_tom["grid_paid"], {**kwh, "friendly_name": "Solar Predictor Paid Grid Import Tomorrow",
                 "icon": "mdi:transmission-tower-import", "including_free_window": t_tom["grid"]})
        self.put("next_grid_draw", next_draw or "unknown", {"device_class": "timestamp",
                 "friendly_name": "Solar Predictor Next Paid Grid Draw", "icon": "mdi:clock-alert-outline"})
        self.last_error = None  # reaching here means this cycle succeeded
        bt = m.backtest or {}
        pct = {"unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:bullseye-arrow"}
        self.put("accuracy_solar", bt.get("solar", "unknown"), {**pct, "friendly_name": "Solar Predictor Backtest Accuracy Solar",
                 "basis": f"daily totals, last {bt.get('days', 0)} days held out", "hourly_accuracy": bt.get("solar_hourly"),
                 "daily": bt.get("daily", [])})
        self.put("accuracy_consumption", bt.get("cons", "unknown"), {**pct, "friendly_name": "Solar Predictor Backtest Accuracy Consumption",
                 "base_load": bt.get("base"), "ev_charging": bt.get("ev")})
        self.put("accuracy_grid", bt.get("grid_paid", "unknown"), {**pct, "friendly_name": "Solar Predictor Backtest Accuracy Paid Grid Import",
                 "all_grid_import": bt.get("grid")})
        da = live.get("dayahead", {})
        self.put("live_accuracy_solar", da.get("solar", "unknown"), {**pct, "friendly_name": "Solar Predictor Live Accuracy Solar (day-ahead, 30d)",
                 "days": da.get("days", 0), "consumption": da.get("cons"), "paid_grid": da.get("grid"),
                 "same_day": live.get("morning")})
        self.put("live_accuracy_benchmark", da.get("bench_solar", "unknown"), {**pct,
                 "friendly_name": "Solar Predictor Benchmark Accuracy (Solcast, day-ahead, 30d)", "days": da.get("days", 0)})
        self.put("status", "ok" if not self.last_error else "error", {"friendly_name": "Solar Predictor Status",
                 "icon": "mdi:check-network-outline", "version": VERSION, "trained_on": str(self.trained_on),
                 "trained_hours": m.trained_rows, "last_update": now.isoformat(), "last_error": self.last_error})
        log.info("Updated: solar today %.1f (corr %.2f), tomorrow %.1f | use today %.1f (EV %.1f), tomorrow %.1f (EV %.1f)"
                 " | paid grid %.1f / %.1f", t_today["solar"], corr, t_tom["solar"], t_today["cons"], t_today["ev"],
                 t_tom["cons"], t_tom["ev"],
                 t_today["grid_paid"], t_tom["grid_paid"])

    def run(self) -> None:
        period = 60 * int(self.opt.get("update_minutes", 15))
        while True:
            now = self.now_local()
            try:
                if self.model is None or (self.trained_on != now.date() and now.hour >= 2):
                    self.train()
                self.update()
                self.last_error = None
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                log.error("Cycle failed: %s\n%s", e, traceback.format_exc())
                self.put("status", "error", {"friendly_name": "Solar Predictor Status", "last_error": self.last_error,
                                             "version": VERSION})
            time.sleep(period)


def main() -> None:
    opt = load_options()
    logging.basicConfig(level=getattr(logging, opt.get("log_level", "info").upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("Solar Predictor %s starting", VERSION)
    App(opt).run()


if __name__ == "__main__":
    main()
