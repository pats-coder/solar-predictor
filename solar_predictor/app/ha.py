"""Home Assistant access through the Supervisor proxy (REST + websocket)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import websocket

log = logging.getLogger("solar_predictor.ha")

TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
REST = os.environ.get("HA_REST_URL", "http://supervisor/core/api")
WS = os.environ.get("HA_WS_URL", "ws://supervisor/core/websocket")


class HA:
    def __init__(self, token: str = TOKEN, rest: str = REST, ws: str = WS):
        self.token, self.rest, self.ws_url = token, rest, ws
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

    # ---------- REST ----------
    def config(self) -> dict:
        r = self.s.get(f"{self.rest}/config", timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self, entity_id: str) -> dict | None:
        r = self.s.get(f"{self.rest}/states/{entity_id}", timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def set_state(self, entity_id: str, state, attributes: dict) -> None:
        r = self.s.post(f"{self.rest}/states/{entity_id}",
                        data=json.dumps({"state": state, "attributes": attributes}, default=str), timeout=30)
        r.raise_for_status()

    # ---------- websocket statistics ----------
    def hourly_change(self, stat_ids: list[str], start: datetime, end: datetime | None = None) -> pd.DataFrame:
        """Hourly energy change (kWh) per statistic, indexed by UTC hour start."""
        return self._stats(stat_ids, start, end, "change", {"energy": "kWh"})

    def hourly_mean(self, stat_ids: list[str], start: datetime, end: datetime | None = None) -> pd.DataFrame:
        """Hourly mean per statistic (power normalised to kW, so kW mean == kWh in that hour)."""
        return self._stats(stat_ids, start, end, "mean", {"power": "kW"})

    def _stats(self, stat_ids, start, end, kind: str, units: dict) -> pd.DataFrame:
        end = end or datetime.now(timezone.utc)
        frames = []
        chunk = timedelta(days=60)
        t = start
        ws = websocket.create_connection(self.ws_url, timeout=120)
        try:
            ws.recv()
            ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth = json.loads(ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"HA websocket auth failed: {auth}")
            msg_id = 1
            while t < end:
                t2 = min(t + chunk, end)
                ws.send(json.dumps({
                    "id": msg_id, "type": "recorder/statistics_during_period",
                    "start_time": t.isoformat(), "end_time": t2.isoformat(),
                    "statistic_ids": stat_ids, "period": "hour", "types": [kind],
                    "units": units,
                }))
                while True:
                    resp = json.loads(ws.recv())
                    if resp.get("id") == msg_id:
                        break
                if not resp.get("success"):
                    raise RuntimeError(f"statistics query failed: {resp}")
                for sid, rows in resp["result"].items():
                    if rows:
                        df = pd.DataFrame(rows)
                        if kind not in df:
                            continue
                        df = df[["start", kind]].rename(columns={kind: "value"})
                        df["sid"] = sid
                        frames.append(df)
                msg_id += 1
                t = t2
        finally:
            ws.close()
        if not frames:
            return pd.DataFrame(columns=stat_ids)
        df = pd.concat(frames)
        start_col = df["start"]
        if pd.api.types.is_numeric_dtype(start_col):
            df["start"] = pd.to_datetime(start_col, unit="ms", utc=True)
        else:
            df["start"] = pd.to_datetime(start_col, utc=True)
        out = df.pivot_table(index="start", columns="sid", values="value", aggfunc="sum")
        return out.reindex(columns=stat_ids)
