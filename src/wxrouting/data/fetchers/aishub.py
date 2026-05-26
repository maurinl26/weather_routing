"""Fetcher AISHub — flux AIS temps réel (reciprocity).

API : http://www.aishub.net/api/...?username=USER&format=1&output=json
Limite : 1 requête / minute, snapshot des cibles actuelles (pas d'historique).
La clé `AISHUB_USERNAME` se passe par variable d'environnement.

⚠️ AIS = positions navires. Pour le routage on en tire :
- des trajectoires (déduites par interpolation des positions successives),
- des obs de vent SI le navire embarque un capteur, ce qui n'est pas
  remonté par l'AIS standard (champ optionnel "weather" rarement utilisé).

Ce fetcher remonte donc des positions ; le vent doit venir d'un canal séparé
(Signal K, fichier SCADA, ou capteur perso) — voir `synthetic_ais.py` pour
un exemple de génération.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from .base import BBox, Fetcher, RawObsSchema

AISHUB_URL = "https://data.aishub.net/ws.php"


class AISHubFetcher(Fetcher):
    name = "aishub"

    def __init__(
        self,
        username: str | None = None,
        timeout_s: int = 30,
    ):
        self.username = username or os.environ.get("AISHUB_USERNAME")
        if not self.username:
            raise RuntimeError(
                "AISHub requires a username — set AISHUB_USERNAME env var "
                "or pass `username=`."
            )
        self.timeout_s = timeout_s

    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        params = {
            "username": self.username,
            "format": 1,
            "output": "json",
            "compress": 0,
            "latmin": bbox.lat_min, "latmax": bbox.lat_max,
            "lonmin": bbox.lon_min, "lonmax": bbox.lon_max,
        }
        resp = requests.get(AISHUB_URL, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        if not payload or payload[0].get("ERROR"):
            return RawObsSchema.empty()

        records = payload[1] if len(payload) > 1 else []
        if not records:
            return RawObsSchema.empty()

        df = pd.DataFrame(records)
        # Schéma AISHub : MMSI, LAT, LON, COG, SOG, TIME, … (timestamp Unix).
        df = df.rename(
            columns={
                "MMSI": "platform_id",
                "LAT": "lat",
                "LON": "lon",
                "TIME": "timestamp",
                "COG": "course_over_ground",
                "SOG": "speed_over_ground",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]

        # Pas de vent dans l'AIS standard → on émet des lignes "position" en
        # encodant COG/SOG comme variables canoniques séparées. L'adapter
        # ignorera (mapping vide) — ce fetcher reste *physically-agnostic*.
        out = []
        for var_in, var_out in [
            ("speed_over_ground", "platform_speed"),
            ("course_over_ground", "platform_course"),
        ]:
            sub = df[["timestamp", "lat", "lon", var_in, "platform_id"]].copy()
            sub["variable"] = var_out
            sub = sub.rename(columns={var_in: "value"}).dropna(subset=["value"])
            out.append(sub)
        return RawObsSchema.validate(pd.concat(out, ignore_index=True))
