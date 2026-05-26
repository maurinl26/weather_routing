"""Fetcher VOS — Voluntary Observing Ships (messages SHIP/BBXX via GTS).

C'est *l'obs d'opportunité officielle* de l'OMM : navires marchands qui
transmettent vent/pression/SST 3-4×/jour. Diffusé via le GTS, archivé par
l'ECMWF (catalogue MARS, `class=od stream=oper type=ob obstype=ship`) et
par NOAA/NCEI (IMMA — International Maritime Meteorological Archive).

Trois backends possibles, par ordre de simplicité :
1. **NOAA ICOADS** (`https://psl.noaa.gov/data/gridded/data.icoads.html`) —
   produit gridé, pas d'obs ponctuelles mais agrégat mensuel ouvert.
2. **NOAA IMMA delayed-mode** — fichiers texte, obs ponctuelles, ouvert.
3. **ECMWF MARS / Reading Data Server** — vraies obs temps quasi réel mais
   compte ECMWF requis.

On implémente IMMA (équilibre ouverture / granularité ponctuelle).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cf_names import AIR_PRESSURE_AT_SEA_LEVEL, WIND_FROM_DIRECTION, WIND_SPEED_10M
from .base import BBox, Fetcher, RawObsSchema

IMMA_ROOT = "https://www.ncei.noaa.gov/data/marine/imma1/r3.0.1/"


class VOSFetcher(Fetcher):
    name = "vos_gts"

    def __init__(
        self,
        cache_dir: str = "data/vos_imma",
        platform_types: tuple[str, ...] = ("SHIP",),
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.platform_types = platform_types

    def _download_month(self, year: int, month: int) -> Path:
        target = self.cache_dir / f"IMMA1_R3.0.1_{year}-{month:02d}.imma"
        if target.exists():
            return target
        import requests

        url = f"{IMMA_ROOT}{year:04d}/IMMA1_R3.0.1_{year:04d}-{month:02d}"
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return target

    def _parse_imma(self, path: Path) -> pd.DataFrame:
        """Parse IMMA1 fixed-width.

        Champs IMMA1 (extrait minimal) — voir doc NOAA pour la spec complète :
            YR (1-4), MO (5-6), DY (7-8), HR (9-12), LAT (13-17), LON (18-23),
            IM/ATTI/TI/LI/DS/VS (24-..), W (wind speed, kn), D (wind dir, deg),
            SLP (sea level pressure, hPa).
        """
        rows: list[dict] = []
        with open(path) as f:
            for line in f:
                if len(line) < 100:
                    continue
                try:
                    yr = int(line[0:4])
                    mo = int(line[4:6])
                    dy = int(line[6:8])
                    hr_str = line[8:12].strip() or "0"
                    hr = float(hr_str) / 100.0 if hr_str else 0.0
                    lat = float(line[12:17]) / 100.0 - 90.0   # offset IMMA
                    lon = float(line[17:23]) / 100.0
                    if lon > 180:
                        lon -= 360
                    ws = _imma_float(line[55:58])
                    wd = _imma_float(line[58:61])
                    slp = _imma_float(line[68:73])
                except Exception:
                    continue
                ts = pd.Timestamp(year=yr, month=mo, day=dy, tz="UTC") + pd.Timedelta(hours=hr)
                rows.append(
                    {"timestamp": ts, "lat": lat, "lon": lon, "ws": ws, "wd": wd, "slp": slp}
                )
        if not rows:
            return RawObsSchema.empty()
        df = pd.DataFrame(rows)
        out: list[pd.DataFrame] = []
        if "ws" in df:
            sub = df.dropna(subset=["ws"]).copy()
            sub["variable"] = WIND_SPEED_10M
            sub["value"] = sub["ws"] * 0.5144  # kn → m/s
            out.append(sub[["timestamp", "lat", "lon", "variable", "value"]])
        if "wd" in df:
            sub = df.dropna(subset=["wd"]).copy()
            sub["variable"] = WIND_FROM_DIRECTION
            sub["value"] = sub["wd"]
            out.append(sub[["timestamp", "lat", "lon", "variable", "value"]])
        if "slp" in df:
            sub = df.dropna(subset=["slp"]).copy()
            sub["variable"] = AIR_PRESSURE_AT_SEA_LEVEL
            sub["value"] = sub["slp"] * 100.0  # hPa → Pa
            out.append(sub[["timestamp", "lat", "lon", "variable", "value"]])
        return pd.concat(out, ignore_index=True)

    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        start = pd.Timestamp(t0)
        end = pd.Timestamp(t1)
        cur = start.replace(day=1)
        parts: list[pd.DataFrame] = []
        while cur < end:
            try:
                path = self._download_month(cur.year, cur.month)
                df = self._parse_imma(path)
                df = df[
                    (df["lat"] >= bbox.lat_min) & (df["lat"] <= bbox.lat_max)
                    & (df["lon"] >= bbox.lon_min) & (df["lon"] <= bbox.lon_max)
                ]
                parts.append(df)
            except Exception:
                pass
            cur = (cur + pd.offsets.MonthBegin(1)).normalize()
        if not parts:
            return RawObsSchema.empty()
        df = pd.concat(parts, ignore_index=True)
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
        return RawObsSchema.validate(df)


def _imma_float(s: str) -> float | None:
    s = s.strip()
    if not s or set(s) <= {"9"}:  # missing indicator IMMA
        return None
    try:
        return float(s)
    except ValueError:
        return None
