"""Source AIS + capteurs embarqués (voiliers, cargos à assistance vélique).

Format d'entrée attendu (Parquet ou CSV) :
  mmsi, timestamp, lat, lon, u_obs, v_obs, sigma_u, sigma_v
Le vent vrai est supposé déjà calculé à partir du vent apparent et du COG/SOG
fournis par l'AIS (fait en amont — pas dans ce module).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Observation, ObservationSource
from .operators import make_bilinear_H


class AISWindSource(ObservationSource):
    name = "ais"

    def __init__(
        self,
        path: str,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        u10_channel: int,
        v10_channel: int,
        default_sigma_ms: float = 1.5,
    ):
        self.path = Path(path)
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.u10_channel = u10_channel
        self.v10_channel = v10_channel
        self.default_sigma_ms = default_sigma_ms

    def _read(self) -> pd.DataFrame:
        if self.path.suffix == ".parquet":
            return pd.read_parquet(self.path)
        return pd.read_csv(self.path, parse_dates=["timestamp"])

    def load(self, t0: str, t1: str) -> list[Observation]:
        df = self._read()
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
        if df.empty:
            return []
        coords = df[["lat", "lon", "timestamp"]].to_numpy()
        # Convertit le timestamp en heures depuis t0 — utile aux solveurs DA.
        t0_dt = np.datetime64(t0)
        coords[:, 2] = (coords[:, 2].astype("datetime64[ns]") - t0_dt) / np.timedelta64(1, "h")
        coords = coords.astype(np.float64)

        sig_u = df.get("sigma_u", pd.Series([self.default_sigma_ms] * len(df))).to_numpy()
        sig_v = df.get("sigma_v", pd.Series([self.default_sigma_ms] * len(df))).to_numpy()

        return [
            Observation(
                y=df["u_obs"].to_numpy(),
                coords=coords,
                sigma_o=sig_u,
                H=make_bilinear_H(self.u10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_u_component_of_wind",
                source=self.name,
            ),
            Observation(
                y=df["v_obs"].to_numpy(),
                coords=coords,
                sigma_o=sig_v,
                H=make_bilinear_H(self.v10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_v_component_of_wind",
                source=self.name,
            ),
        ]
