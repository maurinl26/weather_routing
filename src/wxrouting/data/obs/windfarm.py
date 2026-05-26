"""Source champs éoliens offshore — vent à hub height (~100 m).

Format SCADA agrégé (Parquet) :
  farm_id, lat, lon, hub_height_m, timestamp, wind_speed, sigma
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Observation, ObservationSource
from .operators import make_loglaw_H


class WindFarmSource(ObservationSource):
    name = "windfarm"

    def __init__(
        self,
        path: str,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        u10_channel: int,
        v10_channel: int,
        default_sigma_ms: float = 0.8,
    ):
        self.path = Path(path)
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.u10_channel = u10_channel
        self.v10_channel = v10_channel
        self.default_sigma_ms = default_sigma_ms

    def load(self, t0: str, t1: str) -> list[Observation]:
        df = pd.read_parquet(self.path)
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
        if df.empty:
            return []

        coords = np.stack(
            [
                df["lat"].to_numpy(dtype=np.float64),
                df["lon"].to_numpy(dtype=np.float64),
                (df["timestamp"].to_numpy("datetime64[ns]") - np.datetime64(t0))
                / np.timedelta64(1, "h"),
                df["hub_height_m"].to_numpy(dtype=np.float64),
            ],
            axis=1,
        )
        sigma = df.get(
            "sigma", pd.Series([self.default_sigma_ms] * len(df))
        ).to_numpy()

        return [
            Observation(
                y=df["wind_speed"].to_numpy(),
                coords=coords,
                sigma_o=sigma,
                H=make_loglaw_H(
                    self.u10_channel,
                    self.v10_channel,
                    coords,
                    self.grid_lat,
                    self.grid_lon,
                ),
                var_name="wind_speed_at_hub_height",
                source=self.name,
            )
        ]
