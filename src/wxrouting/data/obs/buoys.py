"""Source bouées fixes Météo-France / CMEMS.

Format NetCDF CF-compliant typique CMEMS in-situ TAC :
  variables WSPD, WDIR, ATMS ; coords TIME, LATITUDE, LONGITUDE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .base import Observation, ObservationSource
from .operators import make_bilinear_H


class BuoySource(ObservationSource):
    name = "buoys"

    def __init__(
        self,
        path: str,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        u10_channel: int,
        v10_channel: int,
        msl_channel: int,
        sigma_wind: float = 1.0,
        sigma_msl: float = 50.0,  # Pa
    ):
        self.path = Path(path)
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.u10_channel = u10_channel
        self.v10_channel = v10_channel
        self.msl_channel = msl_channel
        self.sigma_wind = sigma_wind
        self.sigma_msl = sigma_msl

    def load(self, t0: str, t1: str) -> list[Observation]:
        ds = xr.open_mfdataset(str(self.path), combine="by_coords")
        ds = ds.sel(TIME=slice(t0, t1))
        if ds["TIME"].size == 0:
            return []

        # Vent : on décompose (WSPD, WDIR) -> (u, v). Convention météo :
        # WDIR = direction d'où vient le vent, 0° = Nord.
        wspd = ds["WSPD"].values.ravel()
        wdir_rad = np.deg2rad(ds["WDIR"].values.ravel())
        u = -wspd * np.sin(wdir_rad)
        v = -wspd * np.cos(wdir_rad)

        lat = np.broadcast_to(ds["LATITUDE"].values, ds["WSPD"].shape).ravel()
        lon = np.broadcast_to(ds["LONGITUDE"].values, ds["WSPD"].shape).ravel()
        t = (ds["TIME"].values.astype("datetime64[ns]") - np.datetime64(t0)) / np.timedelta64(1, "h")
        t = np.broadcast_to(t[:, None], ds["WSPD"].shape).ravel()
        coords = np.stack([lat, lon, t], axis=1).astype(np.float64)

        obs = [
            Observation(
                y=u, coords=coords,
                sigma_o=np.full_like(u, self.sigma_wind),
                H=make_bilinear_H(self.u10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_u_component_of_wind", source=self.name,
            ),
            Observation(
                y=v, coords=coords,
                sigma_o=np.full_like(v, self.sigma_wind),
                H=make_bilinear_H(self.v10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_v_component_of_wind", source=self.name,
            ),
        ]
        if "ATMS" in ds:
            p = ds["ATMS"].values.ravel()
            obs.append(
                Observation(
                    y=p, coords=coords,
                    sigma_o=np.full_like(p, self.sigma_msl),
                    H=make_bilinear_H(self.msl_channel, coords, self.grid_lat, self.grid_lon),
                    var_name="mean_sea_level_pressure", source=self.name,
                )
            )
        return obs
