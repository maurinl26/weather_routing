"""Source scattéromètre ASCAT (Metop) — vent vecteur 10 m océan.

On consomme les produits L2 ou L3 (par ex. KNMI/EUMETSAT) déjà rééchantillonnés.
Format attendu (NetCDF) : variables `wind_speed`, `wind_dir`, masque qualité.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .base import Observation, ObservationSource
from .operators import make_bilinear_H


class ASCATSource(ObservationSource):
    name = "ascat"

    def __init__(
        self,
        path: str,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        u10_channel: int,
        v10_channel: int,
        sigma_ms: float = 1.5,
        quality_mask_var: str | None = "wvc_quality_flag",
    ):
        self.path = Path(path)
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.u10_channel = u10_channel
        self.v10_channel = v10_channel
        self.sigma_ms = sigma_ms
        self.quality_mask_var = quality_mask_var

    def load(self, t0: str, t1: str) -> list[Observation]:
        ds = xr.open_mfdataset(str(self.path), combine="by_coords")
        ds = ds.sel(time=slice(t0, t1))

        wspd = ds["wind_speed"].values.ravel()
        wdir = np.deg2rad(ds["wind_dir"].values.ravel())
        lat = ds["lat"].values.ravel()
        lon = ds["lon"].values.ravel()
        t = (ds["time"].broadcast_like(ds["wind_speed"]).values.ravel()
             .astype("datetime64[ns]") - np.datetime64(t0)) / np.timedelta64(1, "h")

        mask = np.isfinite(wspd) & np.isfinite(wdir)
        if self.quality_mask_var and self.quality_mask_var in ds:
            mask &= ds[self.quality_mask_var].values.ravel() == 0

        wspd, wdir = wspd[mask], wdir[mask]
        lat, lon, t = lat[mask], lon[mask], t[mask]

        u = -wspd * np.sin(wdir)
        v = -wspd * np.cos(wdir)
        coords = np.stack([lat, lon, t], axis=1).astype(np.float64)

        return [
            Observation(
                y=u, coords=coords,
                sigma_o=np.full_like(u, self.sigma_ms),
                H=make_bilinear_H(self.u10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_u_component_of_wind", source=self.name,
            ),
            Observation(
                y=v, coords=coords,
                sigma_o=np.full_like(v, self.sigma_ms),
                H=make_bilinear_H(self.v10_channel, coords, self.grid_lat, self.grid_lon),
                var_name="10m_v_component_of_wind", source=self.name,
            ),
        ]
