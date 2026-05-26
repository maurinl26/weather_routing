"""Générateur d'AIS synthétique — voiliers d'opportunité pour TP.

Produit N trajectoires great-circle aléatoires dans la bbox, échantillonne
le vent d'un champ de référence (xarray.Dataset ERA5) le long de la route,
ajoute un bruit gaussien — comme un capteur perso embarqué.

Permet aux étudiants de tourner les TP DA sans clé API, et donne un terrain
de jeu propre pour comparer nudging / EnKF / DPS avec une vérité connue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..cf_names import EASTWARD_WIND_10M, NORTHWARD_WIND_10M
from .base import BBox, Fetcher, RawObsSchema


class SyntheticAISFetcher(Fetcher):
    name = "synthetic_ais"

    def __init__(
        self,
        reference_field: xr.Dataset | None = None,
        n_vessels: int = 20,
        n_points_per_track: int = 24,
        speed_ms: float = 6.0,            # ~12 nœuds
        noise_sigma_ms: float = 1.0,      # bruit sur u, v
        seed: int = 42,
        u_var: str = "10m_u_component_of_wind",
        v_var: str = "10m_v_component_of_wind",
    ):
        self.reference_field = reference_field
        self.n_vessels = n_vessels
        self.n_points_per_track = n_points_per_track
        self.speed_ms = speed_ms
        self.noise_sigma_ms = noise_sigma_ms
        self.rng = np.random.default_rng(seed)
        self.u_var = u_var
        self.v_var = v_var

    # ------------------------------------------------------------------
    def _great_circle_track(
        self, start: tuple[float, float], end: tuple[float, float], n: int
    ) -> np.ndarray:
        """Interpolation linéaire (suffisante à l'échelle Golfe de Gascogne)."""
        lats = np.linspace(start[0], end[0], n)
        lons = np.linspace(start[1], end[1], n)
        return np.stack([lats, lons], axis=1)

    def _sample_ref(self, lat: float, lon: float, t: np.datetime64) -> tuple[float, float]:
        if self.reference_field is None:
            return (0.0, 0.0)
        # ERA5 longitude ∈ [0, 360[
        lon_q = lon % 360.0
        u = float(self.reference_field[self.u_var].interp(
            latitude=lat, longitude=lon_q, time=t, method="linear"
        ))
        v = float(self.reference_field[self.v_var].interp(
            latitude=lat, longitude=lon_q, time=t, method="linear"
        ))
        return (u, v)

    # ------------------------------------------------------------------
    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        t_start = np.datetime64(t0)
        t_end = np.datetime64(t1)
        rows: list[dict] = []

        for vessel in range(self.n_vessels):
            start_lat = self.rng.uniform(bbox.lat_min, bbox.lat_max)
            start_lon = self.rng.uniform(bbox.lon_min, bbox.lon_max)
            end_lat = self.rng.uniform(bbox.lat_min, bbox.lat_max)
            end_lon = self.rng.uniform(bbox.lon_min, bbox.lon_max)

            track = self._great_circle_track(
                (start_lat, start_lon), (end_lat, end_lon), self.n_points_per_track
            )
            duration_s = (t_end - t_start) / np.timedelta64(1, "s")
            times = t_start + np.linspace(
                0, duration_s, self.n_points_per_track
            ).astype("timedelta64[s]")

            for (lat, lon), t in zip(track, times, strict=True):
                u_true, v_true = self._sample_ref(lat, lon, t)
                u_obs = u_true + self.rng.normal(0.0, self.noise_sigma_ms)
                v_obs = v_true + self.rng.normal(0.0, self.noise_sigma_ms)
                rows.append({
                    "timestamp": pd.Timestamp(t).tz_localize("UTC"),
                    "lat": lat, "lon": lon,
                    "variable": EASTWARD_WIND_10M, "value": u_obs,
                    "uncertainty": self.noise_sigma_ms,
                    "platform_id": f"SYN-{vessel:03d}",
                })
                rows.append({
                    "timestamp": pd.Timestamp(t).tz_localize("UTC"),
                    "lat": lat, "lon": lon,
                    "variable": NORTHWARD_WIND_10M, "value": v_obs,
                    "uncertainty": self.noise_sigma_ms,
                    "platform_id": f"SYN-{vessel:03d}",
                })

        return RawObsSchema.validate(pd.DataFrame(rows))
