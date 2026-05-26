"""DataFrame brut → list[Observation] : seul point qui connaît le state vector.

Le mapping `{variable_cf: ChannelSpec}` paramètre l'adapter. Une variable
qui n'est pas dans le mapping est ignorée silencieusement (un fetcher peut
remonter du `wave_height` même si le modèle ne l'expose pas).

Décomposition automatique : si la source remonte `wind_speed_at_10m` +
`wind_from_direction` (cas typique des bouées CMEMS et VOS) et que le
modèle attend `eastward_wind_at_10m` + `northward_wind_at_10m`, l'adapter
synthétise les composantes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cf_names import (
    AIR_PRESSURE_AT_SEA_LEVEL,
    EASTWARD_WIND_10M,
    NORTHWARD_WIND_10M,
    WIND_FROM_DIRECTION,
    WIND_SPEED_10M,
    WIND_SPEED_AT_HUB,
    canonical,
    decompose_wind,
)
from .base import Observation
from .operators import make_bilinear_H, make_loglaw_H


@dataclass
class ChannelSpec:
    """Comment mapper une variable canonique CF vers un canal du state vector."""

    channel_index: int
    sigma_default: float                  # erreur d'obs par défaut (unités SI)
    extra_channel_index: int | None = None  # pour le log-loi (v10 quand on cible u10)


# Mapping par défaut aligné sur le registry ArchesWeatherGen (cf. data/registry.py).
# L'ordre des canaux y est : [u10, v10, t2m, msl, then 5*13 level vars].
DEFAULT_MAPPING: dict[str, ChannelSpec] = {
    EASTWARD_WIND_10M:           ChannelSpec(channel_index=0, sigma_default=1.5),
    NORTHWARD_WIND_10M:          ChannelSpec(channel_index=1, sigma_default=1.5),
    AIR_PRESSURE_AT_SEA_LEVEL:   ChannelSpec(channel_index=3, sigma_default=50.0),  # Pa
    # log-loi : on a besoin de u10 ET v10 pour calculer ||V(z)||
    WIND_SPEED_AT_HUB:           ChannelSpec(channel_index=0, sigma_default=0.8,
                                             extra_channel_index=1),
}


class ObservationAdapter:
    """Convertit un DataFrame brut (cf. Fetcher) en list[Observation]."""

    def __init__(
        self,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        mapping: dict[str, ChannelSpec] | None = None,
        wind_decomposition: bool = True,
    ):
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.mapping = mapping or DEFAULT_MAPPING
        self.wind_decomposition = wind_decomposition

    # ------------------------------------------------------------------
    def adapt(self, df: pd.DataFrame, t0: str, source: str) -> list[Observation]:
        if df.empty:
            return []
        df = df.copy()
        df["variable"] = df["variable"].map(canonical)

        if self.wind_decomposition:
            df = self._decompose_wind_if_needed(df)

        out: list[Observation] = []
        for var, sub in df.groupby("variable"):
            spec = self.mapping.get(var)
            if spec is None:
                continue
            out.append(self._build(sub, var, spec, t0, source))
        return out

    # ------------------------------------------------------------------
    def _decompose_wind_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Si (WSPD, WDIR) présent mais (u10, v10) absent → décompose."""
        target = {EASTWARD_WIND_10M, NORTHWARD_WIND_10M}
        present = set(df["variable"])
        if target & present:
            return df
        if {WIND_SPEED_10M, WIND_FROM_DIRECTION}.issubset(present):
            ws = df[df["variable"] == WIND_SPEED_10M].set_index(
                ["timestamp", "lat", "lon"]
            )["value"]
            wd = df[df["variable"] == WIND_FROM_DIRECTION].set_index(
                ["timestamp", "lat", "lon"]
            )["value"]
            joined = ws.to_frame("ws").join(wd.to_frame("wd"), how="inner").reset_index()
            u, v = decompose_wind(joined["ws"].to_numpy(), joined["wd"].to_numpy())
            uframe = joined.assign(variable=EASTWARD_WIND_10M, value=u).drop(
                columns=["ws", "wd"]
            )
            vframe = joined.assign(variable=NORTHWARD_WIND_10M, value=v).drop(
                columns=["ws", "wd"]
            )
            # On garde aussi les autres variables d'origine (pression, vagues…).
            others = df[~df["variable"].isin([WIND_SPEED_10M, WIND_FROM_DIRECTION])]
            return pd.concat([others, uframe, vframe], ignore_index=True)
        return df

    # ------------------------------------------------------------------
    def _build(
        self,
        sub: pd.DataFrame,
        var: str,
        spec: ChannelSpec,
        t0: str,
        source: str,
    ) -> Observation:
        t0_dt = np.datetime64(t0)
        t_h = (sub["timestamp"].to_numpy("datetime64[ns]") - t0_dt) / np.timedelta64(
            1, "h"
        )
        coords_cols = [sub["lat"].to_numpy(np.float64), sub["lon"].to_numpy(np.float64), t_h]
        if "hub_height_m" in sub:
            coords_cols.append(sub["hub_height_m"].to_numpy(np.float64))
        coords = np.stack(coords_cols, axis=1)

        sigma = sub.get("uncertainty")
        sigma_arr = (
            np.full(len(sub), spec.sigma_default, dtype=np.float64)
            if sigma is None or sigma.isna().all()
            else sigma.fillna(spec.sigma_default).to_numpy(np.float64)
        )

        H = self._H_for(var, spec, coords)
        return Observation(
            y=sub["value"].to_numpy(np.float64),
            coords=coords,
            sigma_o=sigma_arr,
            H=H,
            var_name=var,
            source=source,
        )

    def _H_for(
        self, var: str, spec: ChannelSpec, coords: np.ndarray
    ) -> Callable:
        if var == WIND_SPEED_AT_HUB:
            assert spec.extra_channel_index is not None, (
                "WIND_SPEED_AT_HUB requires extra_channel_index (v10) in ChannelSpec"
            )
            return make_loglaw_H(
                spec.channel_index,
                spec.extra_channel_index,
                coords,
                self.grid_lat,
                self.grid_lon,
            )
        return make_bilinear_H(
            spec.channel_index, coords, self.grid_lat, self.grid_lon
        )
