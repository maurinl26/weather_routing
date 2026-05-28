"""Champ de vent interrogeable en (lat, lon, t) continu — pont entre la prévision
(tenseur sur grille modèle) et le routeur.

`[PoC routage]`. En v1 la source est un champ réel sur grille (ERA5 / analyse, en
attendant un checkpoint ArchesWeatherGen produisant la prévision). `ConstantWindField`
sert aux tests et aux cas-jouets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

MS_TO_KNOTS = 1.943844


class WindField(ABC):
    """Source de vent : (lat, lon, instant) -> (u, v) à 10 m, en m/s."""

    @abstractmethod
    def wind_uv(self, lats, lons, when: np.datetime64):
        """Renvoie (u, v) en m/s aux points demandés (tableaux diffusables)."""
        ...

    def tws_twd(self, lats, lons, when: np.datetime64):
        """Renvoie (TWS m/s, TWD degrés) — TWD = direction *d'où vient* le vent."""
        u, v = self.wind_uv(lats, lons, when)
        tws = np.hypot(u, v)
        twd = np.degrees(np.arctan2(-u, -v)) % 360.0
        return tws, twd


class ConstantWindField(WindField):
    """Vent uniforme et stationnaire — cas-jouet / tests."""

    def __init__(self, u_ms: float, v_ms: float):
        self.u = float(u_ms)
        self.v = float(v_ms)

    def wind_uv(self, lats, lons, when: np.datetime64):
        shape = np.broadcast(np.asarray(lats), np.asarray(lons)).shape
        return np.full(shape, self.u), np.full(shape, self.v)


class GriddedWindField(WindField):
    """Champ de vent sur grille (xarray) — interpolation espace + temps.

    Le Dataset porte `u_var`, `v_var` sur des dims (time, latitude, longitude).
    Convention longitude du modèle : [0, 360[ ; les longitudes négatives en entrée
    sont converties automatiquement.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        u_var: str = "10m_u_component_of_wind",
        v_var: str = "10m_v_component_of_wind",
    ):
        self.ds = ds
        self.u_var = u_var
        self.v_var = v_var

    def wind_uv(self, lats, lons, when: np.datetime64):
        lat_da = xr.DataArray(np.atleast_1d(np.asarray(lats, float)), dims="pt")
        lon_da = xr.DataArray(np.atleast_1d(np.asarray(lons, float)) % 360.0, dims="pt")
        sel = self.ds.interp(
            latitude=lat_da, longitude=lon_da, time=when, method="linear"
        )
        u = np.asarray(sel[self.u_var].values, float)
        v = np.asarray(sel[self.v_var].values, float)
        if np.ndim(lats) == 0 and np.ndim(lons) == 0:
            return u.reshape(()), v.reshape(())
        return u, v
