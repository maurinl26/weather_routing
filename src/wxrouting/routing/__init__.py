"""Routage météo `[PoC routage]` — au-dessus de la prévision ArchesWeatherGen.

v1 : isochrone (Hagiwara) sur un champ de vent réel, polaire voilier, coût = temps.
Cf. note vault *Routage sur ArchesWeatherGen* pour la feuille de route (DP + ensemble).
"""

from .field import ConstantWindField, GriddedWindField, WindField
from .isochrone import IsochroneRouter, Route
from .polar import Polar

__all__ = [
    "WindField",
    "ConstantWindField",
    "GriddedWindField",
    "Polar",
    "IsochroneRouter",
    "Route",
]
