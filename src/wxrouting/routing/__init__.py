"""Routage météo `[PoC routage]` — au-dessus de la prévision ArchesWeatherGen.

v1 : isochrone (Hagiwara) sur un champ de vent réel, polaire voilier, coût = temps.
Cf. note vault *Routage sur ArchesWeatherGen* pour la feuille de route (DP + ensemble).
"""

from .dp import DPResult, DPRouter, pareto_routes
from .ensemble import EnsembleRoute, route_ensemble
from .field import (
    ConstantWindField,
    EnsembleWindField,
    GriddedWindField,
    WindField,
)
from .isochrone import IsochroneRouter, Route
from .polar import Polar
from .risk import leg_exposure, route_risk

__all__ = [
    "WindField",
    "ConstantWindField",
    "GriddedWindField",
    "EnsembleWindField",
    "Polar",
    "IsochroneRouter",
    "Route",
    "EnsembleRoute",
    "route_ensemble",
    "DPRouter",
    "DPResult",
    "pareto_routes",
    "leg_exposure",
    "route_risk",
]
