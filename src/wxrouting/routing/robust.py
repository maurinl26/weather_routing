"""Routage robuste sous incertitude `[PoC routage]` — Pareto honnête sur l'ensemble.

Principe (cf. note vault *Routage sur ArchesWeatherGen* §incertitude) :

1. On cherche les géométries de routes candidates par **DP sur le champ moyen**
   d'ensemble (tractable : une seule recherche, pas N).
2. On **re-score le risque de chaque route sur CHAQUE membre** (`route_risk`),
   ce qui restitue la dispersion que la moyenne avait lissée.
3. Le front de Pareto oppose alors la **durée espérée** (champ moyen) au
   **risque robuste** (agrégat sur les membres, p90 par défaut).

Hypothèses assumées :
- *Durée* = temps sur le champ moyen (espérance) ; on ne re-time pas la route par
  membre (la dispersion de durée est une extension future).
- *Risque* = exposition au vent fort (`risk.leg_exposure`), re-scorée par membre
  sur les **mêmes (position, heure)** que la route du champ moyen (même planning,
  météo différente).
- *Robuste* = agrégat configurable des expositions membres (`mean`/`p90`/`max`) ;
  on expose aussi `rough_probability` = part des membres où la route rencontre du
  vent au-delà du seuil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dp import DPRouter, _non_dominated_by
from .field import EnsembleWindField
from .isochrone import Route
from .polar import Polar
from .risk import DEFAULT_TWS_SAFE_KN, route_risk


@dataclass
class RobustRoute:
    route: Route
    duration_h: float          # sur le champ moyen (durée espérée)
    risk_mean: float           # exposition moyenne sur les membres (kn·h)
    risk_p90: float            # exposition au 90e centile (conservateur)
    risk_max: float            # pire membre
    rough_probability: float   # part des membres avec exposition > 0
    member_risks: list[float]


def robust_pareto_routes(
    ensemble: EnsembleWindField,
    polar: Polar,
    start: tuple[float, float],
    end: tuple[float, float],
    depart,
    *,
    risk_weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0),
    risk_aggregator: str = "p90",
    tws_safe_kn: float = DEFAULT_TWS_SAFE_KN,
    **dp_kwargs,
) -> list[RobustRoute]:
    """Front de Pareto (durée espérée vs risque robuste) sur un ensemble."""
    mean = ensemble.mean_field()
    candidates = [
        DPRouter(
            mean, polar, risk_weight=w, tws_safe_kn=tws_safe_kn, **dp_kwargs
        ).solve(start, end, depart)
        for w in risk_weights
    ]

    robust: list[RobustRoute] = []
    for c in candidates:
        if not c.reached:
            continue
        member_risks = [
            route_risk(m, c.route, tws_safe_kn) for m in ensemble.members
        ]
        arr = np.array(member_risks, dtype=float)
        robust.append(
            RobustRoute(
                route=c.route,
                duration_h=c.duration_h,
                risk_mean=float(arr.mean()),
                risk_p90=float(np.percentile(arr, 90)),
                risk_max=float(arr.max()),
                rough_probability=float((arr > 0).mean()),
                member_risks=member_risks,
            )
        )

    key = _aggregator_key(risk_aggregator)
    return _non_dominated_by(robust, lambda r: r.duration_h, key)


def _aggregator_key(name: str):
    return {
        "mean": lambda r: r.risk_mean,
        "p90": lambda r: r.risk_p90,
        "max": lambda r: r.risk_max,
    }[name]
