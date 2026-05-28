"""Routage d'ensemble `[PoC routage]` — route chaque membre, agrège.

On exécute l'isochrone sur **chaque membre** de l'ensemble (cf. caractère
génératif d'ArchesWeatherGen / des modèles NWP ENS) puis on résume :

- toutes les routes membres (le « spaghetti » à afficher),
- une **route recommandée** (membre dont la durée est la plus proche de la médiane),
- la **distribution des durées** (p10 / p50 / p90 / moyenne),
- la **fraction de membres** ayant atteint l'arrivée.

Le front Pareto (temps vs risque) viendra quand la métrique de risque sera tranchée
(cf. [[Architecture app routage]] §12) — il s'appuiera sur ces routes membres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import EnsembleWindField
from .isochrone import IsochroneRouter, Route
from .polar import Polar


@dataclass
class EnsembleRoute:
    member_routes: list[Route]        # une route par membre (spaghetti)
    recommended: Route                # route représentative (durée ~ médiane)
    duration_stats_h: dict[str, float]  # {mean, p10, p50, p90} sur les membres atteints
    reached_fraction: float           # part des membres ayant atteint l'arrivée


def route_ensemble(
    ensemble: EnsembleWindField,
    polar: Polar,
    start: tuple[float, float],
    end: tuple[float, float],
    depart: np.datetime64,
    **router_kwargs,
) -> EnsembleRoute:
    routes = [
        IsochroneRouter(member, polar, **router_kwargs).solve(start, end, depart)
        for member in ensemble.members
    ]

    reached = [r for r in routes if r.reached]
    # Stats sur les membres atteints (sinon repli sur tous, faute de mieux).
    basis = reached if reached else routes
    durations = np.array([r.duration_h for r in basis], dtype=float)
    stats = {
        "mean": float(durations.mean()),
        "p10": float(np.percentile(durations, 10)),
        "p50": float(np.percentile(durations, 50)),
        "p90": float(np.percentile(durations, 90)),
    }

    # Route recommandée : celle dont la durée est la plus proche de la médiane.
    median = stats["p50"]
    recommended = min(basis, key=lambda r: abs(r.duration_h - median))

    return EnsembleRoute(
        member_routes=routes,
        recommended=recommended,
        duration_stats_h=stats,
        reached_fraction=len(reached) / len(routes),
    )
