"""Métrique de risque `[PoC routage]` — exposition au vent fort.

Choix de v1 : le risque d'un tronçon = **temps passé au-delà d'un seuil de
confort de vent**, pondéré par le dépassement. Unité : nœuds·heure au-dessus du
seuil. Simple, monotone, et différencie clairement une route « rapide mais
musclée » d'une route « plus longue mais calme » — ce qu'il faut pour un front
de Pareto temps vs risque.

Extensions naturelles (non faites ici) : intégrer les vagues (Hs), ou évaluer le
risque **sur tout l'ensemble** (proba de dépasser le seuil) — cf.
[[Architecture app routage]] §12.
"""

from __future__ import annotations

import numpy as np

from .field import MS_TO_KNOTS, WindField

DEFAULT_TWS_SAFE_KN = 22.0  # ~ force 6 ; au-delà, l'inconfort/risque s'accumule


def leg_exposure(tws_ms: float, duration_h: float, tws_safe_kn: float = DEFAULT_TWS_SAFE_KN) -> float:
    """Exposition d'un tronçon : (nœuds au-delà du seuil) × heures."""
    over = max(0.0, float(tws_ms) * MS_TO_KNOTS - tws_safe_kn)
    return over * float(duration_h)


def route_risk(field: WindField, route, tws_safe_kn: float = DEFAULT_TWS_SAFE_KN) -> float:
    """Exposition totale d'une route sur un champ donné (kn·h).

    Réutilisable pour re-scorer une route sur n'importe quel membre d'ensemble
    (risque robuste = max/quantile sur les membres).
    """
    pts = route.points
    total = 0.0
    for (lat, lon, t0), (_, _, t1) in zip(pts[:-1], pts[1:], strict=False):
        dur_h = float((np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, "h"))
        tws, _ = field.tws_twd(lat, lon, np.datetime64(t0))
        total += leg_exposure(float(tws), dur_h, tws_safe_kn)
    return total
