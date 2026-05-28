"""Routage par programmation dynamique `[PoC routage]` — coût composite.

Plus court chemin **dépendant du temps** sur une grille lat/lon : depuis chaque
nœud (à son heure d'arrivée), on évalue le vent, la polaire donne la vitesse, donc
le temps de tronçon ; le coût minimisé est

    coût = Σ ( temps_tronçon + λ · exposition_vent_fort )

`λ = risk_weight`. Dijkstra (coûts ≥ 0) sur le graphe ; on suit l'heure d'arrivée
par nœud pour interroger le vent au bon pas de temps. Balayer λ donne le **front
de Pareto** temps vs risque (`pareto_routes`).

Plus flexible que l'isochrone (contraintes, coût composite) ; cf.
[[Routage sur ArchesWeatherGen]].
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import gcd

import numpy as np

from .field import MS_TO_KNOTS, WindField
from .geo import angle_to_180, haversine_km, initial_bearing_deg
from .isochrone import KNOTS_TO_KMH, Route
from .polar import Polar
from .risk import DEFAULT_TWS_SAFE_KN, leg_exposure


@dataclass
class DPResult:
    route: Route
    duration_h: float
    risk: float          # exposition totale au vent fort (kn·h)
    reached: bool


def _neighbor_offsets(radius: int = 2) -> list[tuple[int, int]]:
    """Décalages de voisinage coprimes (|di|,|dj| ≤ radius) — 16 caps pour radius=2.

    La coprimalité évite de « sauter » par-dessus un nœud intermédiaire colinéaire.
    """
    offs = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            if gcd(abs(di), abs(dj)) == 1:
                offs.append((di, dj))
    return offs


class DPRouter:
    def __init__(
        self,
        field: WindField,
        polar: Polar,
        *,
        grid_deg: float = 0.25,
        margin_deg: float = 1.0,
        risk_weight: float = 0.0,
        tws_safe_kn: float = DEFAULT_TWS_SAFE_KN,
        neighbor_radius: int = 2,
        max_hours: float = 240.0,
    ):
        self.field = field
        self.polar = polar
        self.grid_deg = float(grid_deg)
        self.margin_deg = float(margin_deg)
        self.risk_weight = float(risk_weight)
        self.tws_safe_kn = float(tws_safe_kn)
        self.offsets = _neighbor_offsets(neighbor_radius)
        self.max_hours = float(max_hours)

    def solve(self, start: tuple[float, float], end: tuple[float, float], depart) -> DPResult:
        depart = np.datetime64(depart)
        lats, lons = self._grid(start, end)
        nlat, nlon = len(lats), len(lons)
        s = (_nearest(lats, start[0]), _nearest(lons, start[1]))
        e = (_nearest(lats, end[0]), _nearest(lons, end[1]))

        def nid(i: int, j: int) -> int:
            return i * nlon + j

        n = nlat * nlon
        cost = np.full(n, np.inf)
        time_h = np.full(n, np.inf)       # heures depuis depart
        risk = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)

        src = nid(*s)
        cost[src] = 0.0
        time_h[src] = 0.0
        risk[src] = 0.0
        pq: list[tuple[float, int]] = [(0.0, src)]
        end_id = nid(*e)

        while pq:
            c, u = heapq.heappop(pq)
            if c > cost[u]:
                continue
            if u == end_id:
                break
            ui, uj = divmod(u, nlon)
            ulat, ulon = float(lats[ui]), float(lons[uj])
            when = depart + np.timedelta64(int(round(time_h[u] * 3600)), "s")
            tws, twd = self.field.tws_twd(ulat, ulon, when)
            tws = float(tws)
            twd = float(twd)
            tws_kn = tws * MS_TO_KNOTS

            for di, dj in self.offsets:
                vi, vj = ui + di, uj + dj
                if not (0 <= vi < nlat and 0 <= vj < nlon):
                    continue
                v = nid(vi, vj)
                vlat, vlon = float(lats[vi]), float(lons[vj])
                dist = float(haversine_km(ulat, ulon, vlat, vlon))
                brg = float(initial_bearing_deg(ulat, ulon, vlat, vlon))
                twa = angle_to_180(brg - twd)
                speed_kmh = float(self.polar.boat_speed(tws_kn, twa)) * KNOTS_TO_KMH
                if speed_kmh <= 1e-6:
                    continue  # no-go
                leg_h = dist / speed_kmh
                new_time = time_h[u] + leg_h
                if new_time > self.max_hours:
                    continue
                leg_risk = leg_exposure(tws, leg_h, self.tws_safe_kn)
                new_cost = cost[u] + leg_h + self.risk_weight * leg_risk
                if new_cost < cost[v]:
                    cost[v] = new_cost
                    time_h[v] = new_time
                    risk[v] = risk[u] + leg_risk
                    prev[v] = u
                    heapq.heappush(pq, (new_cost, v))

        reached = np.isfinite(cost[end_id])
        route = self._reconstruct(prev, end_id, lats, lons, nlon, depart, time_h, reached)
        return DPResult(
            route=route,
            duration_h=float(time_h[end_id]) if reached else float("inf"),
            risk=float(risk[end_id]) if reached else float("inf"),
            reached=bool(reached),
        )

    # ------------------------------------------------------------------
    def _grid(self, start, end):
        lat_lo = min(start[0], end[0]) - self.margin_deg
        lat_hi = max(start[0], end[0]) + self.margin_deg
        lon_lo = min(start[1], end[1]) - self.margin_deg
        lon_hi = max(start[1], end[1]) + self.margin_deg
        lats = np.arange(lat_lo, lat_hi + 1e-9, self.grid_deg)
        lons = np.arange(lon_lo, lon_hi + 1e-9, self.grid_deg)
        return lats, lons

    def _reconstruct(self, prev, end_id, lats, lons, nlon, depart, time_h, reached) -> Route:
        if not reached:
            return Route(points=[], distance_km=0.0, duration_h=float("inf"), reached=False)
        chain = []
        u = end_id
        while u != -1:
            chain.append(u)
            u = prev[u]
        chain.reverse()

        points = []
        for u in chain:
            i, j = divmod(u, nlon)
            t = depart + np.timedelta64(int(round(time_h[u] * 3600)), "s")
            points.append((float(lats[i]), float(lons[j]), t))
        distance_km = sum(
            float(haversine_km(a[0], a[1], b[0], b[1]))
            for a, b in zip(points[:-1], points[1:], strict=False)
        )
        return Route(
            points=points,
            distance_km=distance_km,
            duration_h=float(time_h[end_id]),
            reached=True,
        )


def _nearest(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def pareto_routes(
    field: WindField,
    polar: Polar,
    start: tuple[float, float],
    end: tuple[float, float],
    depart,
    *,
    risk_weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0),
    **dp_kwargs,
) -> list[DPResult]:
    """Balaye λ et renvoie le front de Pareto (durée vs risque), trié par durée.

    Chaque λ donne une route optimale pour ce compromis ; on conserve les routes
    **non dominées** (aucune autre n'est à la fois plus rapide ET moins risquée).
    """
    candidates = [
        DPRouter(field, polar, risk_weight=w, **dp_kwargs).solve(start, end, depart)
        for w in risk_weights
    ]
    candidates = [c for c in candidates if c.reached]
    return _non_dominated_by(candidates, lambda r: r.duration_h, lambda r: r.risk)


def _non_dominated_by(items, time_key, risk_key):
    """Front de Pareto générique : minimiser `time_key` ET `risk_key`.

    Garde les éléments non dominés, triés par temps, dédoublonnés sur (temps, risque).
    """
    front = []
    for r in items:
        if any(
            time_key(o) <= time_key(r)
            and risk_key(o) <= risk_key(r)
            and (time_key(o) < time_key(r) or risk_key(o) < risk_key(r))
            for o in items
            if o is not r
        ):
            continue
        front.append(r)
    seen: set[tuple[float, float]] = set()
    uniq = []
    for r in sorted(front, key=time_key):
        k = (round(time_key(r), 3), round(risk_key(r), 3))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq
