"""Routage isochrone (méthode de Hagiwara) sur un champ de vent variable.

`[PoC routage]` — baseline v1. On propage un front d'isochrones : à chaque pas de
temps, depuis chaque point du front, on éventaille un faisceau de caps, on lit le
vent local, on en déduit la vitesse via la polaire, puis on avance. Le front est
élagué par secteur de relèvement depuis le départ (on garde le point le plus
avancé par secteur), ce qui dessine l'enveloppe atteignable.

Limites v1 assumées : un seul membre de prévision (pas d'incertitude), pas de
courants/vagues, coût = temps seul (cf. [[Routage sur ArchesWeatherGen]] pour la
cible DP + ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import MS_TO_KNOTS, WindField
from .geo import (
    angle_to_180,
    destination_point,
    haversine_km,
    initial_bearing_deg,
)
from .polar import Polar

KNOTS_TO_KMH = 1.852


@dataclass
class Route:
    """Route produite par le routeur."""

    points: list[tuple[float, float, np.datetime64]]  # (lat, lon, instant)
    distance_km: float
    duration_h: float
    reached: bool  # True si la destination a été atteinte (sinon best-effort)


@dataclass
class _Node:
    lat: float
    lon: float
    time: np.datetime64
    parent: int  # index du parent dans la liste de nœuds (-1 pour le départ)


class IsochroneRouter:
    def __init__(
        self,
        field: WindField,
        polar: Polar,
        *,
        step_hours: float = 3.0,
        n_headings: int = 36,
        heading_spread_deg: float = 110.0,
        sector_deg: float = 5.0,
        max_steps: int = 240,
    ):
        self.field = field
        self.polar = polar
        self.step_h = float(step_hours)
        self.n_headings = int(n_headings)
        self.spread = float(heading_spread_deg)
        self.sector_deg = float(sector_deg)
        self.max_steps = int(max_steps)

    def solve(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        depart: np.datetime64,
    ) -> Route:
        start_lat, start_lon = float(start[0]), float(start[1])
        end_lat, end_lon = float(end[0]), float(end[1])
        depart = np.datetime64(depart)
        step_dt = np.timedelta64(int(round(self.step_h * 3600)), "s")

        nodes: list[_Node] = [_Node(start_lat, start_lon, depart, -1)]
        front = [0]
        offsets = np.linspace(-self.spread, self.spread, self.n_headings)
        end_idx: int | None = None

        for step in range(self.max_steps):
            when = depart + step * step_dt

            # 1. Terminaison : un point du front peut-il rallier l'arrivée ce pas-ci ?
            finish = self._try_finish(nodes, front, end_lat, end_lon, when)
            if finish is not None:
                parent, finish_h = finish
                arrival = nodes[parent].time + np.timedelta64(
                    int(round(finish_h * 3600)), "s"
                )
                nodes.append(_Node(end_lat, end_lon, arrival, parent))
                end_idx = len(nodes) - 1
                break

            # 2. Expansion du front (vectorisée).
            new_front = self._expand(
                nodes, front, offsets, when + step_dt, start_lat, start_lon,
                end_lat, end_lon,
            )
            if not new_front:
                break  # plus de progression possible (calme / no-go partout)
            front = new_front

        reached = end_idx is not None
        if not reached:
            end_idx = min(
                front, key=lambda i: haversine_km(nodes[i].lat, nodes[i].lon, end_lat, end_lon)
            )

        return self._backtrack(nodes, end_idx, depart, reached)

    # ------------------------------------------------------------------
    def _try_finish(self, nodes, front, end_lat, end_lon, when):
        best: tuple[int, float] | None = None
        for i in front:
            n = nodes[i]
            dist = float(haversine_km(n.lat, n.lon, end_lat, end_lon))
            brg = float(initial_bearing_deg(n.lat, n.lon, end_lat, end_lon))
            tws, twd = self.field.tws_twd(n.lat, n.lon, when)
            twa = angle_to_180(brg - float(twd))
            speed_kmh = float(self.polar.boat_speed(float(tws) * MS_TO_KNOTS, twa)) * KNOTS_TO_KMH
            if speed_kmh > 1e-6 and dist <= speed_kmh * self.step_h:
                finish_h = dist / speed_kmh
                if best is None or finish_h < best[1]:
                    best = (i, finish_h)
        return best

    # ------------------------------------------------------------------
    def _expand(self, nodes, front, offsets, when, start_lat, start_lon, end_lat, end_lon):
        flat = np.array([nodes[i].lat for i in front])
        flon = np.array([nodes[i].lon for i in front])
        tws, twd = self.field.tws_twd(flat, flon, when)
        tws_kn = np.atleast_1d(tws) * MS_TO_KNOTS
        twd = np.atleast_1d(twd)

        brg_end = initial_bearing_deg(flat, flon, end_lat, end_lon)  # (Nf,)
        headings = (brg_end[:, None] + offsets[None, :]) % 360.0     # (Nf, Nh)
        twa = angle_to_180(headings - twd[:, None])
        bsp_kn = self.polar.boat_speed(tws_kn[:, None], twa)         # (Nf, Nh)
        dist_km = bsp_kn * KNOTS_TO_KMH * self.step_h
        nlat, nlon = destination_point(flat[:, None], flon[:, None], headings, dist_km)

        parent = np.repeat(np.asarray(front), self.n_headings)
        nlat, nlon, dist_km = nlat.ravel(), nlon.ravel(), dist_km.ravel()
        keep = dist_km > 1e-3
        if not keep.any():
            return []
        nlat, nlon, parent = nlat[keep], nlon[keep], parent[keep]

        # Élagage par secteur de relèvement depuis le départ : on garde le point
        # le plus éloigné du départ dans chaque secteur (enveloppe atteignable).
        brg0 = initial_bearing_deg(start_lat, start_lon, nlat, nlon)
        d0 = haversine_km(start_lat, start_lon, nlat, nlon)
        sector = np.floor(brg0 / self.sector_deg).astype(int)

        new_front: list[int] = []
        for sec in np.unique(sector):
            idx = np.flatnonzero(sector == sec)
            best = idx[np.argmax(d0[idx])]
            nodes.append(_Node(float(nlat[best]), float(nlon[best]), when, int(parent[best])))
            new_front.append(len(nodes) - 1)
        return new_front

    # ------------------------------------------------------------------
    def _backtrack(self, nodes, end_idx, depart, reached: bool) -> Route:
        path: list[_Node] = []
        i = end_idx
        while i != -1:
            path.append(nodes[i])
            i = nodes[i].parent
        path.reverse()

        points = [(n.lat, n.lon, n.time) for n in path]
        distance_km = sum(
            float(haversine_km(a.lat, a.lon, b.lat, b.lon))
            for a, b in zip(path[:-1], path[1:], strict=False)
        )
        duration_h = float((path[-1].time - depart) / np.timedelta64(1, "h"))
        return Route(
            points=points,
            distance_km=distance_km,
            duration_h=duration_h,
            reached=reached,
        )
