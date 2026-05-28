"""Polaire de vitesse voilier : (TWS, TWA) -> vitesse bateau.

`[PoC routage]`. Convention nautique : vitesses en **nœuds**, TWS en nœuds, TWA
(true wind angle) en degrés dans [0, 180]. La conversion depuis le vent du modèle
(m/s) est faite par le routeur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Polar:
    tws: np.ndarray    # (T,) nœuds, croissant
    twa: np.ndarray    # (A,) degrés [0, 180], croissant
    speed: np.ndarray  # (T, A) vitesse bateau en nœuds

    def boat_speed(self, tws_kn, twa_deg):
        """Vitesse bateau (nœuds) par interpolation bilinéaire sur la table.

        Diffusable : `tws_kn` et `twa_deg` peuvent être scalaires ou tableaux.
        Hors grille → clampé aux bords (plateau de coque, no-go).
        """
        tws_b, twa_b = np.broadcast_arrays(
            np.asarray(tws_kn, float), np.asarray(twa_deg, float)
        )
        ti = np.clip(np.searchsorted(self.tws, tws_b) - 1, 0, len(self.tws) - 2)
        ai = np.clip(np.searchsorted(self.twa, twa_b) - 1, 0, len(self.twa) - 2)
        t0, t1 = self.tws[ti], self.tws[ti + 1]
        a0, a1 = self.twa[ai], self.twa[ai + 1]
        wt = np.clip((tws_b - t0) / (t1 - t0), 0.0, 1.0)
        wa = np.clip((twa_b - a0) / (a1 - a0), 0.0, 1.0)
        s = (
            (1 - wt) * (1 - wa) * self.speed[ti, ai]
            + (1 - wt) * wa * self.speed[ti, ai + 1]
            + wt * (1 - wa) * self.speed[ti + 1, ai]
            + wt * wa * self.speed[ti + 1, ai + 1]
        )
        return s

    @classmethod
    def from_csv(cls, path: str | Path) -> Polar:
        """Charge une polaire CSV : 1re ligne = TWA, 1re colonne = TWS, cellules = vitesse.

        Séparateur `;` (format courant des polaires exportées). Première cellule ignorée.
        """
        rows = [
            line.split(";")
            for line in Path(path).read_text().strip().splitlines()
            if line.strip()
        ]
        twa = np.array([float(x) for x in rows[0][1:]])
        tws = np.array([float(r[0]) for r in rows[1:]])
        speed = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
        return cls(tws=tws, twa=twa, speed=speed)

    @classmethod
    def example(cls) -> Polar:
        """Polaire générique de monocoque de croisière (modèle analytique lissé).

        Suffisante pour les démonstrations PoC ; à remplacer par une polaire réelle
        (publique IMOCA/ORMA ou fournie par un partenaire) le moment venu.
        """
        tws = np.array([0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 35.0])
        twa = np.array([0.0, 30, 40, 50, 60, 75, 90, 110, 120, 135, 150, 165, 180])
        speed = np.array(
            [[_model_speed(w, a) for a in twa] for w in tws], dtype=float
        )
        return cls(tws=tws, twa=twa, speed=speed)


def _model_speed(tws_kn: float, twa_deg: float) -> float:
    """Modèle analytique simple : no-go < 30°, montée puis plateau de coque."""
    if twa_deg < 30.0:
        return 0.0
    s_wind = 8.5 * (1.0 - np.exp(-tws_kn / 8.0))  # plateau ~8.5 nœuds
    if twa_deg <= 95.0:
        ang = (twa_deg - 30.0) / (95.0 - 30.0)            # remontée au près
    else:
        ang = 1.0 - 0.35 * (twa_deg - 95.0) / (180.0 - 95.0)  # léger taper au portant
    return float(s_wind * ang)
