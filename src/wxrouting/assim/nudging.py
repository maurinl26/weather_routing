"""Nudging — baseline pédagogique.

Idée : on rapproche le state vector des observations par un terme de relaxation
proportionnel à l'innovation. Ce n'est PAS un solveur bayésien, mais c'est le
plus simple pour introduire la notion d'opérateur d'observation H et
d'innovation d = y - H(x).
"""

from __future__ import annotations

import torch

from ..data.obs import Observation
from .base import Assimilator, AssimResult


class NudgingAssimilator(Assimilator):
    def __init__(self, alpha: float = 0.1, window_hours: int = 24, **_: object):
        self.alpha = alpha
        self.window_hours = window_hours

    def assimilate(
        self, x_b: torch.Tensor, observations: list[Observation]
    ) -> AssimResult:
        x = x_b.clone()
        innov: dict[str, torch.Tensor] = {}

        # Pour chaque obs : on calcule d = y - H(x_b), puis on "pousse" le canal
        # ciblé vers les obs au voisinage. Implémentation naïve à la maille la
        # plus proche — suffisante pour la baseline.
        for obs in observations:
            y_pred = obs.H(x_b)                                # (B, N)
            y_true = torch.from_numpy(obs.y).to(x_b.device).float()
            d = y_true.unsqueeze(0) - y_pred                   # (B, N)
            innov[obs.source + "/" + obs.var_name] = d

            # Note pédagogique : un vrai nudging spatial nécessiterait
            # l'adjoint de H (splatting des innovations sur la grille).
            # On se contente ici d'un terme global pondéré par la moyenne
            # des innovations — la forme la plus pauvre mais la plus claire.
            # Voir EnKFAssimilator / DPSAssimilator pour les versions correctes.
            channel = _channel_for(obs)
            if channel is not None:
                x[:, channel] += self.alpha * d.mean()

        return AssimResult(x_a=x, x_a_ens=None, innov=innov)


def _channel_for(obs: Observation) -> int | None:
    """Récupère l'index canal cible à partir de l'opérateur H.

    Convention : les H construits par `make_bilinear_H` capturent leur
    `channel_index` dans la closure ; le solveur a besoin de ce mapping
    pour appliquer le terme de relaxation. Pour rester découplé, on lit
    une annotation `channel_index` qu'on a posée à la construction.
    """
    return getattr(obs.H, "channel_index", None)
