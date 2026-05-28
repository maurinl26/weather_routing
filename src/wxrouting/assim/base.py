"""Contrat des solveurs d'assimilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ..data.obs import Observation


@dataclass
class AssimResult:
    """Résultat d'une fenêtre d'assimilation.

    x_a       : analyse (B, C, H, W) — ensemble moyen
    x_a_ens   : ensemble d'analyses (N, B, C, H, W) si disponible, sinon None
    innov     : innovation d = y - H(x_b), par source — pour diagnostic
    """

    x_a: torch.Tensor
    x_a_ens: torch.Tensor | None
    innov: dict[str, torch.Tensor]


class Assimilator(ABC):
    """Solveur d'assimilation — consomme un état de background + des obs."""

    # True si le solveur a besoin du modèle (sampler / score). Le runner refuse
    # alors de tourner sans checkpoint. Nudging reste à False.
    requires_model: bool = False

    def bind_model(self, pl_module: Any) -> None:
        """Branche les sorties du modèle (sampler / score / denoise) sur le solveur.

        No-op par défaut (cas nudging). Les solveurs qui consomment le modèle
        surchargent et lèvent si l'interface attendue est absente — la dépendance
        est ainsi déclarée par le solveur, pas devinée par le runner.
        """
        return

    @abstractmethod
    def assimilate(
        self,
        x_b: torch.Tensor,                     # background : (B, C, H, W)
        observations: list[Observation],
    ) -> AssimResult: ...
