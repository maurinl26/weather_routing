"""Contrat des solveurs d'assimilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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

    @abstractmethod
    def assimilate(
        self,
        x_b: torch.Tensor,                     # background : (B, C, H, W)
        observations: list[Observation],
    ) -> AssimResult: ...
