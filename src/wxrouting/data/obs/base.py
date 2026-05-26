"""Contrat d'observation — partagé par tous les solveurs d'assimilation.

Une `Observation` représente un batch hétérogène d'obs collectées dans une
fenêtre d'assimilation. L'opérateur H(x) est porté par la source : il sait
comment passer du state vector modèle (C, H, W) aux mesures (N,).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Observation:
    """Batch d'observations sur une fenêtre d'assimilation.

    Attributs
    ---------
    y : (N,)            valeurs mesurées
    coords : (N, 3+)    (lat, lon, time_hours[, level_or_height])
    sigma_o : (N,)      écart-type d'erreur d'observation
    H : Callable        x: (B, C, H, W) -> y_pred: (B, N)
    var_name : str      variable du state vector ciblée (ex. "10m_u_component_of_wind")
    source : str        nom de la source ("ais", "buoys", "windfarm", "ascat")
    """

    y: np.ndarray
    coords: np.ndarray
    sigma_o: np.ndarray
    H: Callable[[torch.Tensor], torch.Tensor]
    var_name: str
    source: str

    def __len__(self) -> int:
        return int(self.y.shape[0])


class ObservationSource(ABC):
    """Source d'observations — produit des `Observation` sur une fenêtre [t0, t1]."""

    name: str

    @abstractmethod
    def load(self, t0: str, t1: str) -> list[Observation]:
        """Renvoie une liste d'Observation (potentiellement une par variable cible)."""
        ...
