"""Observations d'opportunité — interface unique pour l'assimilation.

Le pipeline est en deux temps :
- un `Fetcher` (cf. `wxrouting.data.fetchers`) télécharge des obs brutes
  (`pandas.DataFrame`), agnostique du paramètre physique ;
- un `ObservationAdapter` mappe ces obs sur le state vector du modèle
  (canaux + opérateur H) et renvoie des `Observation`.

Les solveurs d'assimilation consomment uniquement `Observation`.
"""

from .adapter import DEFAULT_MAPPING, ChannelSpec, ObservationAdapter
from .base import Observation, ObservationSource

__all__ = [
    "ChannelSpec",
    "DEFAULT_MAPPING",
    "Observation",
    "ObservationAdapter",
    "ObservationSource",
]
