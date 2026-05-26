"""Observations d'opportunité — interface unique pour l'assimilation.

Sources : AIS/voiliers, fermes éoliennes offshore, bouées Météo-France/CMEMS,
scattéromètre ASCAT. Toutes implémentent `ObservationSource`.
"""

from .base import Observation, ObservationSource

__all__ = ["Observation", "ObservationSource"]
