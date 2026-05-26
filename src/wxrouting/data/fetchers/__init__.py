"""Fetchers agnostiques du paramètre physique.

Un fetcher télécharge des observations brutes pour une période + une bbox
et renvoie un `pandas.DataFrame` normalisé. Il ne sait rien du modèle météo,
des canaux, ni des opérateurs H. C'est l'adapter (`obs/adapter.py`) qui prend
le relais.

Schéma de DataFrame :
    timestamp (datetime64), lat, lon, variable (CF std name), value,
    uncertainty (float, optionnel), platform_id (str, optionnel),
    + colonnes spécifiques source (extra metadata ; ex. hub_height_m)
"""

from .base import BBox, Fetcher, RawObsSchema

__all__ = ["BBox", "Fetcher", "RawObsSchema"]
