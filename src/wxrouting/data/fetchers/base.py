"""Contrat Fetcher — agnostique du paramètre physique."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BBox:
    """Bounding box géographique (degrés)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Colonnes obligatoires du DataFrame renvoyé par tout Fetcher.
# Les fetchers PEUVENT ajouter des colonnes (ex. "hub_height_m",
# "platform_id", "quality_flag") — l'adapter sait les ignorer
# ou les exploiter selon le mapping configuré.
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "lat", "lon", "variable", "value")
OPTIONAL_COLUMNS: tuple[str, ...] = ("uncertainty", "platform_id", "quality_flag")


class RawObsSchema:
    """Helpers de validation et de normalisation."""

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    @staticmethod
    def empty() -> pd.DataFrame:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS))


class Fetcher(ABC):
    """Source d'observations brutes.

    Une instance représente UNE source physique (CMEMS in-situ, ASCAT, AIS…).
    Elle est instanciable à partir de credentials/URL ; `fetch()` n'est appelé
    qu'au moment de la collecte.
    """

    name: str  # identifiant unique de la source

    @abstractmethod
    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        """Renvoie un DataFrame conforme à `RawObsSchema` sur [t0, t1[ ∩ bbox.

        Le DataFrame peut être vide (= aucune obs disponible) mais doit
        respecter le schéma. `t0` et `t1` sont des chaînes ISO 8601 UTC.
        """
        ...

    def __repr__(self) -> str:
        return f"<Fetcher {self.name}>"
