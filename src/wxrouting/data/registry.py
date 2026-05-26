"""Registry des variables ERA5 attendues par ArchesWeatherGen.

L'ordre est figé — il définit l'index des canaux du state vector du modèle
pré-entraîné. Toute modification casse les poids HF.
"""

from dataclasses import dataclass

SURFACE_VARS: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)

LEVEL_VARS: tuple[str, ...] = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)

PRESSURE_LEVELS: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
)


@dataclass(frozen=True)
class StateVectorSpec:
    surface: tuple[str, ...] = SURFACE_VARS
    level: tuple[str, ...] = LEVEL_VARS
    pressure_levels: tuple[int, ...] = PRESSURE_LEVELS

    @property
    def n_surface_channels(self) -> int:
        return len(self.surface)

    @property
    def n_level_channels(self) -> int:
        return len(self.level) * len(self.pressure_levels)

    @property
    def n_channels(self) -> int:
        return self.n_surface_channels + self.n_level_channels


# Alias court pour les diagnostics de routage (vent surface uniquement).
ROUTING_VARS: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
