"""Registre CF Standard Names + alias.

Les fetchers utilisent les noms CF canoniques (cf-conventions.org). Les
adapters utilisent ce registre pour résoudre les alias source-spécifiques
(WSPD, wind_speed_10m, u10, …) vers la variable canonique, et pour mapper
les variables canoniques vers les canaux du state vector ArchesWeatherGen.
"""

from __future__ import annotations

# Noms canoniques (sous-ensemble pertinent pour le routage maritime).
EASTWARD_WIND_10M = "eastward_wind_at_10m"
NORTHWARD_WIND_10M = "northward_wind_at_10m"
WIND_SPEED_10M = "wind_speed_at_10m"
WIND_FROM_DIRECTION = "wind_from_direction"
AIR_PRESSURE_AT_SEA_LEVEL = "air_pressure_at_sea_level"
AIR_TEMPERATURE_2M = "air_temperature_at_2m"
SEA_SURFACE_WAVE_HS = "sea_surface_wave_significant_height"
WIND_SPEED_AT_HUB = "wind_speed_at_hub_height"

# Alias rencontrés dans les sources → nom canonique.
ALIASES: dict[str, str] = {
    # CMEMS / Météo-France in-situ
    "WSPD": WIND_SPEED_10M,
    "WDIR": WIND_FROM_DIRECTION,
    "ATMS": AIR_PRESSURE_AT_SEA_LEVEL,
    "ATPT": AIR_PRESSURE_AT_SEA_LEVEL,
    "DRYT": AIR_TEMPERATURE_2M,
    "VHM0": SEA_SURFACE_WAVE_HS,
    # ASCAT / KNMI
    "wind_speed": WIND_SPEED_10M,
    "wind_dir": WIND_FROM_DIRECTION,
    "eastward_wind": EASTWARD_WIND_10M,
    "northward_wind": NORTHWARD_WIND_10M,
    # ERA5
    "10m_u_component_of_wind": EASTWARD_WIND_10M,
    "10m_v_component_of_wind": NORTHWARD_WIND_10M,
    "mean_sea_level_pressure": AIR_PRESSURE_AT_SEA_LEVEL,
    "2m_temperature": AIR_TEMPERATURE_2M,
    # VOS (codes SHIP/BBXX)
    "dd": WIND_FROM_DIRECTION,
    "ff": WIND_SPEED_10M,
    "P0": AIR_PRESSURE_AT_SEA_LEVEL,
    # Garmin / NMEA0183 (TWD/TWS = true wind direction/speed)
    "TWS": WIND_SPEED_10M,
    "TWD": WIND_FROM_DIRECTION,
}


def canonical(name: str) -> str:
    """Renvoie le nom CF canonique pour un alias quelconque (passthrough sinon)."""
    return ALIASES.get(name, name)


def decompose_wind(speed: float, direction_deg: float) -> tuple[float, float]:
    """(speed, dir météo) → (eastward, northward).

    Convention météo : `direction_deg` = direction d'où vient le vent, 0° = Nord.
    """
    import numpy as np

    rad = np.deg2rad(direction_deg)
    return (-speed * np.sin(rad), -speed * np.cos(rad))
