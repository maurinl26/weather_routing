"""Cropping régional Golfe de Gascogne.

Conventions ERA5 :
  - latitude décroissante (90 → -90)
  - longitude dans [0, 360[
Les domaines configurés utilisent des longitudes signées ([-180, 180]).
"""

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class Domain:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    pad_pixels: int = 0


BAY_OF_BISCAY = Domain(
    name="bay_of_biscay",
    lat_min=43.0,
    lat_max=48.0,
    lon_min=-10.0,
    lon_max=-1.0,
    pad_pixels=8,
)


def _to_era5_longitude(lon: float) -> float:
    return lon % 360.0


def crop(ds: xr.Dataset, domain: Domain, resolution_deg: float = 0.25) -> xr.Dataset:
    """Crop un Dataset ERA5 sur le domaine donné, avec padding optionnel.

    Le padding est exprimé en pixels grille — il agrandit la fenêtre pour
    préserver le champ réceptif du transformer aux bords du domaine.
    """
    pad = domain.pad_pixels * resolution_deg
    lat_slice = slice(domain.lat_max + pad, domain.lat_min - pad)
    lon0 = _to_era5_longitude(domain.lon_min - pad)
    lon1 = _to_era5_longitude(domain.lon_max + pad)

    if lon0 < lon1:
        lon_slice = slice(lon0, lon1)
        return ds.sel(latitude=lat_slice, longitude=lon_slice)

    # Cas anti-méridien : on concatène les deux moitiés.
    left = ds.sel(latitude=lat_slice, longitude=slice(lon0, 360.0))
    right = ds.sel(latitude=lat_slice, longitude=slice(0.0, lon1))
    return xr.concat([left, right], dim="longitude")
