"""Tests des fetchers d'observation testables hors-ligne.

ASCAT : le téléchargement KNMI est un stub (pas de réseau), mais le **parsing L2**
est la vraie valeur — on le valide sur un fichier NetCDF synthétique au format
swath ASCAT. CMEMS : SDK copernicusmarine requis (compte gratuit) — non testable
ici, on vérifie juste l'import paresseux.
"""

import numpy as np
import xarray as xr

from wxrouting.data.cf_names import WIND_FROM_DIRECTION, WIND_SPEED_10M
from wxrouting.data.fetchers import BBox
from wxrouting.data.fetchers.ascat import ASCATFetcher
from wxrouting.data.fetchers.cmems_insitu import CMEMSInSituFetcher

BBOX = BBox(lat_min=43.0, lat_max=48.0, lon_min=-10.0, lon_max=-1.0)


def _synthetic_l2(path, *, platform_in_bbox=True):
    """Fichier L2 ASCAT jouet : swath (rows, cells) avec lat/lon/vent/temps."""
    r, c = 3, 4
    lat = (np.linspace(44.0, 47.0, r)[:, None] * np.ones((r, c)))
    lon = (np.linspace(-8.0, -3.0, c)[None, :] * np.ones((r, c)))
    if not platform_in_bbox:
        lat += 20.0  # hors bbox
    wspd = np.full((r, c), 9.0)
    wspd[0, 0] = np.nan  # une obs invalide → doit être filtrée
    wdir = np.full((r, c), 270.0)
    time = np.array(
        ["2024-06-01T10:00", "2024-06-01T10:01", "2024-06-01T10:02"],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {
            "wind_speed": (("rows", "cells"), wspd),
            "wind_dir": (("rows", "cells"), wdir),
        },
        coords={
            "lat": (("rows", "cells"), lat),
            "lon": (("rows", "cells"), lon),
            "time": (("rows",), time),
        },
    )
    ds.to_netcdf(path)


def test_ascat_parse_l2_maps_and_filters(tmp_path):
    f = tmp_path / "ascat_metop_b_l2.nc"
    _synthetic_l2(f)
    df = ASCATFetcher(backend="knmi")._parse_l2(f, BBOX)
    assert not df.empty
    assert set(df["variable"]) == {WIND_SPEED_10M, WIND_FROM_DIRECTION}
    # 12 cellules - 1 NaN = 11 obs valides, ×2 variables
    assert (df["variable"] == WIND_SPEED_10M).sum() == 11
    assert (df["variable"] == WIND_FROM_DIRECTION).sum() == 11
    assert df["lat"].between(43, 48).all()


def test_ascat_knmi_fetch_with_cached_file(tmp_path):
    # Le backend KNMI lit les .nc présents en cache → valide le chemin complet
    # une fois un fichier disponible (seul le listing/download est un stub).
    _synthetic_l2(tmp_path / "ascat_20240601_metop_b.nc")
    f = ASCATFetcher(backend="knmi", cache_dir=str(tmp_path))
    df = f.fetch("2024-06-01", "2024-06-02", BBOX)
    assert not df.empty
    assert {"timestamp", "lat", "lon", "variable", "value"} <= set(df.columns)


def test_ascat_knmi_empty_without_cache(tmp_path):
    f = ASCATFetcher(backend="knmi", cache_dir=str(tmp_path))
    assert f.fetch("2024-06-01", "2024-06-02", BBOX).empty


def test_cmems_fetcher_constructs_without_sdk():
    # L'import copernicusmarine est paresseux → instanciation OK sans le SDK.
    f = CMEMSInSituFetcher()
    assert f.name == "cmems_insitu"
    assert f.needs_reference is False
