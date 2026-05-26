"""Sanity check du cropping régional."""

import numpy as np
import xarray as xr

from wxrouting.data.crop import BAY_OF_BISCAY, crop


def _fake_era5(resolution: float = 0.25) -> xr.Dataset:
    lat = np.arange(90.0, -90.0 - resolution, -resolution)
    lon = np.arange(0.0, 360.0, resolution)
    data = np.zeros((len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {"10m_u_component_of_wind": (("latitude", "longitude"), data)},
        coords={"latitude": lat, "longitude": lon},
    )


def test_crop_bay_of_biscay_no_pad():
    ds = _fake_era5()
    d = BAY_OF_BISCAY.__class__(**{**BAY_OF_BISCAY.__dict__, "pad_pixels": 0})
    out = crop(ds, d)
    assert out.latitude.min() >= 43.0 and out.latitude.max() <= 48.0
    # Conversion -10..-1 -> 350..359 dans le repère ERA5 [0, 360[.
    assert (out.longitude.min() >= 350.0) or (out.longitude.max() <= 360.0)


def test_crop_adds_padding():
    ds = _fake_era5()
    no_pad = crop(ds, BAY_OF_BISCAY.__class__(**{**BAY_OF_BISCAY.__dict__, "pad_pixels": 0}))
    padded = crop(ds, BAY_OF_BISCAY)
    assert padded.latitude.size > no_pad.latitude.size
    assert padded.longitude.size > no_pad.longitude.size
