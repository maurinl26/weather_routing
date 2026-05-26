"""Adapter : DataFrame agnostique → list[Observation] correctes."""

import numpy as np
import pandas as pd
import torch

from wxrouting.data.cf_names import (
    AIR_PRESSURE_AT_SEA_LEVEL,
    EASTWARD_WIND_10M,
    NORTHWARD_WIND_10M,
    WIND_FROM_DIRECTION,
    WIND_SPEED_10M,
    canonical,
    decompose_wind,
)
from wxrouting.data.obs import ObservationAdapter


def _grid():
    lat = np.array([48.0, 47.0, 46.0, 45.0, 44.0])
    lon = np.array([350.0, 351.0, 352.0, 353.0, 354.0])
    return lat, lon


def test_canonical_aliases():
    assert canonical("WSPD") == WIND_SPEED_10M
    assert canonical("10m_u_component_of_wind") == EASTWARD_WIND_10M
    assert canonical("unknown_var") == "unknown_var"


def test_wind_decomposition():
    u, v = decompose_wind(np.array([10.0]), np.array([0.0]))  # vent du Nord pur
    assert np.allclose(u, 0.0, atol=1e-9)
    assert np.allclose(v, -10.0)


def test_adapter_decomposes_speed_dir_when_needed():
    lat, lon = _grid()
    adapter = ObservationAdapter(grid_lat=lat, grid_lon=lon)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"] * 2, utc=True),
            "lat": [46.5, 46.5],
            "lon": [-8.0, -8.0],
            "variable": [WIND_SPEED_10M, WIND_FROM_DIRECTION],
            "value": [10.0, 270.0],
        }
    )
    obs_list = adapter.adapt(df, t0="2024-01-01", source="test")
    var_names = {o.var_name for o in obs_list}
    assert EASTWARD_WIND_10M in var_names
    assert NORTHWARD_WIND_10M in var_names


def test_adapter_passthrough_pressure():
    lat, lon = _grid()
    adapter = ObservationAdapter(grid_lat=lat, grid_lon=lon)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "lat": [46.5], "lon": [-8.0],
            "variable": [AIR_PRESSURE_AT_SEA_LEVEL],
            "value": [101300.0],
        }
    )
    obs = adapter.adapt(df, t0="2024-01-01", source="test")
    assert len(obs) == 1 and obs[0].var_name == AIR_PRESSURE_AT_SEA_LEVEL
    # H différentiable
    x = torch.zeros(1, 4, 5, 5, requires_grad=True)
    obs[0].H(x).sum().backward()
    assert x.grad is not None


def test_adapter_ignores_unmapped_variable():
    lat, lon = _grid()
    adapter = ObservationAdapter(grid_lat=lat, grid_lon=lon)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "lat": [46.5], "lon": [-8.0],
            "variable": ["sea_surface_wave_significant_height"],
            "value": [3.2],
        }
    )
    assert adapter.adapt(df, t0="2024-01-01", source="test") == []
