"""SyntheticAISFetcher : schéma conforme et end-to-end avec l'adapter."""

import numpy as np
import pandas as pd

from wxrouting.data.cf_names import EASTWARD_WIND_10M, NORTHWARD_WIND_10M
from wxrouting.data.fetchers import BBox
from wxrouting.data.fetchers.synthetic_ais import SyntheticAISFetcher
from wxrouting.data.obs import ObservationAdapter


def test_synthetic_fetcher_schema():
    f = SyntheticAISFetcher(n_vessels=3, n_points_per_track=4, seed=0)
    bbox = BBox(lat_min=43, lat_max=48, lon_min=-10, lon_max=-1)
    df = f.fetch("2024-01-01", "2024-01-02", bbox)
    for col in ("timestamp", "lat", "lon", "variable", "value"):
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # 3 voiliers × 4 points × 2 composantes (u, v)
    assert len(df) == 3 * 4 * 2


def test_synthetic_to_adapter_end_to_end():
    f = SyntheticAISFetcher(n_vessels=2, n_points_per_track=3, seed=1)
    bbox = BBox(lat_min=43, lat_max=48, lon_min=-10, lon_max=-1)
    df = f.fetch("2024-01-01", "2024-01-02", bbox)

    # Grille minimale couvrant la bbox (lon ERA5 ∈ [0, 360[)
    grid_lat = np.linspace(48, 43, 21)
    grid_lon = np.linspace(350, 359, 37)

    adapter = ObservationAdapter(grid_lat=grid_lat, grid_lon=grid_lon)
    obs = adapter.adapt(df, t0="2024-01-01", source=f.name)
    var_names = {o.var_name for o in obs}
    assert var_names == {EASTWARD_WIND_10M, NORTHWARD_WIND_10M}
    assert all(len(o) == 2 * 3 for o in obs)  # 2 voiliers × 3 points
