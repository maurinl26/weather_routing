"""Tests unitaires du datamodule — sans réseau (on injecte un _full minimal)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wxrouting.data.era5_arco import Era5ArcoDataModule, _Era5WindowDataset


def _tiny_datamodule() -> Era5ArcoDataModule:
    """DataModule avec un champ ERA5 jouet couvrant [2024-06-01, 2024-06-02]."""
    times = pd.date_range("2024-06-01", "2024-06-02", freq="6h")
    lat = np.array([48.0, 47.0, 46.0])      # décroissante, comme ERA5
    lon = np.array([350.0, 351.0, 352.0])   # convention [0, 360[
    shape = (len(times), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.ones(shape)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), 2 * np.ones(shape)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    dm = Era5ArcoDataModule(
        zarr_url="unused",
        storage_options={},
        variables={
            "surface": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "level": [],
            "pressure_levels": [],
        },
        train_period=["2024-06-01", "2024-06-01"],
        val_period=["2024-06-01", "2024-06-01"],
        test_period=["2024-06-01", "2024-06-02"],
        lead_time_hours=6,
        sample_stride_hours=6,
        batch_size=1,
        num_workers=0,
    )
    dm._full = ds  # on court-circuite setup() (qui lit GCS)
    return dm


def test_reference_window_crops_to_window():
    dm = _tiny_datamodule()
    sub = dm.reference_window("2024-06-01T00:00:00", "2024-06-01T12:00:00")
    assert sub["time"].values.min() >= np.datetime64("2024-06-01T00:00:00")
    assert sub["time"].values.max() <= np.datetime64("2024-06-01T12:00:00")
    assert sub.sizes["time"] == 3  # 00, 06, 12


def test_reference_window_rejects_out_of_range():
    dm = _tiny_datamodule()
    with pytest.raises(ValueError, match="outside ERA5 coverage"):
        dm.reference_window("2024-06-01T00:00:00", "2025-01-01T00:00:00")


def test_background_state_shape_and_window_alignment():
    dm = _tiny_datamodule()
    x_b = dm.background_state("2024-06-01T00:00:00")
    # (B=1, C=2 surface channels, H=3, W=3)
    assert tuple(x_b.shape) == (1, 2, 3, 3)
    # canal v == 2 * canal u dans le champ jouet
    assert float(x_b[0, 0].mean()) == pytest.approx(1.0)
    assert float(x_b[0, 1].mean()) == pytest.approx(2.0)


def test_background_state_rejects_out_of_range():
    dm = _tiny_datamodule()
    with pytest.raises(ValueError, match="outside ERA5 coverage"):
        dm.background_state("2023-01-01T00:00:00")


# --- dataset fenêtré (chemin d'entraînement, sans réseau) --------------------

def test_window_dataset_shapes_and_lead_indexing():
    """Valide le windowing (x_t, x_{t+lead}) + le stacking en (C, H, W)."""
    dm = _tiny_datamodule()
    # 6 pas horaires couverts par la fenêtre test (2024-06-01 → 2024-06-02).
    times = pd.date_range("2024-06-01", periods=6, freq="1h")
    lat = np.array([46.0, 45.0, 44.0])
    lon = np.array([350.0, 351.0])
    shape = (len(times), len(lat), len(lon))
    dm._full = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.random.rand(*shape)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.random.rand(*shape)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds = dm._make(dm.test_period)
    assert isinstance(ds, _Era5WindowDataset)
    assert len(ds) >= 1
    sample = ds[0]
    assert set(sample) == {"x", "y"}
    # 2 canaux surface (spec sans niveaux), grille 3×2
    assert tuple(sample["x"].shape) == (2, 3, 2)
    assert tuple(sample["y"].shape) == (2, 3, 2)
    # x et y sont des pas de temps distincts (lead > 0)
    assert not np.allclose(sample["x"].numpy(), sample["y"].numpy())


def test_stack_state_channel_order_with_levels():
    """La lecture niveaux groupée préserve l'ordre var-major / niveau-mineur."""
    from wxrouting.data.era5_arco import _stack_state
    from wxrouting.data.registry import StateVectorSpec

    lat, lon, levels = [45.0, 44.0], [350.0, 351.0], [500, 850]
    snap = xr.Dataset(
        {
            "u10": (("latitude", "longitude"), np.full((2, 2), 1.0)),
            "t": (("level", "latitude", "longitude"),
                  np.stack([np.full((2, 2), 10.0), np.full((2, 2), 20.0)])),
        },
        coords={"latitude": lat, "longitude": lon, "level": levels},
    )
    spec = StateVectorSpec(surface=("u10",), level=("t",), pressure_levels=(500, 850))
    out = _stack_state(snap, spec)
    assert out.shape == (3, 2, 2)            # u10 + t@500 + t@850
    assert out[0].mean() == 1.0              # surface
    assert out[1].mean() == 10.0             # t @ 500 (1er niveau)
    assert out[2].mean() == 20.0             # t @ 850


def test_cache_in_memory_make_works():
    dm = _tiny_datamodule()
    dm.cache_in_memory = True
    times = pd.date_range("2024-06-01", periods=4, freq="1h")
    shape = (len(times), 2, 2)
    dm._full = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.random.rand(*shape)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.random.rand(*shape)),
        },
        coords={"time": times, "latitude": [45.0, 44.0], "longitude": [350.0, 351.0]},
    )
    ds = dm._make(dm.test_period)   # déclenche sub.load()
    assert len(ds) >= 1
    assert tuple(ds[0]["x"].shape) == (2, 2, 2)
