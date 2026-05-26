"""Sanity check : tous les modules importent (hors geoarches qui peut manquer)."""


def test_data_imports():
    from wxrouting.data import crop, era5_arco, era5_cds, registry  # noqa: F401
    from wxrouting.data.obs import ais, ascat, base, buoys, operators, windfarm  # noqa: F401


def test_assim_imports():
    from wxrouting.assim import base, dps, enkf, nudging  # noqa: F401


def test_registry_shape():
    from wxrouting.data.registry import StateVectorSpec
    spec = StateVectorSpec()
    assert spec.n_surface_channels == 4
    assert spec.n_level_channels == 5 * 13
    assert spec.n_channels == 4 + 65
