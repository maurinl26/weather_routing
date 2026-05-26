"""Sanity check : tous les modules importent (hors geoarches qui peut manquer)."""


def test_data_imports():
    from wxrouting.data import cf_names, crop, era5_arco, era5_cds, registry  # noqa: F401
    from wxrouting.data.obs import adapter, base, operators  # noqa: F401


def test_fetchers_imports():
    # Les fetchers réseau (cmems, ascat, emodnet, vos, aishub) importent
    # paresseusement leurs dépendances ; le module lui-même doit charger.
    from wxrouting.data.fetchers import (  # noqa: F401
        aishub,
        ascat,
        base,
        cmems_insitu,
        emodnet_ais,
        synthetic_ais,
        vos_gts,
    )


def test_assim_imports():
    from wxrouting.assim import base, dps, enkf, nudging  # noqa: F401


def test_registry_shape():
    from wxrouting.data.registry import StateVectorSpec
    spec = StateVectorSpec()
    assert spec.n_surface_channels == 4
    assert spec.n_level_channels == 5 * 13
    assert spec.n_channels == 4 + 65
