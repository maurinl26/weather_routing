"""Tests des helpers du runner d'assimilation et des contrats de binding."""

import pandas as pd
import pytest

from wxrouting.assim.dps import DPSAssimilator
from wxrouting.assim.enkf import EnKFAssimilator
from wxrouting.assim.nudging import NudgingAssimilator
from wxrouting.cli.assimilate import _collect_observations, _normalize_iso_utc
from wxrouting.data.fetchers import BBox
from wxrouting.data.fetchers.base import Fetcher
from wxrouting.data.fetchers.synthetic_ais import SyntheticAISFetcher

BBOX = BBox(lat_min=43.0, lat_max=48.0, lon_min=-10.0, lon_max=-1.0)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-06-01T00:00:00Z", "2024-06-01T00:00:00"),   # 'Z' retiré
        ("2024-06-01T00:00:00", "2024-06-01T00:00:00"),    # déjà naïf
        ("2024-06-01T02:00:00+02:00", "2024-06-01T00:00:00"),  # converti en UTC
    ],
)
def test_normalize_iso_utc(raw, expected):
    assert _normalize_iso_utc(raw) == expected


class _BoomFetcher(Fetcher):
    name = "boom"

    def fetch(self, t0, t1, bbox):
        raise RuntimeError("auth failed")


def test_collect_observations_propagates_fetch_errors():
    # Une erreur de fetch (auth/config) ne doit PAS être avalée en repli silencieux.
    with pytest.raises(RuntimeError, match="auth failed"):
        _collect_observations([_BoomFetcher()], BBOX, "t0", "t1", adapter=None)


class _EmptyFetcher(Fetcher):
    name = "empty"

    def fetch(self, t0, t1, bbox):
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "variable", "value"])


def test_collect_observations_skips_empty_sources():
    # Une source sans donnée (DataFrame vide) est simplement ignorée.
    assert _collect_observations([_EmptyFetcher()], BBOX, "t0", "t1", adapter=None) == []


def test_requires_model_flags():
    assert NudgingAssimilator().requires_model is False
    assert EnKFAssimilator().requires_model is True
    assert DPSAssimilator().requires_model is True


class _ModelNoScore:
    pass


class _PlModule:
    def __init__(self, model):
        self.model = model


def test_dps_bind_model_raises_clear_error_when_model_lacks_api():
    assim = DPSAssimilator()
    with pytest.raises(AttributeError, match="score"):
        assim.bind_model(_PlModule(_ModelNoScore()))


def test_synthetic_fetcher_reference_binding():
    f = SyntheticAISFetcher()
    assert f.needs_reference is True
    assert f.reference_field is None
    sentinel = object()
    f.bind_reference(sentinel)
    assert f.reference_field is sentinel


def test_real_fetcher_does_not_need_reference():
    assert _EmptyFetcher().needs_reference is False
