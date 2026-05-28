"""Tests du module de routing v1 (isochrone, sans réseau)."""

import numpy as np
import pytest
import xarray as xr

from wxrouting.routing import (
    ConstantWindField,
    DPRouter,
    EnsembleWindField,
    GriddedWindField,
    IsochroneRouter,
    Polar,
    pareto_routes,
    robust_pareto_routes,
    route_ensemble,
)
from wxrouting.routing.geo import (
    angle_to_180,
    destination_point,
    haversine_km,
    initial_bearing_deg,
)


# --- géométrie ---------------------------------------------------------------

def test_destination_then_distance_roundtrip():
    lat2, lon2 = destination_point(45.0, -5.0, 90.0, 100.0)  # plein est, 100 km
    assert haversine_km(45.0, -5.0, lat2, lon2) == pytest.approx(100.0, abs=0.5)
    assert lat2 == pytest.approx(45.0, abs=0.2)   # cap est ⇒ latitude ~constante
    assert lon2 > -5.0                            # on est allé vers l'est


def test_initial_bearing_cardinal():
    assert initial_bearing_deg(45.0, 0.0, 46.0, 0.0) == pytest.approx(0.0, abs=1e-6)   # nord
    assert initial_bearing_deg(45.0, 0.0, 45.0, 1.0) == pytest.approx(90.0, abs=0.5)   # est


def test_angle_to_180():
    assert angle_to_180(350.0) == pytest.approx(10.0)
    assert angle_to_180(-170.0) == pytest.approx(170.0)
    assert angle_to_180(190.0) == pytest.approx(170.0)


# --- champ de vent -----------------------------------------------------------

def test_constant_field_tws_twd():
    # vent soufflant VERS l'est (u>0) ⇒ vient de l'ouest ⇒ TWD = 270°
    f = ConstantWindField(u_ms=10.0, v_ms=0.0)
    tws, twd = f.tws_twd(45.0, -5.0, np.datetime64("2024-06-01"))
    assert float(tws) == pytest.approx(10.0)
    assert float(twd) == pytest.approx(270.0)


def test_constant_field_from_south():
    # vent soufflant VERS le nord (v>0) ⇒ vient du sud ⇒ TWD = 180°
    f = ConstantWindField(u_ms=0.0, v_ms=5.0)
    _, twd = f.tws_twd(45.0, -5.0, np.datetime64("2024-06-01"))
    assert float(twd) == pytest.approx(180.0)


def test_gridded_field_interpolation():
    times = np.array(["2024-06-01T00", "2024-06-01T06"], dtype="datetime64[ns]")
    lat = np.array([46.0, 45.0, 44.0])
    lon = np.array([354.0, 355.0, 356.0])  # convention [0,360[
    shape = (2, 3, 3)
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), 3.0 * np.ones(shape)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), 4.0 * np.ones(shape)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    f = GriddedWindField(ds)
    # longitude négative -5° => 355° ; doit retrouver (u,v)=(3,4) ⇒ TWS=5
    tws, _ = f.tws_twd(45.0, -5.0, np.datetime64("2024-06-01T03"))
    assert float(tws) == pytest.approx(5.0)


# --- polaire -----------------------------------------------------------------

def test_polar_no_go_and_reach():
    p = Polar.example()
    assert float(p.boat_speed(15.0, 10.0)) == pytest.approx(0.0)  # no-go (TWA<30)
    beam = float(p.boat_speed(15.0, 90.0))
    upwind = float(p.boat_speed(15.0, 40.0))
    assert beam > upwind > 0.0                                    # le travers > le près
    assert float(p.boat_speed(0.0, 90.0)) == pytest.approx(0.0)   # pas de vent, pas de vitesse


def test_polar_from_csv(tmp_path):
    csv = tmp_path / "polar.csv"
    csv.write_text("twa/tws;60;90;120\n6;3.0;4.0;3.5\n12;5.0;6.5;6.0\n")
    p = Polar.from_csv(csv)
    assert p.tws.tolist() == [6.0, 12.0]
    assert p.twa.tolist() == [60.0, 90.0, 120.0]
    assert float(p.boat_speed(6.0, 90.0)) == pytest.approx(4.0)


# --- routeur isochrone -------------------------------------------------------

def test_router_reaches_destination_beam_reach():
    # Vent du sud (TWD 180), trajet plein est ⇒ travers ⇒ marche bien.
    field = ConstantWindField(u_ms=0.0, v_ms=6.0)
    router = IsochroneRouter(
        field, Polar.example(), step_hours=1.0, n_headings=19, max_steps=120
    )
    start, end = (45.0, -1.0), (45.0, 0.0)  # ~78 km
    route = router.solve(start, end, np.datetime64("2024-06-01T00:00:00"))

    assert route.reached
    assert route.points[0][:2] == pytest.approx(start)
    assert route.points[-1][:2] == pytest.approx(end)
    # la route great-circle fait ~78 km ; la route voilée est au moins aussi longue
    assert route.distance_km >= haversine_km(*start, *end) - 1.0
    assert 0.0 < route.duration_h < 24.0


def test_router_best_effort_when_unreachable():
    # Aucun vent ⇒ vitesse nulle partout ⇒ destination jamais atteinte.
    field = ConstantWindField(u_ms=0.0, v_ms=0.0)
    router = IsochroneRouter(field, Polar.example(), step_hours=1.0, max_steps=5)
    route = router.solve((45.0, -1.0), (45.0, 0.0), np.datetime64("2024-06-01T00:00:00"))
    assert route.reached is False


# --- ensemble ----------------------------------------------------------------

def test_ensemble_field_from_dataset():
    import xarray as xr

    times = np.array(["2024-06-01T00", "2024-06-01T06"], dtype="datetime64[ns]")
    lat = np.array([46.0, 45.0])
    lon = np.array([355.0, 356.0])
    shape = (3, 2, 2, 2)  # (member, time, lat, lon)
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("member", "time", "latitude", "longitude"), np.zeros(shape)),
            "10m_v_component_of_wind": (("member", "time", "latitude", "longitude"), 6.0 * np.ones(shape)),
        },
        coords={"member": [0, 1, 2], "time": times, "latitude": lat, "longitude": lon},
    )
    ens = EnsembleWindField.from_dataset(ds)
    assert ens.n_members == 3
    assert isinstance(ens.member(0), GriddedWindField)
    # un membre se comporte comme un champ scalaire normal
    _, twd = ens.member(0).tws_twd(45.0, -5.0, np.datetime64("2024-06-01T03"))
    assert float(twd) == pytest.approx(180.0)  # vent du sud


def test_ensemble_field_single_when_no_member_dim():
    import xarray as xr

    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.zeros((1, 2, 2))),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.ones((1, 2, 2))),
        },
        coords={
            "time": np.array(["2024-06-01"], dtype="datetime64[ns]"),
            "latitude": [46.0, 45.0],
            "longitude": [355.0, 356.0],
        },
    )
    assert EnsembleWindField.from_dataset(ds).n_members == 1


def test_route_ensemble_aggregates_members():
    # 3 membres, vent du sud de force croissante ⇒ durées décroissantes.
    ensemble = EnsembleWindField(
        [ConstantWindField(0.0, 5.0), ConstantWindField(0.0, 7.0), ConstantWindField(0.0, 9.0)]
    )
    res = route_ensemble(
        ensemble,
        Polar.example(),
        (45.0, -1.0),
        (45.0, 0.0),
        np.datetime64("2024-06-01T00:00:00"),
        step_hours=1.0,
        n_headings=19,
        max_steps=120,
    )
    assert len(res.member_routes) == 3
    assert res.reached_fraction == pytest.approx(1.0)
    assert set(res.duration_stats_h) == {"mean", "p10", "p50", "p90"}
    assert res.duration_stats_h["p10"] <= res.duration_stats_h["p90"]
    # le membre le plus venté (9 m/s) est le plus rapide
    durations = [r.duration_h for r in res.member_routes]
    assert durations[2] < durations[0]
    # la route recommandée est l'une des routes membres
    assert res.recommended in res.member_routes


# --- DP + Pareto -------------------------------------------------------------

def _patchy_field() -> GriddedWindField:
    """Vent de base d'ouest (12 kn, beam reach N-S) + patch fort (35 kn) à éviter."""
    import xarray as xr

    lat = np.arange(43.0, 47.0 + 1e-9, 0.5)
    lon_neg = np.arange(-6.0, -4.0 + 1e-9, 0.5)
    times = np.array(["2024-06-01T00", "2024-06-04T00"], dtype="datetime64[ns]")
    u = np.full((len(times), len(lat), len(lon_neg)), 6.0)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon_neg):
            if 44.5 <= la <= 45.5 and -5.5 <= lo <= -4.5:
                u[:, i, j] = 18.0  # ~35 kn > seuil 22 → risque
    v = np.zeros_like(u)
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), u),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), v),
        },
        coords={"time": times, "latitude": lat, "longitude": lon_neg % 360.0},
    )
    return GriddedWindField(ds)


def test_dp_reaches_and_zero_risk_below_threshold():
    field = ConstantWindField(u_ms=6.0, v_ms=0.0)  # 12 kn d'ouest, beam reach
    res = DPRouter(field, Polar.example(), grid_deg=0.5, max_hours=200).solve(
        (45.0, -5.0), (44.0, -5.0), np.datetime64("2024-06-01T00:00:00")
    )
    assert res.reached
    assert res.duration_h > 0
    assert res.risk == 0.0  # 12 kn < seuil 22 → aucune exposition


def test_pareto_front_time_vs_risk():
    field = _patchy_field()
    front = pareto_routes(
        field,
        Polar.example(),
        (46.0, -5.0),
        (44.0, -5.0),
        np.datetime64("2024-06-01T00:00:00"),
        risk_weights=(0.0, 5.0, 50.0),
        grid_deg=0.5,
        margin_deg=1.0,
        max_hours=200,
    )
    assert len(front) >= 2                       # un vrai compromis existe
    # trié par durée croissante : la plus rapide est la plus risquée
    assert front[0].duration_h <= front[-1].duration_h
    assert front[0].risk >= front[-1].risk
    assert front[0].risk > 0.0                   # la route rapide traverse la zone ventée
    assert front[-1].risk < front[0].risk        # une route plus sûre, plus longue, existe


# --- risque robuste sur l'ensemble -------------------------------------------

def _mild_field() -> GriddedWindField:
    """Vent d'ouest uniforme 12 kn (aucune zone ventée)."""
    import xarray as xr

    lat = np.arange(43.0, 47.0 + 1e-9, 0.5)
    lon_neg = np.arange(-6.0, -4.0 + 1e-9, 0.5)
    times = np.array(["2024-06-01T00", "2024-06-04T00"], dtype="datetime64[ns]")
    u = np.full((len(times), len(lat), len(lon_neg)), 6.0)
    v = np.zeros_like(u)
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), u),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), v),
        },
        coords={"time": times, "latitude": lat, "longitude": lon_neg % 360.0},
    )
    return GriddedWindField(ds)


def test_robust_pareto_rescores_risk_across_members():
    # Tempête présente dans 2 membres sur 3 ⇒ la route directe est risquée 2/3 du temps.
    ensemble = EnsembleWindField([_patchy_field(), _mild_field(), _patchy_field()])
    front = robust_pareto_routes(
        ensemble,
        Polar.example(),
        (46.0, -5.0),
        (44.0, -5.0),
        np.datetime64("2024-06-01T00:00:00"),
        risk_weights=(0.0, 5.0, 50.0),
        grid_deg=0.5,
        margin_deg=1.0,
        max_hours=200,
        tws_safe_kn=22.0,
    )
    assert len(front) >= 1
    fast = front[0]                                   # route la plus rapide
    assert len(fast.member_risks) == 3                # re-scorée sur chaque membre
    # la moyenne lisse la dispersion : le pire membre dépasse la moyenne
    assert fast.risk_max > fast.risk_mean > 0.0
    # la route directe rencontre la tempête dans 2 membres sur 3
    assert fast.rough_probability == pytest.approx(2 / 3)
