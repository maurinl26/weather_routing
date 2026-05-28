"""Tests du module de routing v1 (isochrone, sans réseau)."""

import numpy as np
import pytest
import xarray as xr

from wxrouting.routing import ConstantWindField, GriddedWindField, IsochroneRouter, Polar
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
