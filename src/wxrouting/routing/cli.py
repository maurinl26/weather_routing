"""Entrypoint Hydra de routage `[PoC routage]` — wxr-route.

v1 : construit un champ de vent depuis le datamodule (ERA5 sur la fenêtre, en
attendant un checkpoint ArchesWeatherGen pour une vraie prévision), puis route
entre deux points avec une polaire voilier.

Usage :
    wxr-route route.start=[48.36,-4.49] route.end=[43.37,-8.40]
    wxr-route window.t0=2024-06-01T00:00:00Z route.step_hours=1
"""

from __future__ import annotations

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..data.crop import Domain
from .field import GriddedWindField
from .isochrone import IsochroneRouter
from .polar import Polar


def _normalize_iso_utc(ts: str) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.isoformat()


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg.route))

    domain = Domain(**OmegaConf.to_container(cfg.domain, resolve=True))
    datamodule = instantiate(cfg.dataloader, domain=domain)
    datamodule.setup("test")

    t0 = _normalize_iso_utc(cfg.window.t0)
    t1 = _normalize_iso_utc(cfg.window.t1)
    field = GriddedWindField(datamodule.reference_window(t0, t1))

    polar = Polar.from_csv(cfg.route.polar_csv) if cfg.route.get("polar_csv") else Polar.example()

    router = IsochroneRouter(
        field,
        polar,
        step_hours=cfg.route.step_hours,
        n_headings=cfg.route.n_headings,
        heading_spread_deg=cfg.route.heading_spread_deg,
    )
    start = tuple(cfg.route.start)
    end = tuple(cfg.route.end)
    route = router.solve(start, end, np.datetime64(t0))

    print("=" * 60)
    status = "reached" if route.reached else "NOT reached (best effort)"
    print(f"route {start} -> {end} : {status}")
    print(f"distance : {route.distance_km:.0f} km   durée : {route.duration_h:.1f} h "
          f"({len(route.points)} waypoints)")
    for lat, lon, t in route.points:
        print(f"  {str(t)[:16]}  {lat:7.3f}, {lon:8.3f}")


if __name__ == "__main__":
    main()
