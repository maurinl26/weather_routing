"""Fetcher EMODnet Human Activities — densités AIS historiques mensuelles.

Produit : `vesseldensity_*` (raster mensuel par type de navire).
Catalogue : https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search
URL OGC WCS pour download direct des GeoTIFF.

Pour le routage, intéressant comme **prior climatologique** sur la
distribution du trafic (pondération de l'opérateur H d'opportunité), pas
comme obs ponctuelle. C'est pour ça qu'on retourne une obs agrégée par
maille raster, pas par cible AIS individuelle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BBox, Fetcher, RawObsSchema

EMODNET_WCS = (
    "https://ows.emodnet-humanactivities.eu/wcs?"
    "service=WCS&version=2.0.1&request=GetCoverage"
)


class EMODnetAISFetcher(Fetcher):
    name = "emodnet_ais"

    def __init__(
        self,
        coverage_id: str = "emodnet:vesseldensity_all",
        cache_dir: str = "data/emodnet",
    ):
        self.coverage_id = coverage_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, year: int, month: int, bbox: BBox) -> Path:
        target = self.cache_dir / f"{self.coverage_id.replace(':','_')}_{year}-{month:02d}.tif"
        if target.exists():
            return target
        import requests

        params = {
            "coverageId": self.coverage_id,
            "format": "image/tiff",
            "subset": [
                f"Long({bbox.lon_min},{bbox.lon_max})",
                f"Lat({bbox.lat_min},{bbox.lat_max})",
                f"time(\"{year}-{month:02d}-01T00:00:00.000Z\")",
            ],
        }
        with requests.get(EMODNET_WCS, params=params, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return target

    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        import rasterio  # import paresseux

        start = pd.Timestamp(t0)
        end = pd.Timestamp(t1)
        parts: list[pd.DataFrame] = []
        cur = start.replace(day=1)
        while cur < end:
            path = self._download(cur.year, cur.month, bbox)
            with rasterio.open(path) as ds:
                arr = ds.read(1)
                rows, cols = arr.shape
                xs, ys = ds.xy(
                    [r for r in range(rows) for _ in range(cols)],
                    [c for _ in range(rows) for c in range(cols)],
                )
                df = pd.DataFrame(
                    {
                        "timestamp": pd.Timestamp(year=cur.year, month=cur.month, day=1, tz="UTC"),
                        "lat": ys, "lon": xs,
                        "variable": "vessel_density_hours_per_km2",
                        "value": arr.ravel(),
                    }
                )
                df = df[df["value"] > 0]
            parts.append(df)
            # Avance d'un mois
            cur = (cur + pd.offsets.MonthBegin(1)).normalize()
        if not parts:
            return RawObsSchema.empty()
        return RawObsSchema.validate(pd.concat(parts, ignore_index=True))
