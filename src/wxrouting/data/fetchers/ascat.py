"""Fetcher ASCAT (Metop) — vent vecteur 10 m océan.

⚠️ ASCAT L2 n'est PAS réellement anonyme : tous les distributeurs demandent une
inscription (gratuite). Backends :
- **`podaac`** (recommandé) : PO.DAAC / NASA Earthdata via la lib `earthaccess`.
  Compte Earthdata gratuit (env EARTHDATA_USERNAME/PASSWORD ou ~/.netrc).
- **`eumdac`** : EUMETSAT Data Store (credentials EUMETSAT).
- **`knmi`** : lecture des fichiers L2 déjà présents en cache (pas de download —
  stub historique conservé pour les TP/CI hors-ligne).

Produits : L2 25 km (`ascat_25_l2`) ou côtier 12.5 km (`ascat_coastal_l2`).
Le parsing (`_parse_l2`) suit la convention NetCDF KNMI/OSI SAF (= celle de PO.DAAC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..cf_names import WIND_FROM_DIRECTION, WIND_SPEED_10M
from .base import BBox, Fetcher, RawObsSchema


class ASCATFetcher(Fetcher):
    name = "ascat"

    def __init__(
        self,
        backend: Literal["podaac", "eumdac", "knmi"] = "podaac",
        cache_dir: str = "data/ascat",
        platform: str = "metop_b",
        product: str = "ascat_25_l2",
        eumdac_credentials: tuple[str, str] | None = None,
    ):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.platform = platform
        self.product = product
        self.eumdac_credentials = eumdac_credentials

    # ------------------------------------------------------------------
    def _list_files_eumdac(self, t0: str, t1: str, bbox: BBox) -> list[Path]:
        import eumdac  # type: ignore

        assert self.eumdac_credentials is not None, (
            "EUMDAC requires (consumer_key, consumer_secret) credentials"
        )
        token = eumdac.AccessToken(self.eumdac_credentials)
        store = eumdac.DataStore(token)
        col = store.get_collection(self._eumdac_collection_id())
        products = col.search(
            dtstart=t0, dtend=t1, bbox=[bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max]
        )
        out: list[Path] = []
        for p in products:
            target = self.cache_dir / p.filename
            if not target.exists():
                with p.open() as src, open(target, "wb") as dst:
                    dst.write(src.read())
            out.append(target)
        return out

    def _eumdac_collection_id(self) -> str:
        # ASCAT-A/B/C L2 25 km wind product collections (EUMETSAT Data Store).
        mapping = {
            ("metop_b", "ascat_25_l2"): "EO:EUM:DAT:METOP:OAS025",
            ("metop_c", "ascat_25_l2"): "EO:EUM:DAT:0581",
        }
        return mapping[(self.platform, self.product)]

    # ------------------------------------------------------------------
    def _podaac_short_name(self) -> str:
        """short_name PO.DAAC selon plateforme + produit (ex. ASCATB-L2-Coastal)."""
        plat = {"metop_a": "A", "metop_b": "B", "metop_c": "C"}[self.platform]
        kind = "Coastal" if "coastal" in self.product else "25km"
        return f"ASCAT{plat}-L2-{kind}"

    def _list_files_podaac(self, t0: str, t1: str, bbox: BBox) -> list[Path]:
        """Recherche + téléchargement via NASA Earthdata (lib earthaccess).

        Auth : EARTHDATA_USERNAME/PASSWORD ou ~/.netrc (compte gratuit).
        """
        try:
            import earthaccess  # import paresseux
        except ImportError as e:
            raise RuntimeError(
                "Backend podaac : `pip install earthaccess` + compte NASA Earthdata."
            ) from e

        earthaccess.login()  # stratégie auto : env vars puis ~/.netrc
        results = earthaccess.search_data(
            short_name=self._podaac_short_name(),
            temporal=(t0, t1),
            bounding_box=(bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max),
        )
        if not results:
            return []
        paths = earthaccess.download(results, local_path=str(self.cache_dir))
        return [Path(p) for p in paths]

    # ------------------------------------------------------------------
    def _list_files_knmi(self, t0: str, t1: str) -> list[Path]:
        """Téléchargement HTTP des fichiers L2 KNMI.

        L'arborescence KNMI est `/seawinds/ascat/<platform>/.../<YYYY>/<MM>/<DD>/`.
        On utilise une convention simplifiée : à connecter au catalogue OPenDAP
        réel lors de l'intégration. Ici on stub : si rien en cache, le fetch
        renvoie un DataFrame vide (utile en CI / TP sans accès réseau).
        """
        files = sorted(self.cache_dir.glob(f"ascat_*{self.platform}*.nc"))
        if not files:
            # Pas de catalogue HTTP exploré ici — à remplacer par un vrai
            # listing OPenDAP. Voir notebooks/ pour un exemple de fetch
            # manuel via wget/curl.
            return []
        return files

    # ------------------------------------------------------------------
    def _parse_l2(self, path: Path, bbox: BBox) -> pd.DataFrame:
        ds = xr.open_dataset(path)
        lat = ds["lat"].values.ravel()
        lon = ds["lon"].values.ravel()
        wspd = ds["wind_speed"].values.ravel()
        wdir = ds["wind_dir"].values.ravel()
        time = ds["time"].broadcast_like(ds["wind_speed"]).values.ravel()

        mask = (
            np.isfinite(wspd) & np.isfinite(wdir)
            & (lat >= bbox.lat_min) & (lat <= bbox.lat_max)
            & (((lon >= bbox.lon_min) & (lon <= bbox.lon_max))
               | (((lon - 360) >= bbox.lon_min) & ((lon - 360) <= bbox.lon_max)))
        )
        if "wvc_quality_flag" in ds:
            mask &= ds["wvc_quality_flag"].values.ravel() == 0
        if not mask.any():
            return RawObsSchema.empty()

        wspd, wdir = wspd[mask], wdir[mask]
        lat, lon, time = lat[mask], lon[mask], time[mask]

        df_speed = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(time, utc=True),
                "lat": lat, "lon": lon,
                "variable": WIND_SPEED_10M, "value": wspd,
                "uncertainty": np.full_like(wspd, 1.5),
            }
        )
        df_dir = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(time, utc=True),
                "lat": lat, "lon": lon,
                "variable": WIND_FROM_DIRECTION, "value": wdir,
                "uncertainty": np.full_like(wdir, 20.0),
            }
        )
        return pd.concat([df_speed, df_dir], ignore_index=True)

    # ------------------------------------------------------------------
    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        if self.backend == "podaac":
            files = self._list_files_podaac(t0, t1, bbox)
        elif self.backend == "eumdac":
            files = self._list_files_eumdac(t0, t1, bbox)
        else:
            files = self._list_files_knmi(t0, t1)

        if not files:
            return RawObsSchema.empty()

        parts = [self._parse_l2(f, bbox) for f in files]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return RawObsSchema.empty()
        df = pd.concat(parts, ignore_index=True)
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
        return RawObsSchema.validate(df)
