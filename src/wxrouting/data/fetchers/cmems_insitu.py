"""Fetcher CMEMS In-Situ TAC (bouées, mâts météo, navires marchands).

Utilise le SDK officiel `copernicusmarine` (auth via ~/.copernicusmarine ou
les vars d'env COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD).

Datasets pertinents pour le Golfe de Gascogne :
- INSITU_IBI_PHYBGCWAV_DISCRETE_MYNRT_013_033 (Iberian-Biscay-Ireland)
- INSITU_GLO_PHY_TS_DISCRETE_MY_013_001 (global historique)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from ..cf_names import (
    AIR_PRESSURE_AT_SEA_LEVEL,
    AIR_TEMPERATURE_2M,
    SEA_SURFACE_WAVE_HS,
    WIND_FROM_DIRECTION,
    WIND_SPEED_10M,
)
from .base import BBox, Fetcher, RawObsSchema

# Mapping nom de variable NetCDF CMEMS → nom CF canonique de ce projet.
CMEMS_VAR_MAP: dict[str, str] = {
    "WSPD": WIND_SPEED_10M,
    "WDIR": WIND_FROM_DIRECTION,
    "ATMS": AIR_PRESSURE_AT_SEA_LEVEL,
    "DRYT": AIR_TEMPERATURE_2M,
    "VHM0": SEA_SURFACE_WAVE_HS,
}


class CMEMSInSituFetcher(Fetcher):
    name = "cmems_insitu"

    def __init__(
        self,
        dataset_id: str = "cmems_obs-ins_ibi_phybgcwav_mynrt_na_irr",
        cache_dir: str = "data/cmems_insitu",
        variables: list[str] | None = None,
    ):
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.variables = variables or list(CMEMS_VAR_MAP.keys())

    # ------------------------------------------------------------------
    def _download(self, t0: str, t1: str, bbox: BBox) -> Path:
        """Appel copernicusmarine.subset(). Cache idempotent par signature."""
        import copernicusmarine as cm  # import paresseux (lib lourde)

        key = (
            f"{self.dataset_id}_{t0}_{t1}"
            f"_{bbox.lat_min}_{bbox.lat_max}_{bbox.lon_min}_{bbox.lon_max}"
        )
        target = self.cache_dir / (key.replace(":", "-") + ".nc")
        if target.exists():
            return target

        cm.subset(
            dataset_id=self.dataset_id,
            variables=self.variables,
            start_datetime=t0,
            end_datetime=t1,
            minimum_longitude=bbox.lon_min,
            maximum_longitude=bbox.lon_max,
            minimum_latitude=bbox.lat_min,
            maximum_latitude=bbox.lat_max,
            output_filename=target.name,
            output_directory=str(self.cache_dir),
        )
        return target

    # ------------------------------------------------------------------
    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame:
        path = self._download(t0, t1, bbox)
        ds = xr.open_dataset(path)

        rows: list[pd.DataFrame] = []
        for nc_var, cf_var in CMEMS_VAR_MAP.items():
            if nc_var not in ds:
                continue
            da = ds[nc_var]
            # La structure CMEMS in-situ est typiquement (TIME, DEPTH) avec
            # LATITUDE/LONGITUDE en coord scalaire par plateforme, ou
            # (PLATFORM, TIME). On aplatit en (sample, value).
            df = da.to_dataframe(name="value").reset_index()
            df["variable"] = cf_var
            df["timestamp"] = pd.to_datetime(df.get("TIME", df.get("time")), utc=True)
            df["lat"] = df.get("LATITUDE", df.get("latitude"))
            df["lon"] = df.get("LONGITUDE", df.get("longitude"))
            if "PLATFORM" in df:
                df["platform_id"] = df["PLATFORM"].astype(str)
            if "QC" in df:
                # 1 = bonne, 2 = probablement bonne — on garde les deux.
                df = df[df["QC"].isin([1, 2])]
            df = df.dropna(subset=["value", "lat", "lon", "timestamp"])
            rows.append(df[["timestamp", "lat", "lon", "variable", "value", "platform_id"]
                           if "platform_id" in df
                           else ["timestamp", "lat", "lon", "variable", "value"]])

        if not rows:
            return RawObsSchema.empty()
        return RawObsSchema.validate(pd.concat(rows, ignore_index=True))
