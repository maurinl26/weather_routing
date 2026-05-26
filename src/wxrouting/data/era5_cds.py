"""Téléchargement Copernicus CDS — pour compléter ARCO sur les mois récents
ou récupérer des variables absentes du bucket.

Requiert un fichier ~/.cdsapirc avec une clé API valide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from .crop import Domain, crop


class Era5CdsDownloader:
    def __init__(
        self,
        cache_dir: str,
        dataset: str,
        variables: dict[str, Any],
        years: list[int],
        months: list[int] | str = "all",
        domain: Domain | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.variables = variables
        self.years = years
        self.months = list(range(1, 13)) if months == "all" else list(months)
        self.domain = domain

    def _request(self, year: int, month: int) -> dict[str, Any]:
        d = self.domain
        # Aire CDS : [N, W, S, E]
        area = [d.lat_max, d.lon_min, d.lat_min, d.lon_max] if d else None
        req: dict[str, Any] = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": self.variables["surface"] + self.variables["level"],
            "pressure_level": [str(p) for p in self.variables["pressure_levels"]],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
        }
        if area is not None:
            req["area"] = area
        return req

    def download(self) -> list[Path]:
        """Télécharge tous les (year, month) demandés, renvoie la liste des fichiers."""
        import cdsapi  # import paresseux : cdsapi tire des deps réseau

        client = cdsapi.Client()
        out: list[Path] = []
        for y in self.years:
            for m in self.months:
                target = self.cache_dir / f"era5_{y}-{m:02d}.nc"
                if target.exists():
                    out.append(target)
                    continue
                client.retrieve(self.dataset, self._request(y, m), str(target))
                out.append(target)
        return out

    def open(self) -> xr.Dataset:
        """Ouvre tous les fichiers téléchargés en un seul Dataset."""
        files = sorted(self.cache_dir.glob("era5_*.nc"))
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 24})
        if self.domain is not None:
            ds = crop(ds, self.domain)
        return ds
