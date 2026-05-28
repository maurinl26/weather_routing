"""DataModule Lightning sur ARCO-ERA5 (streaming Zarr depuis GCS).

ARCO-ERA5 = "Analysis-Ready, Cloud-Optimized" — pas de téléchargement, on lit
des chunks au vol via fsspec/gcsfs. Le cropping régional réduit ~3000x le
volume effectif transféré.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .crop import Domain, crop
from .registry import StateVectorSpec


def _stack_state(snap: xr.Dataset, spec: StateVectorSpec) -> np.ndarray:
    """Stack surface + (var × level) en (C, H, W) — ordre figé par le registry."""
    surf = [snap[v].values for v in spec.surface]
    lvl = [
        snap[v].sel(level=p).values
        for v in spec.level
        for p in spec.pressure_levels
    ]
    return np.stack(surf + lvl, axis=0).astype(np.float32)


class _Era5WindowDataset(Dataset):
    """Renvoie des couples (x_t, x_{t+lead}) — état d'entrée et cible."""

    def __init__(
        self,
        ds: xr.Dataset,
        spec: StateVectorSpec,
        lead_time_hours: int,
        sample_stride_hours: int,
    ):
        self.ds = ds
        self.spec = spec
        self.lead = lead_time_hours
        # Index des pas de temps valides (t et t+lead doivent exister).
        times = ds["time"].values
        stride = sample_stride_hours // int(
            (times[1] - times[0]) / np.timedelta64(1, "h")
        )
        max_i = len(times) - (lead_time_hours // sample_stride_hours)
        self.indices = list(range(0, max_i, max(stride, 1)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.indices[idx]
        x = _stack_state(self.ds.isel(time=i), self.spec)
        y = _stack_state(self.ds.isel(time=i + self.lead // 6), self.spec)  # ARCO est horaire
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


class Era5ArcoDataModule(L.LightningDataModule):
    def __init__(
        self,
        zarr_url: str,
        storage_options: dict[str, Any],
        variables: dict[str, list],
        train_period: list[str],
        val_period: list[str],
        test_period: list[str],
        lead_time_hours: int,
        sample_stride_hours: int,
        batch_size: int,
        num_workers: int,
        domain: Domain | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["domain"])
        self.zarr_url = zarr_url
        self.storage_options = storage_options
        self.spec = StateVectorSpec(
            surface=tuple(variables["surface"]),
            level=tuple(variables["level"]),
            pressure_levels=tuple(variables["pressure_levels"]),
        )
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period
        self.lead_time_hours = lead_time_hours
        self.sample_stride_hours = sample_stride_hours
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.domain = domain  # injecté par le runner (depuis cfg.domain)
        self._full: xr.Dataset | None = None

    def prepare_data(self) -> None:
        # Rien à télécharger — lecture en streaming.
        pass

    def setup(self, stage: str | None = None) -> None:
        ds = xr.open_zarr(
            self.zarr_url, storage_options=self.storage_options, consolidated=True
        )
        # Garde uniquement les variables du state vector.
        keep = list(self.spec.surface) + list(self.spec.level)
        ds = ds[keep]
        if self.domain is not None:
            ds = crop(ds, self.domain)
        self._full = ds

    # ------------------------------------------------------------------
    # Grille modèle — exposée à l'adapter pour construire les opérateurs H.
    # ------------------------------------------------------------------
    @property
    def grid_lat(self) -> np.ndarray:
        assert self._full is not None, "call setup() first"
        return self._full["latitude"].values

    @property
    def grid_lon(self) -> np.ndarray:
        assert self._full is not None, "call setup() first"
        return self._full["longitude"].values

    def _check_coverage(self, *stamps: np.datetime64) -> None:
        """Lève si un instant sort de la couverture temporelle ERA5 — sinon
        l'interpolation renvoie des NaN silencieux qui corrompent l'analyse."""
        assert self._full is not None, "call setup() first"
        times = self._full["time"].values
        for t in stamps:
            if t < times[0] or t > times[-1]:
                raise ValueError(
                    f"time {t} outside ERA5 coverage [{times[0]}, {times[-1]}]"
                )

    def reference_window(self, t0: str, t1: str) -> xr.Dataset:
        """Champ de référence (vérité du générateur synthétique) borné à [t0, t1].

        Cropé dans le temps pour que l'interpolation des fetchers ne balaie que
        la fenêtre d'assimilation, et validé contre la couverture ERA5.
        """
        assert self._full is not None, "call setup() first"
        t0n, t1n = np.datetime64(t0), np.datetime64(t1)
        self._check_coverage(t0n, t1n)
        return self._full.sel(time=slice(t0, t1))

    def background_state(self, t: str) -> torch.Tensor:
        """État de fond x_b (B=1, C, H, W) à l'instant `t` — aligné sur la fenêtre.

        Sans ce branchement, le fond serait pris au premier pas du test loader
        (~17 mois avant la fenêtre) et les innovations seraient dénuées de sens.
        """
        assert self._full is not None, "call setup() first"
        tn = np.datetime64(t)
        self._check_coverage(tn)
        snap = self._full.sel(time=tn, method="nearest")
        return torch.from_numpy(_stack_state(snap, self.spec)).unsqueeze(0)

    def _make(self, period: list[str]) -> _Era5WindowDataset:
        assert self._full is not None, "setup() must be called first"
        sub = self._full.sel(time=slice(period[0], period[1]))
        return _Era5WindowDataset(
            sub, self.spec, self.lead_time_hours, self.sample_stride_hours
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make(self.train_period),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make(self.val_period),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make(self.test_period),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
