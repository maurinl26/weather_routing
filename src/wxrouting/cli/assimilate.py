"""Entrypoint Hydra d'assimilation.

Pipeline :
    domain + window  ──▶  Fetchers  ──▶  DataFrame brut concaténé
                                              │
                                              ▼
                      datamodule.grid_*  ──▶  ObservationAdapter
                                              │
                                              ▼
                                          list[Observation]
                                              │
                                              ▼
                              x_b (background)  ──▶  Assimilator
                                              │
                                              ▼
                                          AssimResult

Usage :
    wxr-assim experiment=assim_nudging
    wxr-assim experiment=assim_enkf  checkpoint=path/to/ckpt fetchers=cmems_ascat
    wxr-assim experiment=assim_dps   window.t0=2024-06-15T00:00:00Z
"""

from __future__ import annotations

from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..data.crop import Domain
from ..data.fetchers import BBox, Fetcher
from ..data.obs import Observation, ObservationAdapter


def _build_bbox(domain: Domain) -> BBox:
    return BBox(
        lat_min=domain.lat_min, lat_max=domain.lat_max,
        lon_min=domain.lon_min, lon_max=domain.lon_max,
    )


def _instantiate_fetchers(cfg_fetchers: DictConfig, ref_field: Any | None) -> dict[str, Fetcher]:
    """Instancie les fetchers, en injectant le champ de référence aux générateurs
    synthétiques (qui en ont besoin pour produire des obs cohérentes avec ERA5)."""
    out: dict[str, Fetcher] = {}
    for name, sub in cfg_fetchers.items():
        f = instantiate(sub)
        if hasattr(f, "reference_field") and f.reference_field is None:
            f.reference_field = ref_field
        out[name] = f
    return out


def _collect_observations(
    fetchers: dict[str, Fetcher],
    bbox: BBox,
    t0: str,
    t1: str,
    adapter: ObservationAdapter,
) -> list[Observation]:
    """Fetch + adapt — renvoie une seule liste d'Observation, tagguées par source."""
    out: list[Observation] = []
    for name, fetcher in fetchers.items():
        try:
            df = fetcher.fetch(t0, t1, bbox)
        except Exception as e:
            print(f"[fetch] skip {name}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            print(f"[fetch] {name}: 0 obs")
            continue
        obs = adapter.adapt(df, t0=t0, source=name)
        print(f"[fetch] {name}: {len(df)} rows → {len(obs)} Observation groups")
        out.extend(obs)
    return out


def _inject_model_hooks(assim: Any, pl_module: Any) -> None:
    """Branche les sorties du modèle (sampler / score) sur les solveurs qui les
    consomment (EnKF, DPS). Pas d'impact sur le nudging."""
    if hasattr(assim, "sampler") and pl_module is not None:
        assim.sampler = pl_module.sample_ensemble
    if hasattr(assim, "score_fn") and pl_module is not None:
        assim.score_fn = getattr(pl_module.model, "score", None)
        assim.denoise_step = getattr(pl_module.model, "denoise_step", None)


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # 1. Datamodule → grille modèle + champ de référence (pour le synthétique).
    domain = Domain(**OmegaConf.to_container(cfg.domain, resolve=True))
    datamodule = instantiate(cfg.dataloader, domain=domain)
    datamodule.setup("test")

    bbox = _build_bbox(domain)
    adapter = ObservationAdapter(
        grid_lat=datamodule.grid_lat,
        grid_lon=datamodule.grid_lon,
    )

    # 2. Fetch + adapt.
    fetchers = _instantiate_fetchers(cfg.fetchers, ref_field=datamodule.reference_field)
    observations = _collect_observations(
        fetchers, bbox, t0=cfg.window.t0, t1=cfg.window.t1, adapter=adapter
    )
    if not observations:
        print("[assim] no observations collected — aborting")
        return

    # 3. Background : depuis le checkpoint si fourni, sinon depuis le datamodule.
    pl_module = None
    if cfg.get("checkpoint"):
        from ..finetune.lightning_module import ArchesGenFinetune
        pl_module = ArchesGenFinetune.load_from_checkpoint(cfg.checkpoint)
        pl_module.eval()
    batch = next(iter(datamodule.test_dataloader()))
    x_b = batch["x"]

    # 4. Solveur DA.
    assim = instantiate(cfg.assim, _convert_="all")
    _inject_model_hooks(assim, pl_module)

    with torch.no_grad():
        result = assim.assimilate(x_b, observations)

    print("=" * 60)
    print(f"x_a   : {tuple(result.x_a.shape)}")
    if result.x_a_ens is not None:
        print(f"x_a_ens: {tuple(result.x_a_ens.shape)}")
    for k, v in result.innov.items():
        v = v.detach().cpu()
        print(f"innov[{k:40s}] shape={tuple(v.shape)} "
              f"mean={float(v.mean()):+.4f} rms={float((v**2).mean().sqrt()):.4f}")


if __name__ == "__main__":
    main()
