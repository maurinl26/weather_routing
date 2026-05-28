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
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..data.crop import Domain
from ..data.fetchers import BBox, Fetcher
from ..data.obs import Observation, ObservationAdapter


def _normalize_iso_utc(ts: str) -> str:
    """ISO 8601 (avec ou sans 'Z') → chaîne UTC naïve sans suffixe de fuseau.

    np.datetime64 ne sait pas représenter les fuseaux : un 'Z' final n'est que
    silencieusement ignoré aujourd'hui et lèvera dans une future version de
    numpy. On normalise au seuil du CLI pour que fetchers et adapter reçoivent
    une chaîne propre.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.isoformat()


def _build_bbox(domain: Domain) -> BBox:
    return BBox(
        lat_min=domain.lat_min, lat_max=domain.lat_max,
        lon_min=domain.lon_min, lon_max=domain.lon_max,
    )


def _instantiate_fetchers(cfg_fetchers: DictConfig, ref_field: Any | None) -> list[Fetcher]:
    """Instancie les fetchers, en injectant le champ de référence à ceux qui le
    déclarent (générateurs synthétiques) — cf. Fetcher.needs_reference."""
    out: list[Fetcher] = []
    for sub in cfg_fetchers.values():
        f = instantiate(sub)
        if f.needs_reference:
            f.bind_reference(ref_field)
        out.append(f)
    return out


def _collect_observations(
    fetchers: list[Fetcher],
    bbox: BBox,
    t0: str,
    t1: str,
    adapter: ObservationAdapter,
) -> list[Observation]:
    """Fetch + adapt — renvoie une seule liste d'Observation, tagguées par source.

    Une source sans donnée sur la fenêtre renvoie un DataFrame vide (cf. contrat
    Fetcher) et est ignorée ; toute exception (auth, schéma, réseau) remonte —
    on ne masque pas une mauvaise config en repli silencieux sur moins d'obs.
    """
    out: list[Observation] = []
    for fetcher in fetchers:
        df = fetcher.fetch(t0, t1, bbox)
        if df.empty:
            print(f"[fetch] {fetcher.name}: 0 obs")
            continue
        obs = adapter.adapt(df, t0=t0, source=fetcher.name)
        print(f"[fetch] {fetcher.name}: {len(df)} rows → {len(obs)} Observation groups")
        out.extend(obs)
    return out


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

    t0 = _normalize_iso_utc(cfg.window.t0)
    t1 = _normalize_iso_utc(cfg.window.t1)

    # checkpoint=??? (enkf/dps) lève ici si non fourni ; null (nudging) → pas de
    # modèle. Résolu avant le fetch pour échouer tôt, sans gâcher la collecte.
    checkpoint = OmegaConf.select(cfg, "checkpoint", throw_on_missing=True)

    # 2. Fetch + adapt — le champ de référence est borné à la fenêtre.
    fetchers = _instantiate_fetchers(
        cfg.fetchers, ref_field=datamodule.reference_window(t0, t1)
    )
    observations = _collect_observations(fetchers, bbox, t0=t0, t1=t1, adapter=adapter)
    if not observations:
        print("[assim] no observations collected — aborting")
        return

    # 3. Background aligné sur le début de fenêtre (et non le 1er pas du test set).
    pl_module = None
    if checkpoint:
        from ..finetune.lightning_module import ArchesGenFinetune
        pl_module = ArchesGenFinetune.load_from_checkpoint(checkpoint)
        pl_module.eval()
    x_b = datamodule.background_state(t0)

    # 4. Solveur DA — chaque solveur déclare s'il a besoin du modèle et se branche
    #    lui-même (cf. Assimilator.bind_model), plutôt que le runner ne devine.
    assim = instantiate(cfg.assim, _convert_="all")
    if assim.requires_model:
        if pl_module is None:
            raise ValueError(
                f"{type(assim).__name__} requires a model — set checkpoint=path/to/ckpt"
            )
        assim.bind_model(pl_module)

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
