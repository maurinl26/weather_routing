"""Entrypoint Hydra d'assimilation.

Usage :
    wxr-assim experiment=assim_nudging
    wxr-assim experiment=assim_enkf  assim.checkpoint=...
    wxr-assim experiment=assim_dps   assim.checkpoint=...
"""

from __future__ import annotations

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..data.crop import Domain
from ..finetune.lightning_module import ArchesGenFinetune


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    domain = Domain(**OmegaConf.to_container(cfg.domain, resolve=True))
    datamodule = instantiate(cfg.dataloader, domain=domain)
    datamodule.setup("test")

    pl_module = ArchesGenFinetune.load_from_checkpoint(cfg.assim.checkpoint)
    pl_module.eval()

    # Construit le solveur — on injecte les hooks vers le modèle pour EnKF/DPS.
    assim = instantiate(cfg.assim, _convert_="all")
    if hasattr(assim, "sampler"):
        assim.sampler = pl_module.sample_ensemble
    if hasattr(assim, "score_fn"):
        assim.score_fn = getattr(pl_module.model, "score", None)
        assim.denoise_step = getattr(pl_module.model, "denoise_step", None)

    # Boucle d'une fenêtre — à étendre selon le scénario (TP du cours).
    batch = next(iter(datamodule.test_dataloader()))
    x_b = batch["x"]

    # Chargement des observations (à brancher sur les sources réelles).
    obs_sources: list = []  # à instancier depuis cfg.assim.obs_sources

    with torch.no_grad():
        result = assim.assimilate(x_b, obs_sources)
    print({k: v.shape for k, v in result.innov.items()})
    print("x_a shape:", result.x_a.shape)


if __name__ == "__main__":
    main()
