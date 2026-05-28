"""Entrypoint Hydra de fine-tuning.

Usage :
    wxr-train experiment=finetune_bog cluster=local
    wxr-train experiment=finetune_bog cluster=slurm cluster.devices=4
"""

from __future__ import annotations

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from ..data.crop import Domain
from ..finetune.callbacks import ProgressiveUnfreeze


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)

    domain = Domain(**OmegaConf.to_container(cfg.domain, resolve=True))
    datamodule = instantiate(cfg.dataloader, domain=domain)
    module = instantiate(cfg.module)

    callbacks: list[L.Callback] = [
        # Conserve le MEILLEUR checkpoint (val) + le dernier (reprise).
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="finetune-{epoch:02d}",
        ),
    ]
    if cfg.module.freeze.get("unfreeze_after_epochs"):
        callbacks.append(
            ProgressiveUnfreeze(after_epochs=cfg.module.freeze.unfreeze_after_epochs)
        )

    trainer = L.Trainer(
        accelerator=cfg.cluster.accelerator,
        devices=cfg.cluster.devices,
        num_nodes=cfg.cluster.num_nodes,
        strategy=cfg.cluster.strategy,
        precision=cfg.cluster.precision,
        max_epochs=cfg.module.trainer.max_epochs,
        gradient_clip_val=cfg.module.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.module.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.module.trainer.log_every_n_steps,
        val_check_interval=cfg.module.trainer.val_check_interval,
        default_root_dir=cfg.output_dir,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
