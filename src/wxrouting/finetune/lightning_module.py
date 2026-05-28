"""Lightning module qui wrappe ArchesWeatherGen pour le fine-tuning régional.

L'import du modèle est paresseux (et tolérant à l'absence de `geoarches` en
environnement de dev) — ça permet d'instancier le module et de tester la
glue (configs, freeze, dataloader) sans GPU.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .freeze import count_trainable, freeze_backbone


def _load_geoarches_model(repo: str, revision: str) -> torch.nn.Module:
    """Charge ArchesWeatherGen depuis HF Hub via geoarches.

    Encapsulé dans une fonction pour rester optionnel — utile en CI / Mac dev.
    """
    try:
        from geoarches.lightning_modules import load_module  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "geoarches n'est pas installé. `uv sync` ou installer depuis "
            "https://github.com/INRIA/geoarches"
        ) from e
    return load_module(repo, revision=revision)


class ArchesGenFinetune(L.LightningModule):
    """Fine-tuning d'ArchesWeatherGen — perte de débruitage diffusion."""

    def __init__(
        self,
        pretrained_repo: str,
        pretrained_revision: str,
        freeze: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        diffusion: DictConfig,
        trainer: DictConfig,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = _load_geoarches_model(pretrained_repo, pretrained_revision)
        if freeze.get("backbone", False):
            freeze_backbone(self.model)
            trainable, total = count_trainable(self.model)
            # print natif : __init__ s'exécute AVANT l'attache au Trainer
            # (self.print/self.log lèveraient "not attached to a Trainer").
            print(f"[freeze] trainable params: {trainable}/{total}")

    # --------------------------------------------------------------------
    # Training : on s'appuie sur l'API training_step de geoarches si dispo
    # (les modèles diffusion ont déjà leur propre boucle). Sinon, fallback
    # naïf en MSE pour rester fonctionnel en CI.
    # --------------------------------------------------------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if hasattr(self.model, "training_step"):
            loss = self.model.training_step(batch, batch_idx)
        else:
            pred = self.model(batch["x"])
            loss = F.mse_loss(pred, batch["y"])
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if hasattr(self.model, "validation_step"):
            loss = self.model.validation_step(batch, batch_idx)
        else:
            with torch.no_grad():
                pred = self.model(batch["x"])
                loss = F.mse_loss(pred, batch["y"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    # --------------------------------------------------------------------
    # Inférence ensembliste (utilisée par les solveurs DA aval).
    # --------------------------------------------------------------------
    @torch.inference_mode()
    def sample_ensemble(self, x: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Renvoie un ensemble (n, B, C, H, W) — tirages indépendants par diffusion."""
        n = n or int(self.hparams.diffusion.ensemble_size)
        steps = int(self.hparams.diffusion.num_inference_steps)
        if hasattr(self.model, "sample"):
            return torch.stack(
                [self.model.sample(x, num_inference_steps=steps) for _ in range(n)], dim=0
            )
        # Fallback : ensemble dégénéré (utile uniquement en CI).
        return self.model(x).unsqueeze(0).expand(n, *((-1,) * x.ndim))

    def configure_optimizers(self) -> dict[str, Any]:
        from hydra.utils import instantiate

        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = instantiate(self.hparams.optimizer, params=params)
        sch = instantiate(self.hparams.scheduler, optimizer=opt)

        # Warmup linéaire (en epochs) puis cosine — le cosine seul est instable
        # en début de fine-tuning.
        warmup = int(self.hparams.get("warmup_epochs", 0) or 0)
        if warmup > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_sched = LinearLR(opt, start_factor=0.1, total_iters=warmup)
            sch = SequentialLR(opt, schedulers=[warmup_sched, sch], milestones=[warmup])

        return {"optimizer": opt, "lr_scheduler": sch}
