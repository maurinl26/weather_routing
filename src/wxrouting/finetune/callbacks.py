"""Callbacks Lightning : dégel progressif, log d'ensembles, etc."""

from __future__ import annotations

import lightning as L

from .freeze import count_trainable, unfreeze_all


class ProgressiveUnfreeze(L.Callback):
    """Dégèle l'intégralité du modèle après `after_epochs` epochs."""

    def __init__(self, after_epochs: int):
        self.after_epochs = after_epochs
        self._done = False

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._done or trainer.current_epoch < self.after_epochs:
            return
        unfreeze_all(pl_module.model)
        trainable, total = count_trainable(pl_module.model)
        pl_module.print(f"[ProgressiveUnfreeze] all params trainable: {trainable}/{total}")
        # On reconstruit l'optimizer pour qu'il voie les nouveaux params.
        trainer.strategy.setup_optimizers(trainer)
        self._done = True
