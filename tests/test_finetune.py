"""Dérisque le fine-tuning SANS geoarches/GPU/réseau.

On patche le chargement du modèle geoarches par un modèle jouet, ce qui permet
de tester toute la « glue » Lightning (freeze, training_step de repli, ensemble,
optimizers, callback de dégel) avant de payer une instance GPU.
"""

import lightning as L
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import wxrouting.finetune.lightning_module as lm
from wxrouting.finetune.callbacks import ProgressiveUnfreeze
from wxrouting.finetune.freeze import count_trainable, freeze_backbone, unfreeze_all


class _DummyModel(nn.Module):
    """Backbone + tête (préserve la forme) — sans méthode `sample`."""

    def __init__(self, c: int = 4):
        super().__init__()
        self.backbone = nn.Conv2d(c, c, 1)
        self.head = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


def _cfgs():
    return dict(
        freeze=OmegaConf.create({"backbone": True, "unfreeze_after_epochs": 3}),
        optimizer=OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1e-4}),
        scheduler=OmegaConf.create(
            {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 10}
        ),
        diffusion=OmegaConf.create(
            {"num_train_steps": 1000, "num_inference_steps": 4, "ensemble_size": 4}
        ),
        trainer=OmegaConf.create({"max_epochs": 1}),
    )


def _make_module(monkeypatch, c: int = 4) -> lm.ArchesGenFinetune:
    monkeypatch.setattr(lm, "_load_geoarches_model", lambda repo, revision: _DummyModel(c))
    return lm.ArchesGenFinetune(
        pretrained_repo="dummy", pretrained_revision="main", **_cfgs()
    )


# --- freeze ------------------------------------------------------------------

def test_freeze_backbone_keeps_only_heads():
    m = _DummyModel()
    freeze_backbone(m)
    grad = {n: p.requires_grad for n, p in m.named_parameters()}
    assert grad["head.weight"] and grad["head.bias"]
    assert not grad["backbone.weight"] and not grad["backbone.bias"]
    trainable, total = count_trainable(m)
    assert trainable == sum(p.numel() for n, p in m.named_parameters() if "head" in n)
    assert total == sum(p.numel() for p in m.parameters())


def test_unfreeze_all():
    m = _DummyModel()
    freeze_backbone(m)
    unfreeze_all(m)
    assert all(p.requires_grad for p in m.parameters())


# --- module Lightning (geoarches patché) -------------------------------------

def test_module_applies_freeze_on_init(monkeypatch):
    module = _make_module(monkeypatch)
    grad = {n: p.requires_grad for n, p in module.model.named_parameters()}
    assert grad["head.weight"] is True
    assert grad["backbone.weight"] is False  # backbone gelé via freeze.backbone


def test_training_step_mse_fallback_runs(monkeypatch):
    module = _make_module(monkeypatch)
    monkeypatch.setattr(module, "log", lambda *a, **k: None)  # pas de Trainer attaché
    batch = {"x": torch.randn(2, 4, 8, 8), "y": torch.randn(2, 4, 8, 8)}
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad


def test_sample_ensemble_fallback_shape(monkeypatch):
    module = _make_module(monkeypatch)
    x = torch.randn(2, 4, 8, 8)
    ens = module.sample_ensemble(x, n=3)
    assert tuple(ens.shape) == (3, 2, 4, 8, 8)  # (membres, B, C, H, W)


def test_configure_optimizers_instantiates(monkeypatch):
    module = _make_module(monkeypatch)
    out = module.configure_optimizers()
    assert "optimizer" in out and "lr_scheduler" in out
    # seuls les params de tête (dégelés) sont optimisés
    n_opt = sum(p.numel() for g in out["optimizer"].param_groups for p in g["params"])
    assert n_opt == count_trainable(module.model)[0]


# --- callback de dégel progressif --------------------------------------------

class _FakeStrategy:
    def setup_optimizers(self, trainer):
        self.called = True


class _FakeTrainer:
    def __init__(self, epoch: int):
        self.current_epoch = epoch
        self.strategy = _FakeStrategy()


class _FakePL:
    def __init__(self, model):
        self.model = model

    def print(self, *a, **k):
        pass


def test_progressive_unfreeze_timing():
    m = _DummyModel()
    freeze_backbone(m)
    cb = ProgressiveUnfreeze(after_epochs=3)

    cb.on_train_epoch_start(_FakeTrainer(1), _FakePL(m))   # avant le seuil
    assert not m.backbone.weight.requires_grad
    assert not cb._done

    cb.on_train_epoch_start(_FakeTrainer(3), _FakePL(m))   # au seuil → dégel total
    assert m.backbone.weight.requires_grad
    assert cb._done


def test_progressive_unfreeze_is_a_lightning_callback():
    assert isinstance(ProgressiveUnfreeze(after_epochs=1), L.Callback)
