"""Utilitaires de gel/dégel sélectif du backbone ArchesWeatherGen."""

from __future__ import annotations

import torch.nn as nn


def freeze_backbone(
    model: nn.Module,
    head_keywords: tuple[str, ...] = ("head", "out", "decoder"),
) -> None:
    """Gèle tout sauf les modules dont le nom contient un des mots-clés.

    Par défaut : on dégèle têtes / couches de sortie / décodeur — c'est la
    surface "régionalisable" du modèle.
    """
    for name, p in model.named_parameters():
        p.requires_grad = any(kw in name.lower() for kw in head_keywords)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
