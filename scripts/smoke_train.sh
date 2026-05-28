#!/usr/bin/env bash
# SMOKE de fine-tuning — à lancer EN PREMIER sur la box GPU pour valider toute la
# chaîne (geoarches, download HF, streaming ERA5, data path, forward/backward,
# checkpointing) à coût minime AVANT le vrai run.
#
# Fenêtres réduites à ~2 jours + cache RAM + 1 epoch ⇒ quelques minutes une fois
# le modèle chargé. Prérequis : env modèle installé (uv sync --extra model),
# HF_TOKEN si checkpoint privé.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

echo "[smoke] GPU :"; nvidia-smi -L || true

uv run wxr-train \
  experiment=finetune_bog \
  cluster=cloud \
  module.trainer.max_epochs=1 \
  module.warmup_epochs=0 \
  dataloader.cache_in_memory=true \
  dataloader.sample_stride_hours=6 \
  dataloader.batch_size=2 \
  'dataloader.train_period=[2024-06-01,2024-06-03]' \
  'dataloader.val_period=[2024-06-03,2024-06-04]' \
  'dataloader.test_period=[2024-06-03,2024-06-04]'

echo "[smoke] OK si une perte a été loggée et un checkpoint écrit sous outputs/."
echo "[smoke] Lancer ensuite le vrai run : scripts/setup_training.sh"
