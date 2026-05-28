#!/usr/bin/env bash
# Sur la box GPU (cf. infra/training). Installe l'env modèle et lance le run.
# Prérequis : repo cloné/scp ici, et en env : HF_TOKEN (Hugging Face).
# Args supplémentaires passés tels quels à wxr-train (ex. module.trainer.max_epochs=3).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # uv

echo "[setup] GPU :"; nvidia-smi -L || echo "  (pas de GPU détecté !)"

echo "[setup] uv sync --extra model (geoarches + torch CUDA)…"
uv sync --extra model

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] Hugging Face login…"
  uv run huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
else
  echo "[setup] HF_TOKEN absent — OK si le checkpoint pré-entraîné est public."
fi

echo "[setup] Lancement du fine-tuning (cluster=cloud)…"
uv run wxr-train experiment=finetune_bog cluster=cloud "$@"

echo "[setup] Terminé. Checkpoints sous outputs/ — pousser via scripts/push_checkpoint.sh"
