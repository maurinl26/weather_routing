#!/usr/bin/env bash
# Dev sur Mac M-series / poste local : sanity check + 1 epoch sur un tout
# petit set (sample_stride_hours grand) pour vérifier que la glue tourne.
set -euo pipefail

uv run wxr-train \
    experiment=finetune_bog \
    cluster=local \
    dataloader.sample_stride_hours=72 \
    dataloader.batch_size=2 \
    module.trainer.max_epochs=1
