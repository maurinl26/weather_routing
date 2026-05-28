#!/usr/bin/env bash
# Pousse les checkpoints produits vers Scaleway Object Storage (S3-compatible),
# d'où l'app (provider AWG/corrigé) et wxr-assim pourront les charger.
#
# Env requis :
#   SCW_BUCKET            ex. wxrouting-checkpoints
#   SCW_S3_ENDPOINT       ex. https://s3.fr-par.scw.cloud
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  (clés API Scaleway)
set -euo pipefail

: "${SCW_BUCKET:?SCW_BUCKET manquant}"
: "${SCW_S3_ENDPOINT:=https://s3.fr-par.scw.cloud}"

# Récupère le meilleur checkpoint (filename finetune-*.ckpt) le plus récent.
CKPT=$(find outputs -name 'finetune-*.ckpt' -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)
[[ -z "$CKPT" ]] && CKPT=$(find outputs -name '*.ckpt' -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)
[[ -z "$CKPT" ]] && { echo "Aucun .ckpt trouvé sous outputs/"; exit 1; }

DEST="s3://${SCW_BUCKET}/finetune_bog/$(basename "$CKPT")"
echo "[push] $CKPT → $DEST"
aws s3 cp "$CKPT" "$DEST" --endpoint-url "$SCW_S3_ENDPOINT"
echo "[push] OK. Charger côté app/assim via : checkpoint=$DEST (ou téléchargement local)."
