# Box d'entraînement — Scaleway L40S (Terraform)

Instance GPU **éphémère** pour lancer le fine-tuning, puis détruite (coût GPU).
L40S ~€1,40/h → **~17–67 €** pour quelques epochs régionaux. Cf. note vault
*Infra et coûts cloud*.

```bash
export SCW_ACCESS_KEY=... SCW_SECRET_KEY=... SCW_DEFAULT_PROJECT_ID=...
cd infra/training
terraform init
terraform apply \
  -var project_id="$SCW_DEFAULT_PROJECT_ID" \
  -var admin_ip_range="$(curl -s ifconfig.me)/32"
# → ssh root@<ip>

# Sur la box :
git clone <repo> weather_routing && cd weather_routing     # ou scp
export HF_TOKEN=hf_xxx                                       # si checkpoint privé
uv sync --extra model                                        # geoarches + torch CUDA
scripts/smoke_train.sh                                       # 1) SMOKE (~min) : valide la chaîne
scripts/setup_training.sh                                    # 2) vrai run
# … puis pousser le checkpoint :
export SCW_BUCKET=wxrouting-checkpoints
export AWS_ACCESS_KEY_ID=$SCW_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$SCW_SECRET_KEY
scripts/push_checkpoint.sh

# Local, après la run :
cd infra/training && terraform destroy
```

## ⚠️ À vérifier avant `apply` (scaffold non appliqué)
- **`image`** : image GPU OS Scaleway (drivers NVIDIA) — vérifier le slug.
- **`instance_type`** : dispo de `L40S-1-48G` en zone ; sinon `L4-1-24G` (backbone gelé).
- **`admin_ip_range`** : restreindre à votre IP.
- Clone du repo **privé** : token git ou `scp` depuis le poste.

## Sanity avant de payer
Le pipeline (glue, configs, data) est couvert par les tests (`tests/test_finetune.py`,
`test_era5_arco.py`, compose `finetune_bog`). Reste non testable hors box :
install geoarches, download HF, entraînement CUDA réel, VRAM.
