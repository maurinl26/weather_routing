# Architecture d'entraînement

> Document compagnon du [`README`](../README.md). On y détaille **comment** les
> composants logiciels s'articulent — flots de données, contrats inter-modules,
> stratégie de fine-tuning, hooks d'assimilation, et l'orchestration CI / cluster.

---

## 1. Vue d'ensemble

```
                ┌─────────────────────────────────────────────────────────┐
                │                       Hydra (configs/)                  │
                │   config.yaml ─┬─ cluster/  ─┬─ dataloader/  ─┬─ ...    │
                │                └─ module/    └─ experiment/             │
                └───────────────────────────┬─────────────────────────────┘
                                            │ compose
                                            ▼
   ┌──────────────┐    DataModule     ┌─────────────────┐    Lightning   ┌────────────────┐
   │ ARCO-ERA5    │ ─────────────────▶│  Era5Arco       │ ──────────────▶│  ArchesGen     │
   │ (GCS Zarr)   │   crop(Domain)    │  DataModule     │   (x_t, y_t+Δ) │  Finetune      │
   └──────────────┘                   └─────────────────┘                │ (Lightning)    │
   ┌──────────────┐                             │ (fine-tuning)          │  ┌──────────┐  │
   │ Copernicus   │ ─── download ───▶ cache local  ▶ ouverture xarray ──▶│  │ geoarches│  │
   │ CDS          │                                                      │  │ (HF ckpt)│  │
   └──────────────┘                                                      │  └──────────┘  │
                                                                         └────────┬───────┘
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐           │
   │ AIS / bateaux│  │ Wind farms   │  │ Bouées CMEMS │  │ ASCAT      │           │
   │  (Parquet)   │  │  (SCADA)     │  │  (NetCDF)    │  │  (NetCDF)  │           │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘           │
          └─────────────────┴─────────┬───────┴────────────────-┘                 │
                                      ▼                                           │
                              ObservationSource                                   │
                       (y, coords, σ_o, H différentiable)                         │
                                      │                                           │
                                      ▼                                           ▼
                                ┌────────────────────────────────────────────────────┐
                                │           Assimilator (assim/)                     │
                                │   nudging ──▶ EnKF ──▶ DPS (score-based)           │
                                │   consomme x_b (background) + obs                  │
                                │   produit  x_a (analyse) + ensemble                │
                                └────────────────────────────────────────────────────┘
```

Trois entrées clés à comprendre :

1. **Hydra** orchestre tout. Une expérience = une composition de groupes
   (`cluster + dataloader + module + experiment`). Aucun chemin codé en dur,
   aucune logique de switch dans le code.
2. **Lightning** porte la boucle d'entraînement et abstrait le compute
   (CPU / MPS / 1 GPU / DDP multi-nœuds). Le `cluster=...` ne change que
   `accelerator/devices/strategy/precision`.
3. **Une interface unique `Observation`** : c'est le verrou pédagogique du
   cours. Les trois solveurs (nudging, EnKF, DPS) acceptent exactement la
   même structure de données, ce qui rend leur comparaison directe.

---

## 2. Flots de données

### 2.1 Background (ERA5)

**ARCO-ERA5** est lu en *streaming* — pas de copie locale du dataset global.
Le bucket public `gs://gcp-public-data-arco-era5/...` expose un Zarr
chunké par variable et par temps. `xarray.open_zarr` + `gcsfs` (token
`anon`) donnent un `Dataset` paresseux ; le `crop(Domain)` réduit la
fenêtre à ~5°×9° **avant** matérialisation. Sur le Golfe de Gascogne, on
charge typiquement < 1% des données globales.

**Copernicus CDS** sert de complément :
- mois récents non encore publiés sur ARCO,
- variables ou niveaux pression absents.
Téléchargement chunked par `(year, month)` dans `data/era5_cds/` ; ouverture
ensuite avec `xr.open_mfdataset`. Nécessite `~/.cdsapirc`.

### 2.2 Observations d'opportunité

Le pipeline est en **deux étages** explicitement séparés. C'est le seul
endroit où on a accepté un peu de cérémonie, parce que l'invariant
"agnostique du paramètre physique" est central pour le cours.

```
┌──────────────────────────┐  pd.DataFrame  ┌───────────────────────────┐
│  Fetcher  (fetchers/)    │ ─────────────▶ │  ObservationAdapter       │
│                          │   normalisé    │  (obs/adapter.py)         │
│  CMEMS / ASCAT / AISHub  │   schéma CF    │                           │
│  EMODnet / VOS / Synth.  │                │  seul à connaître :       │
│                          │                │   - le state vector       │
│  Ne SAIT RIEN du modèle  │                │   - les opérateurs H      │
└──────────────────────────┘                └────────────┬──────────────┘
                                                         │ Observation[]
                                                         ▼
                                                  Assimilator (DA)
```

**Couche Fetcher** (`data/fetchers/`) — un module par source réelle. Tous
exposent la même méthode :

```python
class Fetcher(ABC):
    name: str
    def fetch(self, t0: str, t1: str, bbox: BBox) -> pd.DataFrame: ...
```

Le DataFrame est *normalisé* sur le schéma `(timestamp, lat, lon,
variable, value, [uncertainty, platform_id, ...])`. La colonne `variable`
contient un nom **CF standard** (cf. `data/cf_names.py`). Un fetcher peut
remonter du vent, des vagues, de la SST ou de la densité de trafic — le
code en aval s'adapte via le mapping.

Fetchers livrés :

| Module | Source | Auth | Cas d'usage |
|---|---|---|---|
| `cmems_insitu.py` | CMEMS In-Situ TAC (bouées, mâts) | Copernicus Marine | obs *ref* haute qualité |
| `ascat.py` | Scattéromètre Metop (KNMI ou EUMETSAT) | anonyme ou EUMDAC | vent océan satellite |
| `aishub.py` | AISHub (live, reciprocity) | clé AISHub | positions AIS temps réel |
| `emodnet_ais.py` | EMODnet Human Activities | anonyme | densité AIS historique |
| `vos_gts.py` | NOAA IMMA (VOS reports) | anonyme | obs météo navires marchands |
| `synthetic_ais.py` | générateur de voiliers virtuels | aucune | TP du cours sans clé |

**Couche Adapter** (`obs/adapter.py`) — paramétrée par un dict
`{variable_cf: ChannelSpec(channel_index, sigma_default, ...)}`. Elle :

1. **canonicalise** les alias de variable (`WSPD → wind_speed_at_10m`),
2. **décompose** automatiquement `(speed, direction) → (u, v)` si la source
   remonte du polaire et le modèle attend du cartésien,
3. **construit** l'opérateur `H` adapté (bilinéaire ou log-loi pour
   hub height), et **renvoie** des `Observation` :

```python
@dataclass
class Observation:
    y:        ndarray           # mesures (N,)
    coords:   ndarray           # (N, 3+)   lat, lon, t_hours, [z]
    sigma_o:  ndarray           # erreur d'obs (N,)
    H:        Callable          # x:(B,C,H,W) -> y_pred:(B,N)   DIFFÉRENTIABLE
    var_name: str               # nom CF canonique
    source:   str               # nom du fetcher d'origine
```

Variables non mappées → silencieusement ignorées (utile : un fetcher CMEMS
peut remonter du `wave_height` même si le modèle ne le sait pas).

**Propriété clé** : tous les `H` sont écrits en PyTorch pur ; `autograd`
fonctionne au travers, ce qui est obligatoire pour le DPS (qui rétro-propage
la log-vraisemblance jusqu'au state vector).

---

## 3. Le modèle et le state vector

ArchesWeatherGen attend un state vector très précis (ordre des canaux figé
par `data/registry.py`) :

| | nombre de canaux |
|---|---|
| Variables surface (`u10, v10, t2m, msl`)            | 4  |
| Variables niveaux × pressions (5 vars × 13 niveaux) | 65 |
| **Total**                                           | **69** |

Modifier l'ordre ou en supprimer **casse les poids HF**. Pour le routage on
ne consulte que `u10, v10, msl` en sortie, mais le pipeline porte les 69
canaux de bout en bout.

Le modèle lui-même est chargé paresseusement via
`geoarches.lightning_modules.load_module(repo, revision)`. L'import est
encapsulé pour que la CI (sans `geoarches`) puisse instancier les configs et
tester la glue sans télécharger les poids.

---

## 4. Boucle de fine-tuning

```
       ┌───────────────────────────────────────────────────┐
       │     ArchesGenFinetune (Lightning module)          │
       │                                                   │
   x ──▶│  model = geoarches.load_module(...)               │── loss
       │  freeze_backbone(...)                             │
       │                                                   │
       │  training_step:                                   │
       │     ├─ si geoarches a sa propre boucle diffusion :│
       │     │     return model.training_step(batch)        │
       │     └─ sinon (CI / fallback)  : MSE(pred, target) │
       │                                                   │
       │  configure_optimizers: AdamW + CosineAnnealingLR  │
       └───────────────────────────────────────────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │  Callback ProgressiveUnfreeze│  ← dégèle backbone après N epochs
          └──────────────────────────────┘
```

**Stratégie de gel** (`finetune/freeze.py`) :
- on filtre les paramètres dont le nom **contient** `head|out|decoder` →
  seuls ceux-là sont entraînables au démarrage. Cela isole la "surface
  régionalisable" du modèle ;
- le callback `ProgressiveUnfreeze` dégèle tout après `unfreeze_after_epochs`
  epochs et **reconstruit l'optimiseur** (sinon les nouveaux paramètres ne
  reçoivent pas de gradient effectif).

LoRA est en placeholder — facile à insérer en bridant les têtes via `peft`.

**Ensemble pour la DA** : `sample_ensemble(x, n)` exécute `n` tirages
indépendants du modèle de diffusion. C'est l'unique point d'entrée
consommé par les solveurs aval (EnKF, DPS).

---

## 5. Solveurs d'assimilation

Tous reçoivent `(x_b, [Observation, ...])` et renvoient un `AssimResult`
(`x_a`, ensemble optionnel, dictionnaire d'innovations pour diagnostic).

| Solveur | Hypothèse sur B | Coût | Usage pédagogique |
|---|---|---|---|
| **Nudging** | aucune (terme de relaxation) | trivial | introduire `d = y - H(x)` |
| **EnKF**    | gaussienne empirique sur ensemble | `N` tirages de diffusion + résolution `(n_obs × n_obs)` | montrer l'avantage des modèles génératifs : ensemble *gratuit* |
| **DPS**     | prior implicite `∇ log p(x)` appris | `T × N` évaluations du score + grad de la vraisemblance | DA bayésienne moderne, sans hypothèse de gaussianité sur B |

L'injection des hooks modèle → solveur se fait dans `cli/assimilate.py` :

```python
assim.sampler      = pl_module.sample_ensemble        # EnKF
assim.score_fn     = pl_module.model.score            # DPS
assim.denoise_step = pl_module.model.denoise_step     # DPS
```

Découpler ainsi évite que `assim/` dépende de `finetune/` (et donc de
`geoarches`) — c'est cette indépendance qui permet les tests unitaires sans
GPU.

---

## 6. Orchestration : Hydra × Lightning × cluster

### Composition des configs

```
configs/
├── config.yaml              # racine, defaults: cluster/local + dataloader/era5_arco_bog + module/...
├── cluster/                 # local | slurm | cloud
├── dataloader/              # era5_arco_bog | era5_cds_bog
├── module/                  # archesweathergen_finetune
└── experiment/              # finetune_bog | assim_{nudging,enkf,dps}
                             # un `experiment` OVERRIDE les autres groupes
```

Un même `wxr-train experiment=finetune_bog` tourne :
- en local : `cluster=local` → MPS/CPU, `devices=1`, `precision=32`
- sur SLURM : `cluster=slurm` → DDP, `bf16-mixed`, `devices=4`
- sur cloud : `cluster=cloud` → 1 GPU, `bf16-mixed`

**Aucune ligne de code Python à changer.** C'est l'invariant compute-agnostique.

### Soumission SLURM

`scripts/slurm.sbatch` reçoit `EXP` par variable d'environnement, lit
`SLURM_NTASKS_PER_NODE` pour aligner `cluster.devices`, et appelle
`srun uv run wxr-train`. Les variables `MASTER_ADDR/MASTER_PORT` sont
exportées avant `srun` pour Lightning DDP.

### Cloud / Docker

`scripts/Dockerfile` part de `pytorch/pytorch:2.4.0-cuda12.4` + `uv sync`.
L'ENTRYPOINT est `uv run` ; le CMD lance directement `wxr-train`. Adapté
à Lambda, RunPod, GCP A100/H100 via job template.

---

## 7. CI / CD

Trois workflows GitHub Actions, rôles séparés :

| Workflow | Trigger | But |
|---|---|---|
| **`ci.yml`** | push/PR `main` | Ruff + pytest, matrice Py 3.11/3.12, sans `geoarches` (deps minimales) |
| **`smoke_train.yml`** | PR `main` | Compose la config Hydra `finetune_bog` à blanc — détecte les casses de typage / clés manquantes / instanciation cassée, sans GPU ni dataset |
| **`train_dispatch.yml`** | `workflow_dispatch` manuel | SSH vers cluster SLURM, `git pull`, `uv sync`, `sbatch scripts/slurm.sbatch` |

**Choix volontaires** :

- **`geoarches` exclu de la CI** : tire torch GPU + libs CUDA, trop lourd pour
  des runners GitHub-hosted. Les modules qui en dépendent (Lightning module)
  importent paresseusement et CI valide le reste.
- **Pas d'entraînement réel en CI** : un smoke-train de qualité (compose +
  glue Lightning) attrape 90% des régressions sans brûler de GPU. Un vrai
  run se déclenche à la demande via `train_dispatch`.
- **Pas de runner GPU self-hosted** par défaut : c'est une option à activer
  si tu veux des tests d'intégration GPU sur PR. Coût élevé pour la valeur
  ajoutée dans un repo pédagogique.

---

## 8. Reproductibilité

- **`seed_everything`** en tête d'entrypoint, valeur lue depuis `cfg.seed`.
- **Hydra dump** la config résolue dans `outputs/<date>/<heure>/.hydra/` —
  rejouer une expérience = relire ce dossier.
- **Checkpoints** Lightning dans `outputs/.../checkpoints/` (ignorés par git).
- **Données** non versionnées dans le repo ; DVC en option (cf. `.gitignore`).

---

## 9. Points d'extension explicitement laissés ouverts

Ce sont des hooks délibérément non implémentés — chacun est un bon TP pour
le cours.

1. **Localisation Gaspari-Cohn dans l'EnKF** (`assim/enkf.py`) — la
   structure est posée mais la matrice de localisation n'est pas appliquée.
2. **Adjoint exact pour le nudging spatial** — la version actuelle pousse
   la moyenne des innovations sur tout le canal ; un vrai nudging
   splatterait les innovations à leurs voisinages.
3. **LoRA** pour le fine-tuning — `finetune/freeze.py` ne supporte que
   freeze/unfreeze ; brancher `peft` est mécanique.
4. **Couplage solveur de routage** (`src/wxrouting/routing/`) — pas créé,
   à brancher selon le solveur isochrone retenu.
5. **Boucle d'assimilation cyclée** dans `cli/assimilate.py` — ne traite
   qu'une fenêtre, à itérer (analyse → propagation → analyse).
