# Weather Routing — Fine-tuning d'ArchesWeatherGen sur le Golfe de Gascogne

> Adapter un modèle météo neuronal (ArchesWeatherGen, INRIA) à une zone
> régionale — le Golfe de Gascogne — puis y assimiler des **observations
> d'opportunité** (voiliers, cargos à assistance vélique, fermes éoliennes
> offshore, bouées, ASCAT) pour produire des prévisions de vent utiles au
> **routage maritime**.

Le dépôt sert **deux objectifs** qui partagent la même pile d'assimilation :

1. **Cours d'assimilation de données** — support pédagogique (TP, trois schémas
   de DA du nudging au score-based).
2. **PoC routage commercial** — démonstrateur pour la prospection (cf. vault
   Obsidian `routage-maritime/`), qui **réutilise** toute la pile DA comme couche
   de prévision et y ajoute le routage proprement dit.

### Périmètres — légende

Chaque brique ci-dessous est étiquetée :

- **`[Cours DA]`** — développée pour le cours d'assimilation, **réutilisée telle
  quelle** par la PoC comme couche de prévision wind-aware.
- **`[PoC routage]`** — spécifique à la PoC commerciale (hors périmètre du cours).

> En pratique, **toute la pile DA est `[Cours DA]` réutilisée** ; le delta propre
> à la PoC (`[PoC routage]`) se réduit au **module de routage**, à l'**extraction
> de données réelles** et à l'**évaluation/livrables commerciaux**.

---

## 1. Le modèle — ArchesWeatherGen &nbsp;`[Cours DA]`

**ArchesWeatherGen** est le modèle météo *génératif* (basé diffusion) de l'équipe
ARCHES d'INRIA, packagé dans la librairie [`geoarches`](https://github.com/INRIA/geoarches).
C'est un descendant des architectures *Pangu-Weather* / *GraphCast* / *GenCast*,
mais avec deux propriétés qui en font une excellente cible pédagogique :

- **Génératif** — il échantillonne une distribution `p(x_{t+Δt} | x_t)` plutôt
  qu'une moyenne conditionnelle. On peut donc construire des **ensembles** à
  faible coût et formuler l'assimilation comme un problème bayésien.
- **Backbone transformer** modulaire, fine-tunable sur des domaines régionaux
  à partir des poids globaux pré-entraînés sur ERA5.

| | |
|---|---|
| Pas de temps natif | 6 h (lead time configurable) |
| Résolution native | 1.5° global (0.25° sur ArchesWeather) |
| Variables surface | `u10`, `v10`, `t2m`, `msl` |
| Variables niveaux | `z`, `t`, `u`, `v`, `q` sur 13 niveaux pression |
| Entrée | snapshot ERA5 (ou prévision) |
| Sortie | snapshot t+6h (échantillonné par diffusion) |

**Pour le routage**, on s'intéresse surtout à `u10`, `v10` et `msl` ; les niveaux
pression restent dans le state vector pour conserver la cohérence dynamique
(le modèle pré-entraîné les attend en entrée).

### Pourquoi un modèle génératif pour la DA ?

L'assimilation de données classique (4D-Var, EnKF) repose sur une estimation
de la covariance d'erreur de prévision **B**. Avec un modèle de diffusion :

- on remplace **B** par un **prior implicite** appris (le score `∇ log p(x)`),
- l'assimilation devient un **score-based posterior sampling**
  (`∇ log p(x | y) = ∇ log p(x) + ∇ log p(y | x)`),
- on peut faire du **EnKF** quasi-gratuit en tirant N membres d'ensemble.

C'est l'angle pédagogique principal du cours.

---

## 2. Les données &nbsp;`[Cours DA]`

### 2.1 Champ d'arrière-plan (*background*) — ERA5

Deux sources, complémentaires :

- **ARCO-ERA5** (Google Cloud Storage, public, Analysis-Ready Cloud-Optimized) —
  utilisé pour le **training** : on lit en *streaming* via Zarr, pas de
  téléchargement massif. Idéal pour itérer.
- **Copernicus CDS** — utilisé ponctuellement pour récupérer des variables ou
  des périodes manquantes (par ex. les derniers mois non encore publiés sur ARCO).

Sous-domaine retenu (à confirmer) :

```
lat ∈ [43°N, 48°N]
lon ∈ [-10°E, -1°E]
```

soit ~5°×9° autour du plateau continental armoricain et du gouf de Capbreton.
Stocké en **Zarr** local après cropping pour les loops d'entraînement.

### 2.2 Observations d'opportunité

Quatre sources hétérogènes, chacune avec son **opérateur d'observation H** :

| Source | Variable | Géométrie | H(x) typique |
|---|---|---|---|
| **AIS + capteurs voiliers/cargos** | vent apparent ⇒ vent vrai à ~10 m | Lagrangienne, irrégulière | interp bilinéaire (lat,lon,t) sur `u10,v10` |
| **Champs éoliens offshore** | vent à hub height (~100 m) | Points fixes (mâts, SCADA) | interp + extrapolation log-loi sur le profil vertical |
| **Bouées Météo-France / CMEMS** | vent 10 m, pression, houle | Points fixes | interp (lat,lon,t) |
| **Scatteromètre ASCAT** | vent vecteur 10 m océan | Fauchée satellite, irrégulière | rééchantillonnage sur grille modèle, masque terre |

Le pipeline de données expose une **interface unique** :

```python
obs: dict = {
    "y":        ndarray,   # observations
    "coords":   ndarray,   # (lat, lon, time, [level])
    "sigma_o":  ndarray,   # erreur d'observation
    "H":        Callable,  # opérateur x -> H(x) différentiable
}
```

C'est le contrat qu'attendent les solveurs d'assimilation (cf. §4).

---

## 3. Fine-tuning régional &nbsp;`[Cours DA]`

Le modèle global d'ArchesWeatherGen ne "connaît" pas les spécificités du
Golfe de Gascogne (brises thermiques, accélérations sur le rail d'Ouessant,
effet de cap, vagues de tempête atlantique). On le **spécialise** par
fine-tuning sur la fenêtre régionale.

### Stratégie

1. **Charger les poids pré-entraînés** d'ArchesWeatherGen depuis Hugging Face.
2. **Restreindre le dataloader** à la fenêtre Golfe de Gascogne (cropping
   ERA5 + padding pour conserver le champ réceptif du transformer).
3. **Geler le backbone** dans un premier temps, fine-tuner seulement les
   blocs de sortie (LoRA-style possible).
4. **Dégeler progressivement** si le budget compute le permet.
5. **Valider** sur 2023-2024 (hors set d'entraînement) avec métriques :
   RMSE/CRPS sur `u10, v10`, spread/skill de l'ensemble.

### Stack technique

- [`geoarches`](https://github.com/INRIA/geoarches) — modèle + dataloader ERA5
- **PyTorch Lightning** — boucle d'entraînement, multi-GPU (DDP) prêt
- **Hydra** — composition de configs (cluster / dataloader / module)
- **uv** — gestion des dépendances Python
- **DVC** — versionnement des datasets cropés et des checkpoints

L'entrée est volontairement **agnostique du compute** : la même config Hydra
tourne sur Mac M-series (prototypage), sur SLURM (cluster acad.) ou sur GPU
cloud à la demande — seul change le groupe `cluster=...`.

---

## 4. Assimilation de données &nbsp;`[Cours DA]`

C'est le cœur pédagogique. Le repo expose **trois schémas** d'assimilation,
du plus classique au plus moderne, tous branchés sur le **même** state vector
et la **même** interface d'observations.

### 4.1 Nudging (baseline simple)

Le plus simple : à chaque pas de temps, on rapproche la prévision des
observations par un terme de relaxation `-α (x - H⁻¹(y))`. Pédagogiquement
utile pour introduire la notion d'innovation `d = y - H(x)`.

### 4.2 EnKF (Ensemble Kalman Filter)

On exploite le caractère **génératif** du modèle : N membres d'ensemble
échantillonnés par diffusion donnent une estimation empirique de la
covariance d'erreur **B**. Mise à jour classique :

```
x_a = x_b + K (y - H(x_b))         avec K = B Hᵀ (H B Hᵀ + R)⁻¹
```

C'est ici que l'on illustre l'intérêt d'un modèle génératif : **plus besoin de
faire tourner un modèle ensembliste coûteux**, l'ensemble *sort* du modèle.

### 4.3 Score-based / Diffusion Posterior Sampling

Le schéma le plus moderne : on conditionne directement la diffusion sur les
observations. À chaque étape de débruitage :

```
∇ log p(x_t | y) = ∇ log p(x_t)  +  ∇ log p(y | x_t)
                   └──── score ────┘   └─ vraisemblance gaussienne ─┘
```

Le premier terme est appris (c'est le modèle ArchesWeatherGen lui-même), le
second est analytique (`H` linéaire, `R` diagonale). Cela donne des
**analyses cohérentes** avec la physique apprise par le modèle, sans
hypothèse de gaussianité sur **B**.

### Application au routage &nbsp;`[PoC routage]`

Une fois l'analyse `x_a` produite, on relance ArchesWeatherGen depuis `x_a`
pour obtenir une **prévision de vent (ensemble) à 6/24/72 h** sur le Golfe de
Gascogne, puis on alimente un **solveur de routage** pour produire la route
optimale d'un voilier ou d'un cargo à assistance vélique.

Le routage est la brique **propre à la PoC commerciale** (au-delà du cours DA) :
il consomme l'ensemble de prévision comme champ de vent et propage l'incertitude
jusqu'à une enveloppe de routes. Module à venir dans `src/wxrouting/routing/`
(cf. note vault *Routage sur ArchesWeatherGen* et la PoC `routage-maritime/`).

---

## 5. Organisation du dépôt (à venir)

```
weather_routing/
├── configs/                # Hydra: cluster, dataloader, module, experiment
├── src/wxrouting/
│   ├── data/
│   │   ├── era5_arco.py    # streaming ARCO-ERA5
│   │   ├── era5_cds.py     # téléchargement CDS ponctuel
│   │   ├── crop.py         # cropping Golfe de Gascogne
│   │   └── obs/            # AIS, wind farms, buoys, ASCAT — H operators
│   ├── finetune/           # [Cours DA] boucles Lightning, callbacks LoRA
│   ├── assim/              # [Cours DA] nudging, EnKF, diffusion posterior sampling
│   └── routing/            # [PoC routage] solveur isochrone / DP (à venir)
├── notebooks/              # [Cours DA] supports de cours (TP)
├── scripts/                # SLURM, Docker, lancement cloud
└── tests/
```

---

## 6. Références

- **ArchesWeatherGen** — Couairon et al., *ArchesWeather & ArchesWeatherGen:
  a deterministic and generative model for efficient ML weather forecasting*, 2024.
- **GenCast** — Price et al., *GenCast: Diffusion-based ensemble forecasting
  for medium-range weather*, Nature 2024.
- **Score-based DA** — Rozet & Louppe, *Score-based Data Assimilation*, 2023.
- **ARCO-ERA5** — Carver & Merose, *ARCO-ERA5: An Analysis-Ready, Cloud-
  Optimized Reanalysis Dataset*, 2023.
- **Cours** — *(à compléter — lien vers le support de cours)*
