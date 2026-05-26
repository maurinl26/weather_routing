"""EnKF stochastique sur ensemble fourni par la diffusion.

Étape de mise à jour (perturbed obs) :
    K = B Hᵀ (H B Hᵀ + R)⁻¹
    x_a^(i) = x_b^(i) + K (y + ε^(i) - H(x_b^(i)))     , ε^(i) ~ N(0, R)

où B est estimée empiriquement depuis l'ensemble.

L'intérêt par rapport à un EnKF classique : pas besoin de propager un
ensemble dans un modèle de prévision coûteux — l'ensemble *sort* du modèle
de diffusion.
"""

from __future__ import annotations

import torch

from ..data.obs import Observation
from .base import Assimilator, AssimResult


def _gaspari_cohn(distances: torch.Tensor, c: float) -> torch.Tensor:
    """Fonction de localisation Gaspari-Cohn (compact support 2c)."""
    r = distances / c
    f = torch.zeros_like(r)
    m1 = r <= 1
    m2 = (r > 1) & (r <= 2)
    r1 = r[m1]
    r2 = r[m2]
    f[m1] = (((-r1 / 4 + 0.5) * r1 + 0.625) * r1 - 5/3) * r1**2 + 1
    f[m2] = ((((r2 / 12 - 0.5) * r2 + 0.625) * r2 + 5/3) * r2 - 5) * r2 + 4 - 2 / (3 * r2)
    return f


class EnKFAssimilator(Assimilator):
    def __init__(
        self,
        ensemble_size: int = 32,
        inflation: float = 1.05,
        localization_radius_km: float = 300.0,
        window_hours: int = 24,
        sampler=None,            # callable: x_b -> ensemble (N, B, C, H, W)
        **_: object,
    ):
        self.ensemble_size = ensemble_size
        self.inflation = inflation
        self.loc_c = localization_radius_km
        self.window_hours = window_hours
        self.sampler = sampler   # injecté par le runner (typiquement pl_module.sample_ensemble)

    def assimilate(
        self, x_b: torch.Tensor, observations: list[Observation]
    ) -> AssimResult:
        assert self.sampler is not None, "sampler must be injected (cf. cli/assimilate.py)"
        ens = self.sampler(x_b, self.ensemble_size)              # (N, B, C, H, W)
        N = ens.shape[0]

        # Inflation multiplicative autour de la moyenne d'ensemble.
        mean = ens.mean(dim=0, keepdim=True)
        ens = mean + self.inflation * (ens - mean)

        innov: dict[str, torch.Tensor] = {}

        for obs in observations:
            # H(x) pour chaque membre : (N, B, n_obs)
            Hx = torch.stack([obs.H(ens[i]) for i in range(N)], dim=0)
            y = torch.from_numpy(obs.y).to(x_b.device).float()    # (n_obs,)
            sigma = torch.from_numpy(obs.sigma_o).to(x_b.device).float()
            R = torch.diag(sigma**2)                              # (n_obs, n_obs)

            d = y.unsqueeze(0).unsqueeze(0) - Hx                  # (N, B, n_obs)
            innov[obs.source + "/" + obs.var_name] = d.mean(0)

            # Statistiques d'ensemble.
            Hx_anom = Hx - Hx.mean(0, keepdim=True)               # (N, B, n_obs)
            x_anom = ens - ens.mean(0, keepdim=True)              # (N, B, C, H, W)

            B_HT = torch.einsum("nbchw,nbm->bchwm", x_anom, Hx_anom) / (N - 1)
            HBHT = torch.einsum("nbm,nbk->bmk", Hx_anom, Hx_anom) / (N - 1)

            # Localisation : à brancher quand on a les distances obs-grille.
            # (laissé en placeholder pour le cours — TP à compléter)

            inv = torch.linalg.solve(HBHT + R, torch.eye(HBHT.shape[-1], device=x_b.device))
            K = torch.einsum("bchwm,bmk->bchwk", B_HT, inv)        # (B, C, H, W, n_obs)

            # Perturbed obs.
            eps = torch.randn(N, *y.shape, device=x_b.device) * sigma
            y_pert = y + eps                                       # (N, n_obs)
            d_pert = y_pert.unsqueeze(1) - Hx                      # (N, B, n_obs)
            update = torch.einsum("bchwk,nbk->nbchw", K, d_pert)
            ens = ens + update

        x_a = ens.mean(dim=0)
        return AssimResult(x_a=x_a, x_a_ens=ens, innov=innov)
