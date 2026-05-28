"""Diffusion Posterior Sampling — assimilation comme conditionnement de diffusion.

À chaque étape de débruitage :
    ∇ log p(x_t | y) = score(x_t)  +  g · ∇_x log p(y | x)
                       └─ ArchesWeatherGen ─┘    └─ analytique gaussien ─┘

où `g` est le `guidance_scale`. Le terme d'attache aux données vaut :
    log p(y | x) = -½ Σ_i ((y_i - H_i(x)) / σ_i)²

On rétro-propage à travers H (différentiable) pour obtenir le gradient.

Référence : Rozet & Louppe, *Score-based Data Assimilation* (NeurIPS 2023) ;
Chung et al., *Diffusion Posterior Sampling* (ICLR 2023).
"""

from __future__ import annotations

from typing import Any

import torch

from ..data.obs import Observation
from .base import Assimilator, AssimResult


class DPSAssimilator(Assimilator):
    requires_model = True

    def __init__(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        ensemble_size: int = 16,
        window_hours: int = 24,
        denoise_step=None,        # callable: (x_t, t) -> x_{t-1}
        score_fn=None,            # callable: (x_t, t) -> score (∇ log p(x_t))
        **_: object,
    ):
        self.T = num_inference_steps
        self.g = guidance_scale
        self.ensemble_size = ensemble_size
        self.window_hours = window_hours
        self.denoise_step = denoise_step
        self.score_fn = score_fn

    def bind_model(self, pl_module: Any) -> None:
        model = pl_module.model
        missing = [a for a in ("score", "denoise_step") if not hasattr(model, a)]
        if missing:
            raise AttributeError(
                f"DPS requires the diffusion model to expose {missing}; "
                f"{type(model).__name__} does not. Use a checkpoint/backbone that "
                f"implements score(x, t) and denoise_step(x, t, score=...)."
            )
        self.score_fn = model.score
        self.denoise_step = model.denoise_step

    def _log_likelihood_grad(
        self, x: torch.Tensor, observations: list[Observation]
    ) -> torch.Tensor:
        """∇_x log p(y | x) — gradient de la log-vraisemblance gaussienne."""
        x = x.detach().requires_grad_(True)
        total = x.new_zeros(())
        for obs in observations:
            y = torch.from_numpy(obs.y).to(x.device).float()
            sigma = torch.from_numpy(obs.sigma_o).to(x.device).float()
            Hx = obs.H(x)                                # (B, N)
            res = (y.unsqueeze(0) - Hx) / sigma
            total = total - 0.5 * (res**2).sum()
        grad, = torch.autograd.grad(total, x)
        return grad

    def assimilate(
        self, x_b: torch.Tensor, observations: list[Observation]
    ) -> AssimResult:
        assert self.denoise_step is not None and self.score_fn is not None, (
            "denoise_step et score_fn doivent être injectés (cf. cli/assimilate.py)"
        )

        innov: dict[str, torch.Tensor] = {}
        ens = []
        for _ in range(self.ensemble_size):
            x = x_b + torch.randn_like(x_b)                  # init bruitée
            for t in reversed(range(self.T)):
                # Score appris (modèle).
                s = self.score_fn(x, t)
                # Guidance par les observations.
                g = self._log_likelihood_grad(x, observations)
                x = self.denoise_step(x, t, score=s + self.g * g)
            ens.append(x)
        ens_t = torch.stack(ens, dim=0)                      # (N, B, C, H, W)

        # Diagnostic d'innovation post-analyse (sur la moyenne d'ensemble).
        x_a = ens_t.mean(0)
        for obs in observations:
            with torch.no_grad():
                Hx = obs.H(x_a)
                y = torch.from_numpy(obs.y).to(x_a.device).float()
                innov[obs.source + "/" + obs.var_name] = y.unsqueeze(0) - Hx

        return AssimResult(x_a=x_a, x_a_ens=ens_t, innov=innov)
