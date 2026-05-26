"""Solveurs d'assimilation de données.

Trois schémas, du plus simple au plus moderne, tous sur la même interface :
- nudging        : relaxation directe vers les obs (baseline)
- enkf           : Ensemble Kalman Filter — ensemble fourni par la diffusion
- dps            : Diffusion Posterior Sampling — conditionne la diffusion sur y
"""

from .base import Assimilator, AssimResult

__all__ = ["Assimilator", "AssimResult"]
