"""Opérateurs d'observation H différentiables (PyTorch).

Hypothèse de grille : régulière (lat, lon), latitude décroissante.
Tous les opérateurs prennent x: (B, C, H, W) et renvoient y_pred: (B, N).
"""

from __future__ import annotations

import numpy as np
import torch


def _grid_index(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Indices entiers + poids bilinéaires pour des points (lat, lon)."""
    # lat décroissante : on inverse pour searchsorted
    lat_sorted = grid_lat[::-1]
    i_rev = np.searchsorted(lat_sorted, lat) - 1
    i_rev = np.clip(i_rev, 0, len(lat_sorted) - 2)
    i = (len(lat_sorted) - 2) - i_rev  # remettre dans l'ordre décroissant
    # lon croissante
    lon_q = lon % 360.0
    grid_lon_q = grid_lon % 360.0
    j = np.clip(np.searchsorted(grid_lon_q, lon_q) - 1, 0, len(grid_lon_q) - 2)

    wi = (grid_lat[i] - lat) / (grid_lat[i] - grid_lat[i + 1])
    wj = (lon_q - grid_lon_q[j]) / (grid_lon_q[j + 1] - grid_lon_q[j])
    return i, j, wi, wj


def make_bilinear_H(
    channel_index: int,
    coords: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
):
    """H bilinéaire pour des obs (lat, lon) sur un canal du state vector.

    `coords[:, 0]` = lat, `coords[:, 1]` = lon ; les autres colonnes sont
    ignorées (le temps est géré par le solveur DA qui appelle H au bon pas).
    """
    i, j, wi, wj = _grid_index(coords[:, 0], coords[:, 1], grid_lat, grid_lon)
    i_t = torch.from_numpy(i).long()
    j_t = torch.from_numpy(j).long()
    wi_t = torch.from_numpy(wi).float()
    wj_t = torch.from_numpy(wj).float()

    def H(x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W) ; canal cible :
        c = x[:, channel_index]                          # (B, H, W)
        wi_b = wi_t.to(c.device)
        wj_b = wj_t.to(c.device)
        v00 = c[:, i_t, j_t]
        v01 = c[:, i_t, j_t + 1]
        v10 = c[:, i_t + 1, j_t]
        v11 = c[:, i_t + 1, j_t + 1]
        return (
            (1 - wi_b) * (1 - wj_b) * v00
            + (1 - wi_b) * wj_b * v01
            + wi_b * (1 - wj_b) * v10
            + wi_b * wj_b * v11
        )  # (B, N)

    H.channel_index = channel_index   # type: ignore[attr-defined]
    return H


def make_loglaw_H(
    u_index: int,
    v_index: int,
    coords: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    z_ref: float = 10.0,
    z0: float = 2e-4,  # rugosité mer ouverte (m)
):
    """H pour mesures de vent à hub height (~100 m) via profil log neutre.

    `coords[:, 3]` = hauteur (m).  Renvoie la norme du vent extrapolée.
    Approximation : profil log neutre marin, valide en première approche
    pour des fermes offshore loin de la côte.
    """
    H_u = make_bilinear_H(u_index, coords, grid_lat, grid_lon)
    H_v = make_bilinear_H(v_index, coords, grid_lat, grid_lon)
    z = torch.from_numpy(coords[:, 3]).float()
    scale = torch.log(z / z0) / np.log(z_ref / z0)  # (N,)

    def H(x: torch.Tensor) -> torch.Tensor:
        u10 = H_u(x)
        v10 = H_v(x)
        s = scale.to(u10.device)
        speed10 = torch.sqrt(u10**2 + v10**2 + 1e-12)
        return speed10 * s  # (B, N) vitesse à z

    return H
