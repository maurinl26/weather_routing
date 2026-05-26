"""Vérifie que l'opérateur H bilinéaire est correct et différentiable."""

import numpy as np
import torch

from wxrouting.data.obs.operators import make_bilinear_H


def test_bilinear_H_exact_on_grid_point():
    grid_lat = np.array([48.0, 47.75, 47.5, 47.25, 47.0])
    grid_lon = np.array([350.0, 350.25, 350.5, 350.75, 351.0])  # = -10..-9
    # x : 2 canaux (u10, v10), une seule maille non nulle
    x = torch.zeros(1, 2, 5, 5)
    x[0, 0, 2, 2] = 7.0   # u10 au centre

    coords = np.array([[47.5, -9.5, 0.0]])  # centre grille
    H = make_bilinear_H(channel_index=0, coords=coords, grid_lat=grid_lat, grid_lon=grid_lon)
    y = H(x)
    assert torch.allclose(y, torch.tensor([[7.0]]), atol=1e-5)
    assert H.channel_index == 0


def test_bilinear_H_differentiable():
    grid_lat = np.array([48.0, 47.0])
    grid_lon = np.array([350.0, 351.0])
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    coords = np.array([[47.5, 350.5, 0.0]])
    H = make_bilinear_H(0, coords, grid_lat, grid_lon)
    y = H(x)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
