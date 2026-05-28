"""Géométrie sphérique great-circle (degrés en entrée/sortie, km pour distances).

Toutes les fonctions sont vectorisées numpy : les arguments scalaires ou tableaux
se diffusent (broadcasting), ce qui permet de propager un front d'isochrones en une
seule passe.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance great-circle (km) entre deux points (ou tableaux de points)."""
    p1, l1, p2, l2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dphi = p2 - p1
    dlmb = l2 - l1
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Cap initial (degrés compas, 0=N, 90=E) de (lat1,lon1) vers (lat2,lon2)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def destination_point(lat, lon, bearing_deg, dist_km):
    """Point atteint depuis (lat,lon) en suivant `bearing_deg` sur `dist_km`.

    Renvoie (lat2, lon2) en degrés, longitude normalisée dans [-180, 180).
    """
    delta = np.asarray(dist_km) / EARTH_RADIUS_KM
    theta = np.radians(bearing_deg)
    p1 = np.radians(lat)
    l1 = np.radians(lon)
    sin_p2 = np.sin(p1) * np.cos(delta) + np.cos(p1) * np.sin(delta) * np.cos(theta)
    p2 = np.arcsin(np.clip(sin_p2, -1.0, 1.0))
    y = np.sin(theta) * np.sin(delta) * np.cos(p1)
    x = np.cos(delta) - np.sin(p1) * sin_p2
    l2 = l1 + np.arctan2(y, x)
    lat2 = np.degrees(p2)
    lon2 = (np.degrees(l2) + 540.0) % 360.0 - 180.0
    return lat2, lon2


def angle_to_180(angle_deg):
    """Replie un angle (degrés) dans [0, 180] — pour calculer un TWA."""
    return np.abs((np.asarray(angle_deg) + 180.0) % 360.0 - 180.0)
