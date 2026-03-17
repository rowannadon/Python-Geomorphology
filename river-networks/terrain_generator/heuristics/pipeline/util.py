import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "distance_to_mask",
    "compute_continentality",
    "compute_aspect_effect",
    "compute_elevation_zones",
]


def distance_to_mask(mask_true_targets: np.ndarray, cellsize: float) -> np.ndarray:
    """Return distance in meters from each False pixel to the nearest True pixel."""
    inv = ~mask_true_targets
    dist_px = distance_transform_edt(inv)
    return (dist_px * cellsize).astype(np.float32)


def compute_continentality(dist_coast_km: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Estimate continentality index (seasonal temperature variation)."""
    cont = np.tanh(dist_coast_km / 500.0)
    lat_factor = 1.0 + 0.5 * np.abs(np.sin(np.radians(lat_deg[:, None] * 2)))
    return (cont * lat_factor).astype(np.float32)


def compute_aspect_effect(aspect_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Return -1..1 aspect effect (polar-facing negative, equator-facing positive)."""
    h = aspect_deg.shape[0]
    lat = lat_deg[:, None]
    aspect_rad = np.radians(aspect_deg)
    ideal_aspect = np.where(lat > 0, np.pi, 0.0)
    aspect_effect = np.cos(aspect_rad - ideal_aspect)
    return aspect_effect.astype(np.float32)


def compute_elevation_zones(elev_m: np.ndarray) -> np.ndarray:
    """Map elevations to discrete zones 0..6 for biome/foliage heuristics."""
    zones = np.zeros_like(elev_m, dtype=np.uint8)
    zones[elev_m < 200.0] = 0
    zones[(elev_m >= 200.0) & (elev_m < 500.0)] = 1
    zones[(elev_m >= 500.0) & (elev_m < 1000.0)] = 2
    zones[(elev_m >= 1000.0) & (elev_m < 2000.0)] = 3
    zones[(elev_m >= 2000.0) & (elev_m < 3000.0)] = 4
    zones[(elev_m >= 3000.0) & (elev_m < 4000.0)] = 5
    zones[elev_m >= 4000.0] = 6
    return zones
