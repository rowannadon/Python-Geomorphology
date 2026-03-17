from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .biome import BIOME_TABLE
from .util import compute_aspect_effect, compute_continentality, compute_elevation_zones

__all__ = ["compute_foliage_color_rgb", "compute_foliage_densities"]


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _saturate(x):
    return np.clip(x, 0.0, 1.0)


def _rgb_mix(rgb, target, amt):
    return _lerp(rgb, target, amt[..., None])


def _rgb_to_gray(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _sample_wetness_colormap(wet01: np.ndarray) -> np.ndarray:
    stops = np.array([
        [0.76, 0.70, 0.49],
        [0.71, 0.63, 0.37],
        [0.66, 0.79, 0.41],
        [0.42, 0.75, 0.29],
        [0.12, 0.48, 0.22],
    ], dtype=np.float32)
    tpos = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    t = _saturate(wet01)
    idx = np.clip(np.searchsorted(tpos, t, side="right") - 1, 0, len(tpos) - 2)
    t0 = tpos[idx]
    t1 = tpos[idx + 1]
    local = np.where((t1 - t0) > 1e-6, (t - t0) / (t1 - t0), 0.0)
    c0 = stops[idx]
    c1 = stops[idx + 1]
    return _lerp(c0, c1, local[..., None])


def compute_foliage_color_rgb(
    elev: np.ndarray,
    ocean: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    twi: Optional[np.ndarray],
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    dist_coast_m: np.ndarray,
    lat_deg_1d: np.ndarray,
    svf: Optional[np.ndarray],
    tpi_small: Optional[np.ndarray],
    cellsize: float,
) -> np.ndarray:
    """Return uint8 RGB foliage colour map derived from climate and terrain."""
    h, w = elev.shape
    ai = precip_mm / (pet_mm + 1e-6)
    ai_norm = _saturate(ai / 2.0)
    twi_term = np.zeros_like(ai_norm) if twi is None else _saturate((twi - 3.0) / 12.0)
    wet = _saturate(0.7 * ai_norm + 0.3 * twi_term)
    wet = 1.0 - np.exp(-1.4 * wet)

    base = _sample_wetness_colormap(wet)

    t01 = _saturate((temp_c + 10.0) / 45.0)
    warm_tint = np.array([1.00, 0.96, 0.70], dtype=np.float32)
    cool_tint = np.array([0.55, 0.80, 0.80], dtype=np.float32)
    tone_amt = 0.10
    toned = _rgb_mix(_rgb_mix(base, warm_tint, tone_amt * t01), cool_tint, tone_amt * (1.0 - t01))

    cont = compute_continentality(dist_coast_m / 1000.0, lat_deg_1d)
    cont01 = _saturate(cont / 1.2)
    gray = _rgb_to_gray(toned)[..., None]
    toned = _lerp(toned, gray, 0.20 * cont01[..., None])

    elev_z = compute_elevation_zones(elev)
    high = (elev_z >= 4).astype(np.float32)
    toned = _lerp(toned, _rgb_to_gray(toned)[..., None], 0.15 * high[..., None])
    toned = _saturate(toned + 0.06 * high[..., None])

    aspect_eff = compute_aspect_effect(aspect_deg, lat_deg_1d)
    slope01 = _saturate(slope_deg / 45.0)
    brightness = 1.0 + 0.05 * aspect_eff - 0.05 * slope01
    if svf is not None:
        brightness += 0.05 * (svf - 0.5)
    if tpi_small is not None:
        brightness += 0.06 * np.tanh(tpi_small / (3.0 * cellsize))
    brightness = np.clip(brightness, 0.85, 1.15)
    out = _saturate(toned * brightness[..., None])

    riparian = _saturate(wet * (twi_term if twi is not None else 0.0))
    out = _rgb_mix(out, np.array([0.10, 0.45, 0.18], dtype=np.float32), 0.15 * riparian)

    ocean_rgb = np.array(BIOME_TABLE[0][1], dtype=np.float32) / 255.0
    out[ocean] = ocean_rgb

    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def compute_foliage_densities(
    elev: np.ndarray,
    ocean: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    twi: Optional[np.ndarray],
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    dist_coast_m: np.ndarray,
    lat_deg_1d: np.ndarray,
    svf: Optional[np.ndarray],
    tpi_small: Optional[np.ndarray],
    cellsize: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return forest and groundcover density rasters in [0,1]."""
    ai = precip_mm / (pet_mm + 1e-6)
    ai01 = _saturate(ai / 2.0)
    twi_norm = np.zeros_like(ai01) if twi is None else _saturate((twi - 3.0) / 12.0)
    moisture01 = _saturate(0.7 * ai01 + 0.3 * twi_norm)

    def trap(x, a, b, c, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.clip((x - a) / max(1e-6, (b - a)), 0.0, 1.0)
            right = np.clip((d - x) / max(1e-6, (d - c)), 0.0, 1.0)
        return np.minimum(left, right)

    t_tree = trap(temp_c, -5.0, 5.0, 25.0, 32.0)
    t_ground = trap(temp_c, -20.0, -5.0, 30.0, 45.0)

    elev_z = compute_elevation_zones(elev)
    tree_elev_factor = np.ones_like(elev, dtype=np.float32)
    tree_elev_factor[elev_z == 5] = 0.35
    tree_elev_factor[elev_z >= 6] = 0.05

    def smoothstep(lo, hi, x):
        t = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    slope_tree = 1.0 - 0.6 * smoothstep(20.0, 55.0, slope_deg)
    slope_ground = 1.0 - 0.4 * smoothstep(35.0, 80.0, slope_deg)

    aspect_eff = compute_aspect_effect(aspect_deg, lat_deg_1d)
    dryness = 1.0 - moisture01
    aspect_tree = _saturate(1.0 - 0.15 * aspect_eff * dryness)
    aspect_ground = _saturate(1.0 - 0.05 * aspect_eff * dryness)

    tpi_term = np.zeros_like(elev, dtype=np.float32) if tpi_small is None else tpi_small
    tpi_norm = np.tanh(tpi_term / (3.0 * max(cellsize, 1e-3)))
    valley = _saturate(-tpi_norm)
    ridge = _saturate(tpi_norm)
    tpi_tree = _saturate(1.0 + 0.20 * valley - 0.10 * ridge)
    tpi_ground = _saturate(1.0 + 0.15 * ridge - 0.05 * valley)

    if svf is None:
        svf = np.full(elev.shape, 0.5, dtype=np.float32)
    tree_svf = np.exp(-0.5 * ((svf - 0.55) / 0.25) ** 2)
    tree_svf = _lerp(0.85, 1.10, _saturate(tree_svf))
    ground_svf = _saturate(0.9 + 0.25 * (svf - 0.5))

    cont = compute_continentality((dist_coast_m / 1000.0), lat_deg_1d)
    cont01 = _saturate(cont / 1.2)
    cont_tree = _saturate(1.0 - 0.10 * cont01)
    cont_ground = _saturate(1.0 - 0.03 * cont01)

    arid_gate = _saturate((ai - 0.05) / 0.20)
    arid_tree = arid_gate
    arid_ground = _saturate((ai - 0.02) / 0.15)

    cold_tree = _saturate((temp_c + 15.0) / 15.0)
    cold_ground = _saturate((temp_c + 25.0) / 25.0)

    forest_density = (
        (moisture01 ** 0.8)
        * t_tree
        * tree_elev_factor
        * slope_tree
        * aspect_tree
        * tpi_tree
        * tree_svf
        * cont_tree
        * arid_tree
        * cold_tree
    ).astype(np.float32)

    groundcover_density = (
        (0.6 * moisture01 + 0.4 * _saturate(ai / 1.5))
        * t_ground
        * slope_ground
        * aspect_ground
        * tpi_ground
        * ground_svf
        * cont_ground
        * arid_ground
        * cold_ground
    ).astype(np.float32)

    forest_density[ocean] = 0.0
    groundcover_density[ocean] = 0.0
    forest_density = _saturate(forest_density)
    groundcover_density = _saturate(groundcover_density)

    forest_density = gaussian_filter(forest_density, sigma=0.7)
    groundcover_density = gaussian_filter(groundcover_density, sigma=0.7)

    return forest_density.astype(np.float32), groundcover_density.astype(np.float32)
