import math
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

__all__ = [
    "latitude_degrees",
    "prevailing_wind_3cell",
    "prevailing_wind",
    "directional_slope",
    "temperature_from_lat_elev",
    "precipitation_lat_bands",
    "compute_rain_shadow_advanced",
    "precipitation_orographic_advanced",
    "potential_evapotranspiration",
    "actual_evapotranspiration",
]


def latitude_degrees(h: int) -> np.ndarray:
    """Return 1D latitude array spanning 90..-90 degrees for map rows."""
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    lat = 90.0 - 180.0 * ys
    return lat.astype(np.float32)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def prevailing_wind_3cell(
    lat_deg: np.ndarray,
    eq_blend_deg: float = 5.0,
    ferrel_tilt: float = 0.12,
    polar_tilt: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth 3-cell atmospheric circulation model."""
    h = lat_deg.shape[0]
    lat = lat_deg.reshape(h, 1).astype(np.float32)
    a = np.abs(lat)

    w_tr = 1.0 - _smoothstep(25.0, 35.0, a)
    w_po = _smoothstep(55.0, 65.0, a)
    w_we = np.clip(1.0 - w_tr - w_po, 0.0, 1.0)

    s = np.tanh(np.radians(lat) / np.radians(eq_blend_deg))

    u_tr = -s
    v_tr = s

    u_we = np.ones_like(lat)
    v_we = ferrel_tilt * s

    u_po = -np.ones_like(lat)
    v_po = -polar_tilt * s

    u = w_tr * u_tr + w_we * u_we + w_po * u_po
    v = w_tr * v_tr + w_we * v_we + w_po * v_po

    mag = np.sqrt(u * u + v * v)
    tiny = 1e-8
    mask = mag < tiny
    if np.any(mask):
        pref = np.sign(w_we - w_po)
        pref = np.where(pref == 0, 1.0, pref)
        u[mask] = pref[mask]
        v[mask] = (ferrel_tilt - polar_tilt) * 0.5 * s[mask]
        mag[mask] = np.sqrt(u[mask] * u[mask] + v[mask] * v[mask])

    # Preserve the native magnitude so transitional bands naturally weaken the
    # wind-driven effects instead of flipping abruptly when cells change.
    strength = np.clip(mag, 0.0, 1.0)
    mag = np.maximum(mag, tiny)
    u = (u / mag) * strength
    v = (v / mag) * strength
    return u.astype(np.float32), v.astype(np.float32)


def prevailing_wind(
    lat_deg: np.ndarray,
    eq_blend_deg: float = 5.0,
    azimuth_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return constant-direction wind (legacy helper).

    azimuth_deg is measured clockwise from east (0° keeps legacy eastward flow).
    """
    h = lat_deg.shape[0]
    lat_deg.reshape(h, 1)  # shape alignment only
    theta = math.radians(float(azimuth_deg) % 360.0)
    dir_x = math.cos(theta)
    dir_y = math.sin(theta)
    u = np.full((h, 1), dir_x, dtype=np.float32)
    v = np.full((h, 1), dir_y, dtype=np.float32)
    mag = np.sqrt(u * u + v * v)
    mag[mag == 0] = 1.0
    strength = np.clip(mag, 0.0, 1.0)
    tiny = 1e-8
    mag = np.maximum(mag, tiny)
    u = (u / mag) * strength
    v = (v / mag) * strength
    return u.astype(np.float32), v.astype(np.float32)


def directional_slope(dzdx: np.ndarray, dzdy: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project gradient onto wind direction."""
    return (dzdx * u + dzdy * v).astype(np.float32)


def temperature_from_lat_elev(
    lat_deg: np.ndarray,
    elev: np.ndarray,
    lapse_c_per_km: float,
    t_equator_c: float,
    t_pole_c: float,
    pattern: str = "polar",
    gradient_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """Compute temperature field with optional planar gradient orientation."""

    mode = str(pattern).lower()

    if mode in {"polar", "default"}:
        lat_abs = np.abs(lat_deg)[:, None]
        coslat = np.cos(np.radians(lat_abs))
        base = t_pole_c + (t_equator_c - t_pole_c) * (coslat ** 1.0)
    elif mode in {"gradient", "linear"}:
        h, w = elev.shape
        ys = np.linspace(-0.5, 0.5, h, dtype=np.float32)
        xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        theta = np.radians(float(gradient_azimuth_deg) % 360.0)
        dir_x = np.sin(theta)
        dir_y = -np.cos(theta)
        grad = X * dir_x + Y * dir_y
        gmin = float(np.min(grad))
        gmax = float(np.max(grad))
        if gmax - gmin < 1e-6:
            norm = np.zeros_like(grad, dtype=np.float32)
        else:
            norm = (grad - gmin) / (gmax - gmin)
        base = t_equator_c + (t_pole_c - t_equator_c) * norm
    else:
        raise ValueError(f"Unknown temperature pattern: {pattern}")

    temp = base - (lapse_c_per_km * (elev / 1000.0))
    return temp.astype(np.float32)


def precipitation_lat_bands(
    lat_deg: np.ndarray,
    base_mm: float = 1200.0,
    pattern: str = "two_bands",
    width: int = 1,
    gradient_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """Return precipitation template for different latitude patterns.

    width controls the horizontal size of the returned pattern (defaults to 1 for
    legacy broadcasting). For gradient patterns, the azimuth is measured clockwise
    from north (0° = northward increase).
    """

    h = lat_deg.shape[0]
    w = max(1, int(width))
    lat_abs = np.abs(lat_deg)[:, None]

    def g(center: float, sigma: float, sign: float = 1.0) -> np.ndarray:
        return sign * np.exp(-0.5 * ((lat_abs - center) / sigma) ** 2)

    mode = str(pattern).lower()
    if mode in {"gradient", "linear"}:
        ys = np.linspace(-0.5, 0.5, h, dtype=np.float32)
        xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        theta = math.radians(float(gradient_azimuth_deg) % 360.0)
        dir_x = math.sin(theta)
        dir_y = -math.cos(theta)
        grad = X * dir_x + Y * dir_y
        gmin = float(np.min(grad))
        gmax = float(np.max(grad))
        if gmax - gmin < 1e-6:
            norm = np.zeros_like(grad, dtype=np.float32)
        else:
            norm = (grad - gmin) / (gmax - gmin)
        patt = 0.6 + 0.8 * (1.0 - norm)
    else:
        patt = np.ones((h, 1), dtype=np.float32)
        if mode in {"two_bands", "double", "legacy"}:
            patt += 0.9 * g(0.0, 12.0)
            patt += 0.6 * g(60.0, 10.0)
            patt += -0.9 * g(30.0, 10.0)
            patt += -0.5 * g(85.0, 5.0)
        elif mode in {"single_band", "single", "equatorial"}:
            patt += 1.1 * g(0.0, 14.0)
            patt += -0.3 * g(40.0, 12.0)
            patt += -0.4 * g(80.0, 8.0)
        elif mode in {"uniform", "flat", "constant"}:
            pass
        else:
            raise ValueError(f"Unknown precipitation pattern: {pattern}")
        if w > 1:
            patt = np.repeat(patt, w, axis=1)

    P = float(base_mm) * np.clip(patt, 0.1, None)
    return P.astype(np.float32)


def compute_rain_shadow_advanced(
    elev: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cellsize: float,
    max_distance_km: float = 50.0,
    shadow_decay_km: float = 20.0,
    height_threshold_m: float = 100.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Vectorized rain-shadow approximation based on average wind."""
    h, w = elev.shape
    u = wind_u[:, 0] if wind_u.shape[1] == 1 else wind_u.mean(axis=1)
    v = wind_v[:, 0] if wind_v.shape[1] == 1 else wind_v.mean(axis=1)
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    wind_mag = np.sqrt(avg_u ** 2 + avg_v ** 2)
    if wind_mag < 1e-6:
        return np.ones((h, w), dtype=np.float32)
    avg_u /= wind_mag
    avg_v /= wind_mag

    cellsize = max(float(cellsize), 1e-6)
    strength = max(0.0, float(strength))
    if strength == 0.0 or max_distance_km <= 0.0:
        return np.ones((h, w), dtype=np.float32)

    min_step_km = cellsize / 1000.0
    max_samples = 256
    if max_distance_km <= min_step_km:
        distances = np.array([max_distance_km], dtype=np.float32)
    else:
        est_steps = int(np.ceil(max_distance_km / min_step_km))
        n_steps = max(1, min(max_samples, est_steps))
        distances = np.linspace(min_step_km, max_distance_km, n_steps, dtype=np.float32)

    base_y, base_x = np.indices((h, w), dtype=np.float32)
    cval = float(np.min(elev))
    decay_den = max(shadow_decay_km, 1e-6)
    shadow_acc = np.zeros((h, w), dtype=np.float32)

    for dist_km in distances:
        dist_cells = (dist_km * 1000.0) / cellsize
        coords_y = base_y - dist_cells * avg_v
        coords_x = base_x - dist_cells * avg_u
        upwind = map_coordinates(
            elev,
            [coords_y, coords_x],
            order=1,
            mode="constant",
            cval=cval,
        ).astype(np.float32)

        height_diff = np.maximum(0.0, upwind - elev)
        if height_threshold_m > 0.0:
            mask = height_diff > height_threshold_m
            if not np.any(mask):
                continue
            height_diff = np.where(mask, height_diff, 0.0)
        elif not np.any(height_diff > 0.0):
            continue

        decay = math.exp(-dist_km / decay_den)
        contrib = np.minimum(0.3, height_diff / 1000.0) * decay * strength
        if not np.any(contrib > 0.0):
            continue
        shadow_acc = np.minimum(0.8, shadow_acc + contrib.astype(np.float32))

    shadow_mult = 1.0 - shadow_acc
    shadow_mult = gaussian_filter(shadow_mult, sigma=1.0)
    return np.clip(shadow_mult, 0.2, 1.0).astype(np.float32)


def precipitation_orographic_advanced(
    P_lat: np.ndarray,
    elev: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    dist_coast_m: np.ndarray,
    cellsize: float,
    alpha: float = 2.0,
    beta: float = 0.15,
    coast_decay_m: float = 150000.0,
    coast_min_frac: float = 0.75,
    use_advanced_shadow: bool = True,
    shadow_max_distance_km: float = 400.0,
    shadow_decay_km: float = 150.0,
    shadow_height_threshold_m: float = 150.0,
    shadow_strength: float = 1.0,
) -> np.ndarray:
    """Combine latitudinal precipitation with orographic lift and rain shadow."""
    dir_slope = (dzdx * wind_u + dzdy * wind_v).astype(np.float32)
    lift = 1.0 + alpha * np.maximum(0.0, dir_slope)

    if use_advanced_shadow:
        shadow_multiplier = compute_rain_shadow_advanced(
            elev,
            wind_u,
            wind_v,
            cellsize,
            max_distance_km=shadow_max_distance_km,
            shadow_decay_km=shadow_decay_km,
            height_threshold_m=shadow_height_threshold_m,
            strength=shadow_strength,
        )
    else:
        shadow_multiplier = 1.0 / (1.0 + beta * np.maximum(0.0, -dir_slope))

    coast = coast_min_frac + (1.0 - coast_min_frac) * np.exp(
        -dist_coast_m / max(1.0, coast_decay_m)
    )

    P = P_lat * lift * shadow_multiplier * coast
    return np.clip(P, 0.0, None).astype(np.float32)


def potential_evapotranspiration(
    temp_c: np.ndarray,
    lat_deg: np.ndarray,
    k: float = 20.0,
) -> np.ndarray:
    coslat = np.cos(np.radians(np.abs(lat_deg)))[:, None]
    coslat = np.clip(coslat, 0.2, 1.0)
    term = np.clip(temp_c + 5.0, 0.0, None)
    return (k * coslat * term).astype(np.float32)


def actual_evapotranspiration(P_mm: np.ndarray, PET_mm: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return ((P_mm * PET_mm) / (P_mm + PET_mm + eps)).astype(np.float32)
