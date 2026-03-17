from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .ocean import compute_ocean_mask
from .util import (
    compute_aspect_effect,
    compute_continentality,
    compute_elevation_zones,
    distance_to_mask,
)

__all__ = [
    "BIOME_TABLE",
    "compute_moisture_index",
    "compute_wind_exposure",
    "calculate_biome_scores",
    "apply_probabilistic_mixing",
    "assign_biomes_from_scores",
    "classify_biomes_advanced",
]

BIOME_TABLE: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0: ("ocean", (30, 60, 150)),
    1: ("ice sheet", (230, 245, 255)),
    2: ("polar desert", (210, 220, 230)),
    3: ("arctic tundra", (185, 200, 205)),
    4: ("alpine tundra", (170, 185, 190)),
    5: ("alpine meadow", (140, 170, 140)),
    6: ("montane forest", (70, 110, 70)),
    7: ("boreal forest", (50, 90, 60)),
    8: ("mixed boreal", (65, 105, 75)),
    9: ("temperate coniferous", (40, 100, 55)),
    10: ("temperate rainforest", (25, 85, 45)),
    11: ("temperate deciduous", (80, 140, 70)),
    12: ("temperate mixed", (70, 125, 65)),
    13: ("temperate grassland", (160, 180, 90)),
    14: ("prairie", (180, 190, 100)),
    15: ("steppe", (190, 180, 110)),
    16: ("mediterranean woodland", (140, 150, 80)),
    17: ("chaparral", (150, 140, 70)),
    18: ("cold desert", (200, 190, 150)),
    19: ("hot desert", (230, 210, 130)),
    20: ("semi-arid scrubland", (210, 190, 120)),
    21: ("dry savanna", (220, 195, 100)),
    22: ("moist savanna", (200, 180, 90)),
    23: ("tropical dry forest", (110, 150, 70)),
    24: ("tropical seasonal forest", (70, 140, 60)),
    25: ("tropical rainforest", (20, 110, 40)),
    26: ("cloud forest", (30, 100, 50)),
    27: ("mangrove", (60, 120, 100)),
    28: ("freshwater wetland", (80, 180, 170)),
    29: ("salt marsh", (120, 160, 140)),
}


def compute_moisture_index(precip_mm: np.ndarray, pet_mm: np.ndarray, twi: np.ndarray) -> np.ndarray:
    """Enhanced moisture index combining precipitation, PET and TWI."""
    ai = precip_mm / (pet_mm + 1e-6)
    twi_norm = np.clip((twi - 3.0) / 12.0, 0.0, 1.0)
    moisture = 0.7 * ai + 0.3 * twi_norm
    return moisture.astype(np.float32)


def compute_wind_exposure(
    elev: np.ndarray,
    slope_deg: np.ndarray,
    tpi: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """Approximate wind exposure from topography (wind vectors unused placeholder)."""
    topo_exposure = np.tanh(tpi / 100.0)
    slope_exposure = np.tanh(slope_deg / 30.0)
    exposure = 0.6 * topo_exposure + 0.4 * slope_exposure
    return np.clip(exposure, -1.0, 1.0).astype(np.float32)

def calculate_biome_scores(
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    moisture: np.ndarray,
    continentality: np.ndarray,
    wind_exposure: np.ndarray,
    aspect_effect: np.ndarray,
    elev_zones: np.ndarray,
    ocean: np.ndarray,
) -> np.ndarray:
    """Return biome probability cube for each pixel (retuned for better rare-biome expression)."""
    h, w = temp_c.shape
    n_biomes = len(BIOME_TABLE)
    scores = np.zeros((h, w, n_biomes), dtype=np.float32)

    # --- helpers ---
    def gaussian(x, c, w):
        return np.exp(-0.5 * ((x - c) / (w + 1e-6)) ** 2)

    def trap(x, a, b, c, d):
        return np.maximum(
            0.0,
            np.minimum(
                np.minimum((x - a) / (b - a + 1e-6), 1.0),
                np.minimum((d - x) / (d - c + 1e-6), 1.0),
            ),
        )

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x)) + 0.00001

    # --- climate gates (soft) ---
    gate_trop  = sigmoid((temp_c - 18.0) / 2.5)
    gate_temp  = sigmoid((temp_c -  8.0) / 3.0) * sigmoid((22.0 - temp_c) / 3.0)
    gate_boreal = sigmoid(( 8.0 - temp_c) / 3.0) * sigmoid((temp_c + 4.0) / 3.0)
    gate_polar = sigmoid((-5.0 - temp_c) / 2.5)

    # --- aridity (wider semi-dry/dry band) ---  # <<< tuned
    very_dry = sigmoid((0.18 - moisture) / 0.05)
    dry      = sigmoid((0.28 - moisture) / 0.07)
    semi_dry = sigmoid((0.42 - moisture) / 0.08)
    humid    = sigmoid((moisture - 0.60) / 0.08)
    very_humid = sigmoid((moisture - 0.82) / 0.07)

    # Elevation/coast
    land = ~ocean
    montane_mask    = land & (elev_zones >= 3)
    submontane_mask = land & (elev_zones >= 2)
    lowland_mask    = (elev_zones == 0)
    coastal_mask    = land & (distance_to_mask(ocean, 1.0) < 4000)  # tighter coast band 4 km  # <<< tuned
    coastal_boost   = 1.0 + 0.30 * coastal_mask.astype(float)       # helps Mediterranean only  # <<< tuned

    # --- priors (balance) ---  # <<< tuned a bunch
    PRIORS = {
        0: 1.0,
        1: 1.00,  # ice sheet
        2: 1.00,  # polar desert
        3: 1.00,  # arctic tundra
        4: 0.95,  # alpine tundra
        5: 0.85,  # alpine meadow (too big before)
        6: 1.00,  # montane forest
        7: 0.95,  # boreal forest
        8: 1.00,  # mixed boreal
        9: 0.95,  # temperate coniferous
        10: 0.90, # temperate rainforest
        11: 0.95, # temperate deciduous
        12: 1.00, # temperate mixed
        13: 1.25, # temperate grassland ↑
        14: 1.15, # prairie ↑
        15: 1.10, # steppe ↑
        16: 1.25, # mediterranean woodland ↑
        17: 1.10, # chaparral ↑
        18: 1.00, # cold desert
        19: 0.95, # hot desert
        20: 1.25, # semi-arid scrubland ↑
        21: 1.20, # dry savanna ↑
        22: 1.00, # moist savanna ↓ (was 1.05)
        23: 1.25, # tropical dry forest ↑
        24: 1.00, # tropical seasonal
        25: 0.85, # tropical rainforest ↓
        26: 0.85, # cloud forest ↓ (too common)
        27: 0.90, # mangrove ↓
        28: 0.95, # freshwater wetland ↓
        29: 0.90, # salt marsh ↓
    }

    # Oceans deterministic
    scores[:, :, 0][ocean] = 1000.0

    # --- Cryosphere/Polar (unchanged) ---
    scores[:, :, 1][land] = gate_polar[land] * gaussian(precip_mm[land], 150, 120) * gaussian(temp_c[land], -20, 7)
    scores[:, :, 2][land] = gate_polar[land] * dry[land] * gaussian(temp_c[land], -12, 6)
    scores[:, :, 3][land] = gate_boreal[land] * humid[land] * gaussian(temp_c[land], -7, 5)

    # --- Alpine / Montane (narrower alpine meadow) ---
    if np.any(montane_mask):
        scores[:, :, 4][montane_mask] = gate_boreal[montane_mask] * gaussian(moisture[montane_mask], 0.55, 0.16)  # alpine tundra
        # alpine meadow prefers mid moisture band only (not saturated)
        scores[:, :, 5][montane_mask] = gate_temp[montane_mask] * trap(moisture[montane_mask], 0.55, 0.62, 0.78, 0.85)  # <<< tuned
        scores[:, :, 6][montane_mask] = gate_temp[montane_mask] * humid[montane_mask] * gaussian(precip_mm[montane_mask], 1200, 380)

    # --- Boreal / Cool Temperate (unchanged-ish) ---
    scores[:, :, 7][land] = gate_boreal[land] * gaussian(precip_mm[land], 550, 220) * gaussian(continentality[land], 0.6, 0.25) * gaussian(moisture[land], 0.5, 0.18)
    scores[:, :, 8][land] = gate_boreal[land] * gaussian(precip_mm[land], 650, 220) * gaussian(continentality[land], 0.5, 0.25) * gaussian(moisture[land], 0.6, 0.18)

    # --- Temperate Forests (slightly more moisture-demanding) ---  # <<< tuned
    scores[:, :, 9][land]  = gate_temp[land] * gaussian(precip_mm[land], 1000, 450) * gaussian(moisture[land], 0.68, 0.16) * (1 + 0.15 * aspect_effect[land])
    scores[:, :, 10][land] = gate_temp[land] * very_humid[land] * gaussian(precip_mm[land], 1850, 580) * gaussian(continentality[land], 0.2, 0.25)
    scores[:, :, 11][land] = gate_temp[land] * gaussian(precip_mm[land], 800, 320)  * gaussian(moisture[land], 0.58, 0.16) * gaussian(continentality[land], 0.4, 0.25)
    scores[:, :, 12][land] = gate_temp[land] * gaussian(precip_mm[land], 900, 320)  * gaussian(moisture[land], 0.62, 0.16) * gaussian(continentality[land], 0.5, 0.25)

    # --- Temperate Grasslands & Mediterranean (stronger open-biome signals) ---
    # Temperate grassland: drier band + interior + wind exposure  # <<< tuned
    interior_boost = (0.7 + 0.6 * continentality).clip(0.7, 1.3)  # favors interiors
    scores[:, :, 13][land] = gate_temp[land] * semi_dry[land] * gaussian(precip_mm[land], 420, 180) * (1 + 0.50 * wind_exposure[land]) * interior_boost[land]
    scores[:, :, 14][land] = gate_temp[land] * semi_dry[land] * gaussian(continentality[land], 0.7, 0.22)
    # Mediterranean woodland: semi-dry, modestly coastal but not required  # <<< tuned
    scores[:, :, 16][land] = gate_temp[land] * semi_dry[land] * gaussian(precip_mm[land], 560, 220) * (1 - 0.15 * continentality[land]) * coastal_boost[land]
    scores[:, :, 17][land] = gate_temp[land] * dry[land]      * gaussian(precip_mm[land], 400, 160) * (1 + 0.15 * wind_exposure[land])  # chaparral

    # --- Deserts & Semi-arid (broaden scrub) ---
    scores[:, :, 18][land] = (gate_boreal[land] + gate_temp[land]) * very_dry[land] * gaussian(temp_c[land], 6, 6)   * gaussian(continentality[land], 0.9, 0.15)
    scores[:, :, 19][land] = (gate_temp[land] + gate_trop[land]) * very_dry[land] * gaussian(temp_c[land], 27, 6)  * (1 + 0.20 * wind_exposure[land])
    scores[:, :, 20][land] = (gate_temp[land] + gate_trop[land]) * dry[land]      * gaussian(precip_mm[land], 300, 130) * gaussian(moisture[land], 0.25, 0.10)  # <<< tuned

    # --- Tropical savannas/forests (rebalance toward dry) ---
    scores[:, :, 21][land] = gate_trop[land] * semi_dry[land] * gaussian(precip_mm[land], 450, 180) * gaussian(moisture[land], 0.30, 0.12)  # dry savanna (slightly drier)  # <<< tuned
    scores[:, :, 22][land] = gate_trop[land] * gaussian(precip_mm[land], 1000, 220) * gaussian(moisture[land], 0.52, 0.14) * gaussian(continentality[land], 0.5, 0.30)  # moist savanna (narrowed)  # <<< tuned
    scores[:, :, 23][land] = gate_trop[land] * semi_dry[land] * gaussian(precip_mm[land], 900, 280) * gaussian(continentality[land], 0.45, 0.28)  # tropical dry forest ↑  # <<< tuned
    scores[:, :, 24][land] = gate_trop[land] * gaussian(precip_mm[land], 1400, 320) * gaussian(moisture[land], 0.70, 0.12)
    scores[:, :, 25][land] = gate_trop[land] * very_humid[land] * gaussian(precip_mm[land], 2200, 500) * gaussian(continentality[land], 0.15, 0.20)

    # --- Cloud forest (narrower, favor oceanic mid-elevation) ---  # <<< tuned
    cloud_mask = submontane_mask
    if np.any(cloud_mask):
        scores[:, :, 26][cloud_mask] = (gate_trop[cloud_mask] + gate_temp[cloud_mask]) \
            * gaussian(continentality[cloud_mask], 0.35, 0.22) \
            * gaussian(moisture[cloud_mask], 0.92, 0.06) \
            * gaussian(precip_mm[cloud_mask], 1800, 420)

    # --- Coastal biomes (stricter) ---  # <<< tuned
    if np.any(coastal_mask):
        warm_coast = coastal_mask & (temp_c > 22)  # was 20
        scores[:, :, 27][warm_coast] = gate_trop[warm_coast] * gaussian(moisture[warm_coast], 0.97, 0.03) * (lowland_mask[warm_coast]).astype(float)
        cool_coast = coastal_mask & (temp_c < 18)  # was 20
        scores[:, :, 29][cool_coast] = (gate_temp[cool_coast] + gate_boreal[cool_coast]) * gaussian(moisture[cool_coast], 0.90, 0.06) * (lowland_mask[cool_coast]).astype(float)

    # --- Wetlands (stricter saturation) ---  # <<< tuned
    wetland_mask = land & (moisture > 0.93)
    if np.any(wetland_mask):
        scores[:, :, 28][wetland_mask] = gaussian(temp_c[wetland_mask], 10, 9) * gaussian(moisture[wetland_mask], 0.965, 0.04)

    # --- apply priors ---
    for k in range(n_biomes):
        if k in PRIORS:
            scores[:, :, k] *= float(PRIORS[k])

     # place this right before: `scores *= 1.0 + 1e-6 ; return scores.astype(np.float32)`
    BALANCE_ENABLED   = True
    BALANCE_STRENGTH  = 0.75   # 0..1: higher = stronger push toward target
    MULT_MIN, MULT_MAX = 0.50, 2.50  # clamp to avoid wild swings
    N_BALANCE_PASSES  = 2      # 1-2 passes is usually enough
    EPS = 1e-12

    if BALANCE_ENABLED:
        land_mask = ~ocean
        # Tiny floor so empty classes can "emerge" if gates allow at all
        scores[land_mask, 1:] += 1e-9

        for _ in range(N_BALANCE_PASSES):
            # Current land-only "soft" shares (sum of scores per biome over land)
            totals = np.array([
                float(scores[:, :, k][land_mask].sum()) for k in range(n_biomes)
            ], dtype=np.float64)
            totals[0] = 0.0  # ignore ocean in balancing
            total_land_sum = totals.sum()
            if total_land_sum <= 0:
                break

            current = totals / (total_land_sum + EPS)

            # --- Target shares ---
            # Uniform across all non-ocean biomes by default:
            target = np.zeros_like(current)
            target[1:] = 1.0 / (n_biomes - 1)

            # Example: custom targets (uncomment and tweak if you want)
            # target = np.zeros_like(current)
            # custom = {13:0.05, 16:0.03, 20:0.04, 21:0.08, 22:0.08}  # ids -> land share
            # remaining = 1.0 - sum(custom.values())
            # spread = remaining / (n_biomes - 1 - len(custom))
            # for k in range(1, n_biomes):
            #     target[k] = custom.get(k, spread)

            # Multipliers: raise low-share, damp high-share (elasticity via strength)
            multipliers = np.ones(n_biomes, dtype=np.float32)
            for k in range(1, n_biomes):
                if totals[k] > 0:
                    m = (target[k] / (current[k] + EPS)) ** BALANCE_STRENGTH
                    multipliers[k] = float(np.clip(m, MULT_MIN, MULT_MAX))
                else:
                    # If currently zero but allowed by gates, give a modest nudge
                    multipliers[k] = 1.5

            # Apply
            for k in range(1, n_biomes):
                scores[:, :, k] *= multipliers[k]

    scores *= 1.0 + 1e-6
    return scores.astype(np.float32)

def apply_probabilistic_mixing(scores: np.ndarray, mixing_radius: int = 2) -> np.ndarray:
    """Smooth biome scores to create gentle ecotones."""
    h, w, n_biomes = scores.shape
    smoothed = np.zeros_like(scores)
    for i in range(n_biomes):
        smoothed[:, :, i] = gaussian_filter(scores[:, :, i], sigma=mixing_radius)
    return smoothed


def assign_biomes_from_scores(
    scores: np.ndarray,
    ocean: np.ndarray,
    use_probabilistic: bool = False,
    random_seed: int = 42,
    return_membership: bool = False,
):
    """Convert biome score volume to discrete biome IDs and RGB map."""
    h, w, n_biomes = scores.shape
    biome_id = np.zeros((h, w), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    membership = None
    if return_membership:
        membership = np.zeros((h, w, n_biomes), dtype=np.float32)

    biome_id[ocean] = 0
    ocean_color = np.array(BIOME_TABLE[0][1], dtype=np.float32)
    rgb[ocean] = ocean_color
    if membership is not None:
        membership[ocean, 0] = 1.0

    land = ~ocean
    if use_probabilistic:
        np.random.seed(random_seed)
        biome_colors = np.zeros((n_biomes, 3), dtype=np.float32)
        for k in range(n_biomes):
            if k in BIOME_TABLE:
                biome_colors[k] = np.array(BIOME_TABLE[k][1], dtype=np.float32)
        land_indices = np.where(land)
        for i, j in zip(land_indices[0], land_indices[1]):
            pixel_scores = scores[i, j, :]
            land_scores = pixel_scores[1:].copy()
            land_scores = np.maximum(land_scores, 0.0)
            total = land_scores.sum()
            if total > 0:
                weights = land_scores / total
                weighted_color = np.zeros(3, dtype=np.float32)
                for biome_idx in range(1, n_biomes):
                    weight = weights[biome_idx - 1]
                    if weight > 0:
                        weighted_color += weight * biome_colors[biome_idx]
                if membership is not None:
                    membership[i, j, 1:] = weights
                rgb[i, j] = weighted_color
                biome_id[i, j] = np.argmax(land_scores) + 1
            else:
                biome_id[i, j] = 0
                rgb[i, j] = [0, 0, 0]
    else:
        max_biome = np.argmax(scores, axis=2)
        biome_id = max_biome.astype(np.uint8)
        for k, (_, color) in BIOME_TABLE.items():
            mask = biome_id == k
            rgb[mask] = np.array(color, dtype=np.float32)
        if membership is not None:
            flat_membership = membership.reshape(-1, n_biomes)
            flat_membership[np.arange(flat_membership.shape[0]), biome_id.reshape(-1)] = 1.0

    unassigned_land = land & ((biome_id == 0) | (np.all(rgb == 0, axis=2)))
    if np.any(unassigned_land):
        assigned_land = land & (biome_id != 0) & np.any(rgb > 0, axis=2)
        if np.any(assigned_land):
            _, (nearest_i, nearest_j) = distance_transform_edt(
                ~assigned_land, return_indices=True
            )
            for i, j in zip(*np.where(unassigned_land)):
                ni, nj = nearest_i[i, j], nearest_j[i, j]
                biome_id[i, j] = biome_id[ni, nj]
                rgb[i, j] = rgb[ni, nj]
        else:
            biome_id[unassigned_land] = 13
            rgb[unassigned_land] = np.array(BIOME_TABLE[13][1], dtype=np.float32)

    biome_id[ocean] = 0
    rgb[ocean] = ocean_color
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    if membership is not None:
        flat_membership = membership.reshape(-1, n_biomes)
        flat_ids = biome_id.reshape(-1).astype(np.int64)
        sums = flat_membership.sum(axis=1)
        zero_mask = sums <= 0
        if np.any(zero_mask):
            flat_membership[zero_mask, :] = 0.0
            flat_membership[zero_mask, flat_ids[zero_mask]] = 1.0
        membership = flat_membership.reshape(h, w, n_biomes)
        membership[ocean] = 0.0
        membership[ocean, 0] = 1.0
        return biome_id, rgb_uint8, membership.astype(np.float32)
    return biome_id, rgb_uint8


def classify_biomes_advanced(
    elev: np.ndarray,
    sea_level_m: float,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    twi: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    tpi: np.ndarray,
    dist_coast_km: np.ndarray,
    lat_deg: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    mixing_radius: int = 3,
    use_probabilistic: bool = False,
    return_membership: bool = False,
):
    """High-level biome classification pipeline."""
    ocean = compute_ocean_mask(elev, elev.min(), elev.max(), sea_level_m)
    continentality = compute_continentality(dist_coast_km / 1000.0, lat_deg)
    moisture = compute_moisture_index(precip_mm, pet_mm, twi)
    wind_exposure = compute_wind_exposure(elev, slope_deg, tpi, wind_u, wind_v)
    aspect_effect = compute_aspect_effect(aspect_deg, lat_deg)
    elev_zones = compute_elevation_zones(elev)

    scores = calculate_biome_scores(
        temp_c,
        precip_mm,
        moisture,
        continentality,
        wind_exposure,
        aspect_effect,
        elev_zones,
        ocean,
    )
    scores = apply_probabilistic_mixing(scores, mixing_radius)
    return assign_biomes_from_scores(
        scores,
        ocean,
        use_probabilistic,
        return_membership=return_membership,
    )
