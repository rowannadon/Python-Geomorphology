import numpy as np
from scipy.ndimage import binary_propagation, convolve, generate_binary_structure

__all__ = ["compute_ocean_mask", "compute_coastline_mask"]


def compute_ocean_mask(elev_m: np.ndarray, z_min: float, z_max: float, sea_level_m: float) -> np.ndarray:
    """Flood-fill from edges below sea level to detect oceans."""
    h, w = elev_m.shape
    low = elev_m <= sea_level_m
    seed = np.zeros((h, w), dtype=bool)
    seed[0, :] = low[0, :]
    seed[-1, :] = low[-1, :]
    seed[:, 0] = low[:, 0]
    seed[:, -1] = low[:, -1]
    structure = generate_binary_structure(2, 1)
    ocean = binary_propagation(seed, mask=low, structure=structure)
    return ocean


def compute_coastline_mask(ocean: np.ndarray) -> np.ndarray:
    """Return boolean mask highlighting ocean cells adjacent to land."""
    land = ~ocean
    kernel = np.ones((3, 3), dtype=np.int32)
    land_n = convolve(land.astype(np.int32), kernel, mode="nearest")
    coastline = ocean & (land_n > 0)
    return coastline
