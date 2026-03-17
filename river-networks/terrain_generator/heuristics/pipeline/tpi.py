import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["compute_tpi"]


def compute_tpi(elev: np.ndarray, radius_px: int) -> np.ndarray:
    """Topographic position index at specified radius in pixels."""
    if radius_px < 1:
        return np.zeros_like(elev, dtype=np.float32)
    size = 2 * radius_px + 1
    mean = uniform_filter(elev, size=size, mode="reflect")
    return (elev - mean).astype(np.float32)
