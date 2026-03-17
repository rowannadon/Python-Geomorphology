import numpy as np
from scipy.ndimage import convolve

__all__ = ["compute_laplacian_curvature"]


def compute_laplacian_curvature(elev: np.ndarray, cellsize: float) -> np.ndarray:
    """Compute Laplacian curvature (negative laplacian scaled by cellsize)."""
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve(elev, lap_kernel, mode="reflect") / (cellsize ** 2)
    return (-lap).astype(np.float32)
