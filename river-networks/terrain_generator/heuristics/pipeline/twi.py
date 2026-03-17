import numpy as np

__all__ = ["compute_twi"]


def compute_twi(acc: np.ndarray, slope_deg: np.ndarray, cellsize: float) -> np.ndarray:
    """Compute topographic wetness index from accumulation and slope."""
    A = (acc * (cellsize ** 2)).astype(np.float64)
    slope_rad = np.radians(slope_deg).astype(np.float64)
    twi = np.log((A + 1e-8) / (np.tan(slope_rad) + 1e-8))
    return np.where(np.isfinite(twi), twi, 0.0).astype(np.float32)
