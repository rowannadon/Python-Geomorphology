import numpy as np
from scipy.ndimage import sobel

__all__ = [
    "compute_slope_aspect",
    "compute_gradients",
    "compute_normal_from_grad",
    "compute_normals",
]


def compute_slope_aspect(elev: np.ndarray, cellsize: float):
    """Return slope (deg) and aspect (deg) arrays from elevations."""
    dzdx = sobel(elev, axis=1, mode="reflect") / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode="reflect") / (8.0 * cellsize)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)
    with np.errstate(invalid="ignore"):
        aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0.0, 360.0 + aspect_deg, aspect_deg)
    flat = slope_deg < 1e-3
    aspect_deg[flat] = 0.0
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_gradients(elev: np.ndarray, cellsize: float):
    """Return dz/dx and dz/dy gradients."""
    dzdx = sobel(elev, axis=1, mode="reflect") / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode="reflect") / (8.0 * cellsize)
    return dzdx.astype(np.float32), dzdy.astype(np.float32)


def compute_normal_from_grad(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    """Convert gradients to unit normals."""
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= length
    ny /= length
    nz /= length
    return np.dstack([nx, ny, nz]).astype(np.float32)


def compute_normals(elev: np.ndarray, cellsize: float) -> np.ndarray:
    """Convenience helper computing gradients then normals."""
    dzdx, dzdy = compute_gradients(elev, cellsize)
    return compute_normal_from_grad(dzdx, dzdy)
