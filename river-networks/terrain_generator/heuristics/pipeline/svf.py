import math
import numpy as np

__all__ = ["compute_svf"]


def compute_svf(elev: np.ndarray, cellsize: float, dirs: int = 16, radius_m: float = 100.0) -> np.ndarray:
    """Approximate sky-view factor via radial casting."""
    h, w = elev.shape
    radius_px = max(1, int(round(radius_m / cellsize)))
    angles = np.linspace(0.0, 2.0 * math.pi, num=dirs, endpoint=False)
    svf = np.zeros((h, w), dtype=np.float32)
    steps = []
    for a in angles:
        dx = math.cos(a)
        dy = math.sin(a)
        denom = max(abs(dx), abs(dy), 1e-6)
        sx = dx / denom
        sy = dy / denom
        steps.append((sx, sy))
    x0 = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y0 = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    z0 = elev
    for sx, sy in steps:
        max_ang = np.full((h, w), -np.inf, dtype=np.float32)
        x = x0.copy()
        y = y0.copy()
        for k in range(1, radius_px + 1):
            x += sx
            y += sy
            xi = np.clip(np.round(x).astype(int), 0, w - 1)
            yi = np.clip(np.round(y).astype(int), 0, h - 1)
            dz = elev[yi, xi] - z0
            dist = k * cellsize
            ang = np.arctan2(dz, dist).astype(np.float32)
            max_ang = np.maximum(max_ang, ang)
        max_ang = np.clip(max_ang, 0.0, math.pi / 2.0)
        svf += (np.cos(max_ang) ** 2).astype(np.float32)
    svf /= float(dirs)
    return svf
