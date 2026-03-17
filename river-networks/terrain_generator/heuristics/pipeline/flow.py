import math
import numpy as np

__all__ = [
    "d8_flow_direction",
    "d8_flow_accumulation",
]

_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def _neighbor_indices(i: int, j: int, h: int, w: int):
    for di, dj in _OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            yield ni, nj, di, dj


def d8_flow_direction(elev: np.ndarray, cellsize: float, resolve_pits: str = "carve"):
    """Return downstream neighbor indices for classical D8 routing."""
    h, w = elev.shape
    to_i = np.full((h, w), -1, dtype=np.int32)
    to_j = np.full((h, w), -1, dtype=np.int32)
    sqrt2 = math.sqrt(2.0)
    for i in range(h):
        for j in range(w):
            z = elev[i, j]
            best_slope = -np.inf
            best_ni, best_nj = -1, -1
            best_uphill_rise = np.inf
            for ni, nj, di, dj in _neighbor_indices(i, j, h, w):
                dist = cellsize * (sqrt2 if (di != 0 and dj != 0) else 1.0)
                dz = z - elev[ni, nj]
                slope = dz / dist
                if slope > best_slope:
                    best_slope = slope
                    best_ni, best_nj = ni, nj
                if dz <= 0 and (-dz) < best_uphill_rise:
                    best_uphill_rise = -dz
            if best_slope > 0:
                to_i[i, j] = best_ni
                to_j[i, j] = best_nj
            elif resolve_pits == "carve" and best_ni >= 0:
                to_i[i, j] = best_ni
                to_j[i, j] = best_nj
    return to_i, to_j


def d8_flow_accumulation(elev: np.ndarray, cellsize: float, resolve_pits: str = "carve") -> np.ndarray:
    """Return contributing area (in cells) using D8 routing."""
    h, w = elev.shape
    to_i, to_j = d8_flow_direction(elev, cellsize, resolve_pits=resolve_pits)
    acc = np.ones((h, w), dtype=np.float64)
    flat_idx = np.arange(h * w)
    order = np.argsort(elev.flatten())[::-1]
    ii = (flat_idx // w).astype(np.int32)
    jj = (flat_idx % w).astype(np.int32)
    ii = ii[order]
    jj = jj[order]
    for i, j in zip(ii, jj):
        ti, tj = to_i[i, j], to_j[i, j]
        if ti >= 0:
            acc[ti, tj] += acc[i, j]
    return acc.astype(np.float32)
