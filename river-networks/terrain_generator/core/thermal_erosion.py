"""Simple relaxation-based thermal erosion for raster heightfields."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(func):
            return func
        return wrap


_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@njit(cache=True)
def _thermal_iteration(
    heightmap: np.ndarray,
    active_mask: np.ndarray,
    talus: float,
    strength: float,
    activity: np.ndarray,
) -> float:
    """Move material downhill where local relief exceeds the talus threshold."""
    h, w = heightmap.shape
    delta = np.zeros_like(heightmap)
    moved_total = 0.0

    for y in range(h):
        for x in range(w):
            if not active_mask[y, x]:
                continue

            center = heightmap[y, x]
            total_excess = 0.0
            max_excess = 0.0
            excesses = np.zeros(8, dtype=np.float64)

            for idx, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
                ny = y + dy
                nx = x + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if not active_mask[ny, nx]:
                    continue

                diff = center - heightmap[ny, nx]
                if diff <= talus:
                    continue

                excess = diff - talus
                excesses[idx] = excess
                total_excess += excess
                if excess > max_excess:
                    max_excess = excess

            if total_excess <= 0.0 or max_excess <= 0.0:
                continue

            movable = max_excess * strength
            if movable <= 0.0:
                continue
            if movable > center:
                movable = center
            if movable <= 0.0:
                continue

            delta[y, x] -= movable
            weight_scale = movable / total_excess
            for idx, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
                excess = excesses[idx]
                if excess <= 0.0:
                    continue
                ny = y + dy
                nx = x + dx
                delta[ny, nx] += excess * weight_scale

            moved_total += movable

    for y in range(h):
        for x in range(w):
            change = delta[y, x]
            if change != 0.0:
                activity[y, x] += abs(change)
            heightmap[y, x] += change
    return moved_total


class ThermalErosion:
    """Iterative thermal erosion using local slope relaxation."""

    def __init__(
        self,
        iterations: int = 30,
        talus: float = 0.02,
        strength: float = 0.5,
    ):
        self.iterations = max(1, int(iterations))
        self.talus = max(0.0, float(talus))
        self.strength = min(max(float(strength), 0.0), 1.0)

    def erode(
        self,
        heightmap: np.ndarray,
        *,
        land_mask: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the relaxed heightfield, signed change map, and cumulative talus activity."""
        terrain = np.asarray(heightmap, dtype=np.float64).copy()
        initial = terrain.copy()
        activity = np.zeros(terrain.shape, dtype=np.float64)

        if land_mask is None:
            active_mask = np.ones(terrain.shape, dtype=np.bool_)
        else:
            active_mask = np.asarray(land_mask, dtype=np.bool_)
            if active_mask.shape != terrain.shape:
                raise ValueError(
                    f"Land mask shape {active_mask.shape} does not match terrain shape {terrain.shape}."
                )
            active_mask = np.ascontiguousarray(active_mask)

        terrain = np.ascontiguousarray(terrain)
        if progress_callback:
            progress_callback(10, "Thermal erosion: preparing relaxation...")

        for step in range(self.iterations):
            if cancel_check and cancel_check():
                raise RuntimeError("Execution cancelled.")

            moved = _thermal_iteration(terrain, active_mask, self.talus, self.strength, activity)

            if progress_callback:
                progress = int(10 + ((step + 1) / self.iterations) * 85)
                progress_callback(progress, f"Thermal erosion: pass {step + 1}/{self.iterations}")

            if moved <= 1e-12:
                break

        return (
            terrain.astype(np.float32),
            (terrain - initial).astype(np.float32),
            activity.astype(np.float32),
        )
