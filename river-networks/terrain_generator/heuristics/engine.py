"""Terrain heuristics integration utilities for the river-networks application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PyQt5.QtGui import QImage

from .pipeline.climate import latitude_degrees
from .pipeline.engine import TerrainEngine

@dataclass
class HeuristicSettings:
    """User-configurable parameters for heuristic computation."""

    terrain_size_km: float = 0.0
    cellsize: float = 1500.0
    z_min: float = 0.0
    z_max: float = 6000.0
    sea_level_m: float = 0.0
    lapse_rate_c_per_km: float = 6.5
    t_equator_c: float = 30.0
    t_pole_c: float = 0.0
    coast_decay_km: float = 1.75
    orographic_alpha: float = 4.0
    shadow_beta: float = 0.15
    shadow_max_distance_km: float = 400.0
    shadow_decay_km: float = 150.0
    shadow_height_threshold_m: float = 150.0
    shadow_strength: float = 1.0
    precip_lat_pattern: str = "two_bands"
    prevailing_wind_model: str = "three_cell"
    temperature_pattern: str = "polar"
    temperature_gradient_azimuth_deg: float = 0.0
    precip_gradient_azimuth_deg: float = 0.0
    constant_wind_azimuth_deg: float = 0.0
    svf_dirs: int = 16
    svf_radius: float = 100.0
    tpi_radii: Tuple[float, float] = (25.0, 100.0)
    biome_mixing: int = 1
    use_random_biomes: bool = False
    flowacc_texture: Optional[str] = None
    albedo_mode: str = "physical"
    deposition_texture: Optional[str] = None

class HeuristicEngine:
    """Thin wrapper around the standalone TerrainEngine."""

    def __init__(self):
        self._engine = TerrainEngine()
        self._settings = HeuristicSettings()
        self._deposition_map = None
        self._rock_map = None
        self._rock_types: Optional[Tuple[str, ...]] = None
        self._rock_colors: Optional[Tuple[Optional[Tuple[int, int, int]], ...]] = None

    # ------------------------------ helpers ------------------------------
    def _apply_heightmap(self, heightmap: np.ndarray, z_min: float, z_max: float):
        """Inject the terrain heightmap into the underlying engine."""
        if heightmap.ndim != 2:
            raise ValueError("Heightmap must be 2D")

        heightmap = np.asarray(heightmap, dtype=np.float32)
        elev = z_min + (heightmap * (z_max - z_min))

        self._engine.elev = elev.astype(np.float32)
        self._engine.h, self._engine.w = elev.shape
        self._engine.lat1d = latitude_degrees(self._engine.h)
        self._engine._dirty_all()

        if self._deposition_map is not None:
            self.inject_deposition_map(self._deposition_map)
        if self._rock_map is not None:
            self._engine.inject_rock_map(self._rock_map, self._rock_types, self._rock_colors)

    def _apply_settings(self, settings: HeuristicSettings):
        prev = self._settings
        self._settings = settings

        # Update engine parameters in a single call so internal caches are
        # invalidated coherently.
        self._engine.set_settings(
            cellsize=settings.cellsize,
            z_min=settings.z_min,
            z_max=settings.z_max,
            sea_level_m=settings.sea_level_m,
            lapse_rate_c_per_km=settings.lapse_rate_c_per_km,
            t_equator_c=settings.t_equator_c,
            t_pole_c=settings.t_pole_c,
            coast_decay_km=settings.coast_decay_km,
            orographic_alpha=settings.orographic_alpha,
            shadow_beta=settings.shadow_beta,
            shadow_max_distance_km=settings.shadow_max_distance_km,
            shadow_decay_km=settings.shadow_decay_km,
            shadow_height_threshold_m=settings.shadow_height_threshold_m,
            shadow_strength=settings.shadow_strength,
            precip_lat_pattern=settings.precip_lat_pattern,
            prevailing_wind_model=settings.prevailing_wind_model,
            temperature_pattern=settings.temperature_pattern,
            temperature_gradient_azimuth_deg=settings.temperature_gradient_azimuth_deg,
            precip_gradient_azimuth_deg=settings.precip_gradient_azimuth_deg,
            constant_wind_azimuth_deg=settings.constant_wind_azimuth_deg,
            svf_dirs=settings.svf_dirs,
            svf_radius=settings.svf_radius,
            tpi_radii=list(settings.tpi_radii),
            biome_mixing=settings.biome_mixing,
            use_random_biomes=settings.use_random_biomes,
            flowacc_texture=settings.flowacc_texture,
            albedo_mode=settings.albedo_mode,
            deposition_texture=settings.deposition_texture,
        )

    @staticmethod
    def _resolve_cellsize(settings: HeuristicSettings, shape: tuple[int, int]) -> float:
        """Resolve the active horizontal cell size in meters."""
        if settings.terrain_size_km > 0.0:
            resolution = max(float(shape[0]), 1.0)
            return max((settings.terrain_size_km * 1000.0) / resolution, 1e-6)
        return max(float(settings.cellsize), 1e-6)

    def inject_deposition_map(self, deposition_map: np.ndarray):
        """Inject a deposition map from erosion simulation."""
        if deposition_map is not None:
            self._deposition_map = np.asarray(deposition_map, dtype=np.float32)
            # Store in the engine's cache so it can be accessed during computation
            self._engine.cache['deposition'] = self._deposition_map.copy()

    def inject_rock_map(
        self,
        rock_map: np.ndarray,
        rock_types: Optional[Iterable[str]] = None,
        rock_colors: Optional[Iterable[Optional[Iterable[int]]]] = None,
    ):
        """Inject a rock layer index map used for material-aware heuristics."""
        if rock_map is None:
            self._rock_map = None
            self._rock_types = None
            self._rock_colors = None
            self._engine.inject_rock_map(None, None)
            return

        arr = np.asarray(rock_map, dtype=np.int32)
        self._rock_map = np.ascontiguousarray(arr)
        if rock_types is not None:
            self._rock_types = tuple(str(name) for name in rock_types)
        else:
            self._rock_types = None

        normalized_colors: Optional[Tuple[Optional[Tuple[int, int, int]], ...]] = None
        if rock_colors is not None:
            buffer: list[Optional[Tuple[int, int, int]]] = []
            for entry in rock_colors:
                if entry is None:
                    buffer.append(None)
                    continue
                try:
                    components = tuple(int(max(0, min(255, float(c)))) for c in entry[:3])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    buffer.append(None)
                    continue
                buffer.append(components)
            normalized_colors = tuple(buffer)

        self._rock_colors = normalized_colors

        self._engine.inject_rock_map(self._rock_map, self._rock_types, self._rock_colors)

    # --------------------------- public methods --------------------------
    def prepare(self, heightmap: np.ndarray, settings: Optional[HeuristicSettings] = None):
        """Reset the engine with a new heightmap and optional settings."""
        self._engine.reset()
        if settings is None:
            settings = self._settings
        else:
            self._settings = settings

        resolved_cellsize = self._resolve_cellsize(settings, np.asarray(heightmap).shape)
        settings = replace(settings, cellsize=resolved_cellsize)

        self._apply_heightmap(heightmap, settings.z_min, settings.z_max)
        self._apply_settings(settings)

    def compute(
        self,
        selections: Iterable[str],
        cancel_callback=None,
    ) -> Tuple[Dict[str, QImage], Dict[str, np.ndarray]]:
        """Compute the requested heuristic layers.

        Returns a tuple ``(images, arrays)`` that match the TerrainEngine API.
        """
        images_holder: Dict[str, QImage] = {}
        arrays_holder: Dict[str, np.ndarray] = {}
        error: List[str] = []

        finished_event = {"done": False}

        def _on_finished(images: Dict[str, QImage], arrays: Dict[str, np.ndarray]):
            images_holder.update(images)
            arrays_holder.update(arrays)
            finished_event["done"] = True

        def _on_failed(message: str):
            error.append(message)
            finished_event["done"] = True

        self._engine.finished.connect(_on_finished)
        self._engine.failed.connect(_on_failed)

        try:
            self._engine.compute_selected(list(selections), cancel_callback=cancel_callback)
        finally:
            try:
                self._engine.finished.disconnect(_on_finished)
            except TypeError:
                pass
            try:
                self._engine.failed.disconnect(_on_failed)
            except TypeError:
                pass

        if error:
            raise RuntimeError(error[0])

        return images_holder, arrays_holder

    @property
    def qt_engine(self) -> TerrainEngine:
        """Expose the underlying TerrainEngine for signal wiring."""
        return self._engine


def qimage_to_rgba(image: QImage) -> np.ndarray:
    """Convert a ``QImage`` into a contiguous RGBA uint8 numpy array."""
    if image.isNull():
        raise ValueError("Cannot convert a null QImage to an array")
    converted = image.convertToFormat(QImage.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    ptr = converted.bits()
    ptr.setsize(converted.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
    return arr
