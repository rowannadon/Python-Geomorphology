"""Shared execution context for the node graph."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple


class NodeExecutionCancelled(RuntimeError):
    """Raised when an in-flight node execution is cancelled."""


class ExecutionCancellationToken:
    """Thread-safe cooperative cancellation token for pipeline runs."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self._event.is_set():
            raise NodeExecutionCancelled("Execution cancelled.")


class NodeGraphContext:
    """Shared state and caches used across all nodes in the graph."""

    DEFAULT_CELL_SIZE_M = 1500.0

    def __init__(self):
        self._project_settings_node = None
        self._world_settings_node = None
        self._heuristic_cache_lock = threading.RLock()
        self._properties: Dict[str, Any] = {
            "dimension": 1024,
            "seed": 42,
        }
        self.heuristic_cache: Dict[Tuple[str, str], Any] = {}
        self.graph_metadata: Dict[str, Any] = {}

    def clear_runtime_caches(self):
        """Clear caches for graph execution."""
        with self._heuristic_cache_lock:
            self.heuristic_cache.clear()

    def get_cached_heuristic(self, key: Tuple[str, str]):
        with self._heuristic_cache_lock:
            return self.heuristic_cache.get(key)

    def set_cached_heuristic(self, key: Tuple[str, str], value: Any):
        with self._heuristic_cache_lock:
            self.heuristic_cache[key] = value

    def set_project_settings_node(self, node):
        self._project_settings_node = node

    def set_world_settings_node(self, node):
        self._world_settings_node = node

    def get_project_settings(self) -> Dict[str, Any]:
        if self._project_settings_node is not None:
            return self._project_settings_node.collect_settings()
        return dict(self._properties)

    def get_world_settings(self) -> Dict[str, Any]:
        if self._world_settings_node is not None:
            return self._world_settings_node.collect_settings()
        resolution = self.get_resolution()
        terrain_size_km = (self.DEFAULT_CELL_SIZE_M * resolution) / 1000.0
        return {
            "terrain_size_km": terrain_size_km,
            "cellsize": self.DEFAULT_CELL_SIZE_M,
            "z_min": 0.0,
            "z_max": 6000.0,
            "sea_level_m": 0.0,
            "lapse_rate_c_per_km": 6.5,
            "t_equator_c": 30.0,
            "t_pole_c": 0.0,
            "coast_decay_km": 1.75,
            "orographic_alpha": 4.0,
            "shadow_max_distance_km": 400.0,
            "shadow_decay_km": 150.0,
            "shadow_height_threshold_m": 150.0,
            "shadow_strength": 1.0,
            "biome_mixing": 1.5,
            "temperature_pattern": "polar",
            "temperature_gradient_azimuth_deg": 0.0,
            "precip_lat_pattern": "two_bands",
            "precip_gradient_azimuth_deg": 0.0,
            "prevailing_wind_model": "three_cell",
            "constant_wind_azimuth_deg": 25.0,
            "use_random_biomes": False,
            "use_simulated_flow": True,
        }

    def get_terrain_size_km(self) -> float:
        settings = self.get_world_settings()
        try:
            terrain_size_km = float(settings.get("terrain_size_km", 0.0))
        except (TypeError, ValueError):
            terrain_size_km = 0.0
        if terrain_size_km > 0.0:
            return terrain_size_km
        resolution = self.get_resolution()
        try:
            cellsize = float(settings.get("cellsize", self.DEFAULT_CELL_SIZE_M))
        except (TypeError, ValueError):
            cellsize = self.DEFAULT_CELL_SIZE_M
        return max(cellsize, 1e-6) * resolution / 1000.0

    def get_cellsize_m(self, resolution: Optional[int] = None) -> float:
        if resolution is None:
            resolution = self.get_resolution()
        return max(self.get_terrain_size_km() * 1000.0 / max(int(resolution), 1), 1e-6)

    def get_cellsize_km(self, resolution: Optional[int] = None) -> float:
        if resolution is None:
            resolution = self.get_resolution()
        return max(self.get_terrain_size_km() / max(int(resolution), 1), 1e-9)

    def get_resolution(self) -> int:
        settings = self.get_project_settings()
        try:
            return int(settings.get("dimension", 1024))
        except (TypeError, ValueError):
            return 1024

    def get_seed(self) -> int:
        settings = self.get_project_settings()
        try:
            return int(settings.get("seed", 42))
        except (TypeError, ValueError):
            return 42

    def get_property(self, name: str, default: Any = None) -> Any:
        if name == "dimension":
            return self.get_resolution()
        if name == "seed":
            return self.get_seed()
        return self._properties.get(name, default)

    def set_property(self, name: str, value: Any):
        self._properties[name] = value


_global_context = NodeGraphContext()


def get_global_context() -> NodeGraphContext:
    """Return the global node editor context."""
    return _global_context
