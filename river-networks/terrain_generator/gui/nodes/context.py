"""Shared execution context for the node graph."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class NodeGraphContext:
    """Shared state and caches used across all nodes in the graph."""

    def __init__(self):
        self._project_settings_node = None
        self._world_settings_node = None
        self._properties: Dict[str, Any] = {
            "dimension": 1024,
            "seed": 42,
        }
        self.heuristic_cache: Dict[Tuple[str, str], Any] = {}
        self.graph_metadata: Dict[str, Any] = {}

    def clear_runtime_caches(self):
        """Clear caches for graph execution."""
        self.heuristic_cache.clear()

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
        return {
            "cellsize": 1500.0,
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
            "biome_mixing": 1,
            "temperature_pattern": "polar",
            "temperature_gradient_azimuth_deg": 0.0,
            "precip_lat_pattern": "two_bands",
            "precip_gradient_azimuth_deg": 0.0,
            "prevailing_wind_model": "three_cell",
            "constant_wind_azimuth_deg": 25.0,
            "use_random_biomes": False,
            "use_simulated_flow": True,
        }

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
