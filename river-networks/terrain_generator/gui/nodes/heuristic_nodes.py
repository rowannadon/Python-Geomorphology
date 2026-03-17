"""Heuristic map nodes built on top of the shared heuristic engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...heuristics import HeuristicEngine, HeuristicSettings, qimage_to_rgba
from .base_nodes import TerrainBaseNode, _parse_float, _parse_int
from .contracts import (
    HeightfieldData,
    MapOverlayData,
    PORT_TYPE_HEIGHTFIELD,
    PORT_TYPE_MAP_OVERLAY,
    PORT_TYPE_TERRAIN_BUNDLE,
    TerrainBundleData,
    overlay_from_deposition,
    overlay_from_rgb,
    overlay_from_scalar,
    payload_identity_hash,
)


@dataclass(frozen=True)
class HeuristicSpec:
    """Description of a single heuristic node type."""

    node_name: str
    selection_key: str
    array_key: str
    image_key: str
    overlay_kind: str = "scalar"
    default_cellsize: float = 0.0
    uses_tpi_radius: bool = False
    needs_deposition: bool = False
    needs_rock_map: bool = False
    supports_flow_override: bool = False
    uses_temperature_settings: bool = False
    uses_precipitation_settings: bool = False


def _selection_for_spec(spec: HeuristicSpec, radius_m: float) -> str:
    if spec.uses_tpi_radius:
        return f"tpi@{float(radius_m)}"
    return spec.selection_key


class HeuristicMapNode(TerrainBaseNode):
    """Base class for a typed heuristic-map node."""

    SPEC = HeuristicSpec(
        node_name="Heuristic",
        selection_key="slope",
        array_key="slope_deg",
        image_key="slope_deg",
    )
    INPUT_TYPES = {
        "terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,),
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "flow_override": (PORT_TYPE_HEIGHTFIELD,),
        "deposition_map": (PORT_TYPE_HEIGHTFIELD,),
        "rock_map": (PORT_TYPE_HEIGHTFIELD,),
    }
    OUTPUT_TYPES = {"map_overlay": (PORT_TYPE_MAP_OVERLAY,)}

    def __init__(self):
        super().__init__()
        self.NODE_NAME = self.SPEC.node_name
        self._base_name = self.SPEC.node_name
        self.set_name(self.SPEC.node_name)
        self.set_color(90, 120, 170)
        self.add_input("terrain_bundle", color=(140, 200, 210))
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_input("flow_override", color=(140, 180, 220))
        self.add_input("deposition_map", color=(150, 200, 150))
        self.add_input("rock_map", color=(150, 200, 150))
        self.add_output("map_overlay", color=(180, 180, 120))
        self.add_text_input("cellsize_override", "Cell Size Override (m)", text=str(self.SPEC.default_cellsize))
        if self.SPEC.uses_tpi_radius:
            self.add_text_input("radius_m", "Radius (m)", text="25.0")
        if self.SPEC.uses_temperature_settings:
            self.add_combo_menu("temperature_pattern", "Temperature Pattern", items=["polar", "equatorial", "gradient"])
            self.set_property("temperature_pattern", "polar")
        if self.SPEC.uses_precipitation_settings:
            self.add_combo_menu("precip_lat_pattern", "Precipitation Pattern", items=["two_bands", "single_band", "uniform", "gradient"])
            self.set_property("precip_lat_pattern", "two_bands")
            self.add_combo_menu("prevailing_wind_model", "Wind Model", items=["three_cell", "constant"])
            self.set_property("prevailing_wind_model", "three_cell")

    def _resolve_sources(self) -> Tuple[HeightfieldData, Optional[TerrainBundleData]]:
        bundle = self.get_input_data("terrain_bundle", required=False, expected_types=(PORT_TYPE_TERRAIN_BUNDLE,))
        heightfield = self.get_input_heightfield("heightfield", required=False)
        if bundle is not None:
            return bundle.heightfield, bundle
        if heightfield is not None:
            return heightfield, None
        raise ValueError(f"{self._base_name} requires either a terrain bundle or heightfield input.")

    def _build_settings(self) -> Dict[str, Any]:
        world_settings = HeuristicSettings().__dict__.copy()
        world_settings.update(self.context.get_world_settings())
        cellsize_override = _parse_float(self.get_property("cellsize_override"), 0.0)
        if cellsize_override > 0.0:
            world_settings["cellsize"] = cellsize_override
        if self.SPEC.uses_temperature_settings:
            world_settings["temperature_pattern"] = self.get_property("temperature_pattern") or world_settings["temperature_pattern"]
        if self.SPEC.uses_precipitation_settings:
            world_settings["precip_lat_pattern"] = self.get_property("precip_lat_pattern") or world_settings["precip_lat_pattern"]
            world_settings["prevailing_wind_model"] = self.get_property("prevailing_wind_model") or world_settings["prevailing_wind_model"]
        world_settings.pop("use_simulated_flow", None)
        return world_settings

    def _resolve_optional_maps(
        self,
        bundle: Optional[TerrainBundleData],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[str, ...]], Optional[Tuple[Optional[Tuple[int, int, int]], ...]]]:
        flow_override = self.get_input_heightfield("flow_override", required=False)
        deposition_map = self.get_input_heightfield("deposition_map", required=False)
        rock_map = self.get_input_heightfield("rock_map", required=False)

        flow_array = None
        if flow_override is not None:
            flow_array = np.asarray(flow_override.array, dtype=np.float32)
        elif bundle is not None and bundle.river_volume is not None and self.context.get_world_settings().get("use_simulated_flow", True):
            flow_array = np.asarray(bundle.river_volume, dtype=np.float32)

        deposition_array = None
        if deposition_map is not None:
            deposition_array = np.asarray(deposition_map.array, dtype=np.float32)
        elif bundle is not None and bundle.deposition_map is not None:
            deposition_array = np.asarray(bundle.deposition_map, dtype=np.float32)

        rock_array = None
        rock_types = None
        rock_colors = None
        if rock_map is not None:
            rock_array = np.asarray(np.rint(rock_map.array), dtype=np.int32)
        elif bundle is not None and bundle.rock_map is not None:
            rock_array = np.asarray(bundle.rock_map, dtype=np.int32)
            rock_types = bundle.rock_types or None
            rock_colors = bundle.rock_colors or None

        return flow_array, deposition_array, rock_array, rock_types, rock_colors

    def _cache_key(
        self,
        spec: HeuristicSpec,
        selection_key: str,
        heightfield: HeightfieldData,
        settings: Dict[str, Any],
        flow_override: Optional[np.ndarray],
        deposition_map: Optional[np.ndarray],
        rock_map: Optional[np.ndarray],
    ) -> Tuple[str, str]:
        extras = [
            spec.array_key,
            selection_key,
            repr(sorted(settings.items())),
            payload_identity_hash(heightfield),
        ]
        for arr in (flow_override, deposition_map, rock_map):
            if arr is None:
                extras.append("none")
            else:
                extras.append(f"{arr.shape}:{hash(arr.tobytes())}")
        return ("heuristic", "|".join(extras))

    def current_spec(self) -> HeuristicSpec:
        return self.SPEC

    def _compute_overlay(self) -> MapOverlayData:
        spec = self.current_spec()
        heightfield, bundle = self._resolve_sources()
        settings = self._build_settings()
        radius_m = _parse_float(self.get_property("radius_m"), 25.0) if spec.uses_tpi_radius else 0.0
        selection_key = _selection_for_spec(spec, radius_m)
        flow_override, deposition_map, rock_map, rock_types, rock_colors = self._resolve_optional_maps(bundle)
        cache_key = self._cache_key(spec, selection_key, heightfield, settings, flow_override, deposition_map, rock_map)
        if cache_key in self.context.heuristic_cache:
            return self.context.heuristic_cache[cache_key]

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {spec.node_name}")
        engine.prepare(heightfield.array, engine_settings)
        if flow_override is not None:
            engine.qt_engine.cache["acc"] = np.asarray(flow_override, dtype=np.float32).copy()
            engine.qt_engine.params["flowacc_texture"] = None
        if deposition_map is not None:
            engine.inject_deposition_map(deposition_map)
        if spec.needs_rock_map and rock_map is not None:
            engine.inject_rock_map(rock_map, rock_types, rock_colors)
        self.emit_progress(0.5, f"Computing {spec.node_name}")
        images, arrays = engine.compute([selection_key])
        if spec.array_key not in arrays:
            raise ValueError(f"Heuristic result '{spec.array_key}' missing for {spec.node_name}.")
        rgba = qimage_to_rgba(images[spec.image_key])
        array = arrays[spec.array_key]
        land_mask = bundle.land_mask.array if bundle is not None and bundle.land_mask is not None else None

        if spec.overlay_kind == "rgb":
            overlay = overlay_from_rgb(spec.image_key, spec.node_name, array, heightfield)
        elif spec.overlay_kind == "deposition":
            overlay = overlay_from_deposition(spec.image_key, spec.node_name, array, heightfield, land_mask=land_mask)
        else:
            overlay = overlay_from_scalar(spec.image_key, spec.node_name, array, heightfield, land_mask=land_mask)
        overlay = MapOverlayData(
            key=overlay.key,
            display_name=overlay.display_name,
            array=overlay.array,
            rgba=rgba,
            base_heightfield=overlay.base_heightfield,
            overlay_kind=overlay.overlay_kind,
            metadata={"selection_key": selection_key, "array_key": spec.array_key},
        )
        self.context.heuristic_cache[cache_key] = overlay
        return overlay

    def execute(self):
        overlay = self._compute_overlay()
        self.emit_progress(1.0, f"{self._base_name} ready")
        self.set_output_data(overlay)
        self.signals.execution_finished.emit(self)
        return overlay


_HEURISTIC_SPECS: Dict[str, HeuristicSpec] = {
    "SlopeHeuristicNode": HeuristicSpec("Slope", "slope", "slope_deg", "slope_deg"),
    "AspectHeuristicNode": HeuristicSpec("Aspect", "aspect", "aspect_deg", "aspect_deg"),
    "NormalHeuristicNode": HeuristicSpec("Normals", "normal", "normal", "normal", overlay_kind="rgb"),
    "CurvatureHeuristicNode": HeuristicSpec("Curvature", "curvature", "curvature", "curvature"),
    "TPIHeuristicNode": HeuristicSpec("TPI", "tpi", "tpi_25m", "tpi_25m", uses_tpi_radius=True),
    "FlowAccumulationHeuristicNode": HeuristicSpec("Flow Accumulation", "flowacc", "flowacc", "flowacc_log", supports_flow_override=True),
    "TWIHeuristicNode": HeuristicSpec("TWI", "twi", "twi", "twi"),
    "SVFHeuristicNode": HeuristicSpec("Sky View Factor", "svf", "svf", "svf"),
    "TemperatureHeuristicNode": HeuristicSpec("Temperature", "climate", "temp_c", "temp_c", uses_temperature_settings=True),
    "PrecipitationHeuristicNode": HeuristicSpec("Precipitation", "climate", "precip_mm", "precip_mm", uses_precipitation_settings=True),
    "PETHeuristicNode": HeuristicSpec("PET", "climate", "PET", "PET", uses_temperature_settings=True),
    "AETHeuristicNode": HeuristicSpec("AET", "climate", "AET", "AET", uses_temperature_settings=True, uses_precipitation_settings=True),
    "AridityHeuristicNode": HeuristicSpec("Aridity", "climate", "AI", "AI", uses_temperature_settings=True, uses_precipitation_settings=True),
    "BiomeHeuristicNode": HeuristicSpec("Biome", "biome", "biome_rgb", "biome_map", overlay_kind="rgb", uses_temperature_settings=True, uses_precipitation_settings=True),
    "AlbedoHeuristicNode": HeuristicSpec("Albedo", "albedo", "albedo_rgb", "terrain_albedo", overlay_kind="rgb", needs_deposition=True, needs_rock_map=True, uses_temperature_settings=True, uses_precipitation_settings=True),
    "ContinuousAlbedoHeuristicNode": HeuristicSpec("Continuous Albedo", "albedo_continuous", "albedo_continuous_rgb", "terrain_albedo_continuous", overlay_kind="rgb", uses_temperature_settings=True, uses_precipitation_settings=True),
    "FoliageColorHeuristicNode": HeuristicSpec("Foliage Color", "foliage", "foliage_rgb", "foliage_color", overlay_kind="rgb", uses_temperature_settings=True, uses_precipitation_settings=True),
    "ForestDensityHeuristicNode": HeuristicSpec("Forest Density", "forest_density", "forest_density", "forest_density", uses_temperature_settings=True, uses_precipitation_settings=True),
    "GroundcoverDensityHeuristicNode": HeuristicSpec("Groundcover Density", "groundcover_density", "groundcover_density", "groundcover_density", uses_temperature_settings=True, uses_precipitation_settings=True),
}


def _build_heuristic_node_class(class_name: str, spec: HeuristicSpec):
    class _Node(HeuristicMapNode):
        NODE_NAME = spec.node_name
        SPEC = spec

        def __init__(self):
            super().__init__()

        def _build_settings(self) -> Dict[str, Any]:
            settings = super()._build_settings()
            if spec.uses_tpi_radius:
                radius = _parse_float(self.get_property("radius_m"), 25.0)
                settings["tpi_radii"] = (radius,)
            return settings

        def current_spec(self) -> HeuristicSpec:
            if spec.uses_tpi_radius:
                radius = _parse_float(self.get_property("radius_m"), 25.0)
                return HeuristicSpec(
                    node_name=spec.node_name,
                    selection_key=spec.selection_key,
                    array_key=f"tpi_{int(radius)}m",
                    image_key=f"tpi_{int(radius)}m",
                    overlay_kind=spec.overlay_kind,
                    default_cellsize=spec.default_cellsize,
                    uses_tpi_radius=True,
                )
            return spec

    _Node.__name__ = class_name
    return _Node


globals().update({name: _build_heuristic_node_class(name, spec) for name, spec in _HEURISTIC_SPECS.items()})

__all__ = list(_HEURISTIC_SPECS.keys()) + ["HeuristicMapNode", "HeuristicSpec"]
