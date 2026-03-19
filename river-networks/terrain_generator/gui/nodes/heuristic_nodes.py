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
    dependencies: Tuple["HeuristicDependency", ...] = ()


@dataclass(frozen=True)
class HeuristicDependency:
    """A required overlay input used to seed downstream heuristic computation."""

    port_name: str
    cache_key: str
    accepted_array_keys: Tuple[str, ...]


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
        for dependency in self.SPEC.dependencies:
            self.add_input(dependency.port_name, color=(180, 180, 120))
        self.add_output("map_overlay", color=(180, 180, 120))
        self.add_text_input("cellsize_override", "Cell Size Override (m)", text=str(self.SPEC.default_cellsize))
        if self.SPEC.uses_tpi_radius:
            self.add_text_input("radius_m", "Radius (m)", text="25.0")
        if self.SPEC.selection_key == "svf":
            self.add_text_input("svf_dirs", "SVF Directions", text="16")
            self.add_text_input("svf_radius", "SVF Radius (m)", text="100.0")

    def _resolve_sources(self) -> Tuple[HeightfieldData, Optional[TerrainBundleData]]:
        bundle = self.get_input_data("terrain_bundle", required=False, expected_types=(PORT_TYPE_TERRAIN_BUNDLE,))
        heightfield = self.get_input_heightfield("heightfield", required=False)
        if bundle is not None:
            return bundle.heightfield, bundle
        if heightfield is not None:
            return heightfield, None
        raise ValueError(f"{self._base_name} requires either a terrain bundle or heightfield input.")

    def _build_settings(self, dependency_overlays: Optional[Dict[str, MapOverlayData]] = None) -> Dict[str, Any]:
        world_settings = HeuristicSettings().__dict__.copy()
        world_settings.update(self.context.get_world_settings())
        cellsize_override = _parse_float(self.get_property("cellsize_override"), 0.0)
        if cellsize_override > 0.0:
            world_settings["terrain_size_km"] = 0.0
            world_settings["cellsize"] = cellsize_override
        if self.SPEC.selection_key == "svf":
            world_settings["svf_dirs"] = max(1, _parse_int(self.get_property("svf_dirs"), 16))
            world_settings["svf_radius"] = _parse_float(self.get_property("svf_radius"), 100.0)
        world_settings.pop("use_simulated_flow", None)
        return world_settings

    @staticmethod
    def _overlay_array_key(overlay: MapOverlayData) -> str:
        metadata_key = overlay.metadata.get("array_key")
        if metadata_key:
            return str(metadata_key)
        return str(overlay.key)

    def _resolve_dependency_overlays(self, heightfield: HeightfieldData) -> Dict[str, MapOverlayData]:
        overlays: Dict[str, MapOverlayData] = {}
        for dependency in self.SPEC.dependencies:
            overlay = self.get_input_overlay(dependency.port_name, required=True)
            if overlay is None:
                continue
            array_key = self._overlay_array_key(overlay)
            if dependency.accepted_array_keys and array_key not in dependency.accepted_array_keys:
                accepted = ", ".join(dependency.accepted_array_keys)
                raise ValueError(
                    f"{self._base_name} input '{dependency.port_name}' expects one of [{accepted}], received '{array_key}'."
                )
            if overlay.base_heightfield.shape != heightfield.shape:
                raise ValueError(
                    f"{self._base_name} input '{dependency.port_name}' shape {overlay.base_heightfield.shape} "
                    f"does not match terrain shape {heightfield.shape}."
                )
            overlays[dependency.port_name] = overlay
        return overlays

    def _seed_engine_from_dependencies(
        self,
        engine: HeuristicEngine,
        dependency_overlays: Dict[str, MapOverlayData],
    ):
        for dependency in self.SPEC.dependencies:
            overlay = dependency_overlays.get(dependency.port_name)
            if overlay is None:
                continue
            array = np.asarray(overlay.array, dtype=np.float32)
            engine.qt_engine.cache[dependency.cache_key] = array.copy()
            if dependency.cache_key == "acc":
                engine.qt_engine.params["flowacc_texture"] = None

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
        dependency_overlays: Dict[str, MapOverlayData],
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
        for dependency in self.SPEC.dependencies:
            overlay = dependency_overlays.get(dependency.port_name)
            extras.append(
                "none" if overlay is None else f"{dependency.port_name}:{payload_identity_hash(overlay)}"
            )
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
        dependency_overlays = self._resolve_dependency_overlays(heightfield)
        settings = self._build_settings(dependency_overlays)
        radius_m = _parse_float(self.get_property("radius_m"), 25.0) if spec.uses_tpi_radius else 0.0
        selection_key = _selection_for_spec(spec, radius_m)
        flow_override, deposition_map, rock_map, rock_types, rock_colors = self._resolve_optional_maps(bundle)
        cache_key = self._cache_key(spec, selection_key, heightfield, settings, dependency_overlays, flow_override, deposition_map, rock_map)
        if cache_key in self.context.heuristic_cache:
            return self.context.heuristic_cache[cache_key]

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {spec.node_name}")
        engine.prepare(heightfield.array, engine_settings)
        self._seed_engine_from_dependencies(engine, dependency_overlays)
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
            metadata={
                "selection_key": selection_key,
                "array_key": spec.array_key,
                "source_settings": dict(settings),
            },
        )
        self.context.heuristic_cache[cache_key] = overlay
        return overlay

    def execute(self):
        overlay = self._compute_overlay()
        self.emit_progress(1.0, f"{self._base_name} ready")
        self.set_output_data(overlay)
        return overlay


_FLOWACC_DEPENDENCY = HeuristicDependency("flowacc", "acc", ("flowacc",))
_TEMPERATURE_DEPENDENCY = HeuristicDependency("temperature", "temp_c", ("temp_c",))
_PRECIPITATION_DEPENDENCY = HeuristicDependency("precipitation", "P_mm", ("precip_mm",))
_PET_DEPENDENCY = HeuristicDependency("pet", "PET", ("PET",))
_TWI_DEPENDENCY = HeuristicDependency("twi", "twi", ("twi",))
_CLIMATE_STACK_DEPENDENCIES = (
    _TEMPERATURE_DEPENDENCY,
    _PRECIPITATION_DEPENDENCY,
    _PET_DEPENDENCY,
    _TWI_DEPENDENCY,
)


_HEURISTIC_SPECS: Dict[str, HeuristicSpec] = {
    "SlopeHeuristicNode": HeuristicSpec("Slope", "slope", "slope_deg", "slope_deg"),
    "AspectHeuristicNode": HeuristicSpec("Aspect", "aspect", "aspect_deg", "aspect_deg"),
    "NormalHeuristicNode": HeuristicSpec("Normals", "normal", "normal", "normal", overlay_kind="rgb"),
    "CurvatureHeuristicNode": HeuristicSpec("Curvature", "curvature", "curvature", "curvature"),
    "TPIHeuristicNode": HeuristicSpec("TPI", "tpi", "tpi_25m", "tpi_25m", uses_tpi_radius=True),
    "FlowAccumulationHeuristicNode": HeuristicSpec("Flow Accumulation", "flowacc", "flowacc", "flowacc_log", supports_flow_override=True),
    "TWIHeuristicNode": HeuristicSpec("TWI", "twi", "twi", "twi", dependencies=(_FLOWACC_DEPENDENCY,)),
    "SVFHeuristicNode": HeuristicSpec("Sky View Factor", "svf", "svf", "svf"),
    "TemperatureHeuristicNode": HeuristicSpec("Temperature", "climate", "temp_c", "temp_c", uses_temperature_settings=True),
    "PrecipitationHeuristicNode": HeuristicSpec("Precipitation", "climate", "precip_mm", "precip_mm", uses_precipitation_settings=True),
    "PETHeuristicNode": HeuristicSpec("PET", "climate", "PET", "PET", dependencies=(_TEMPERATURE_DEPENDENCY,)),
    "AETHeuristicNode": HeuristicSpec("AET", "climate", "AET", "AET", dependencies=(_PRECIPITATION_DEPENDENCY, _PET_DEPENDENCY)),
    "AridityHeuristicNode": HeuristicSpec("Aridity", "climate", "AI", "AI", dependencies=(_PRECIPITATION_DEPENDENCY, _PET_DEPENDENCY)),
    "BiomeHeuristicNode": HeuristicSpec("Biome", "biome", "biome_rgb", "biome_map", overlay_kind="rgb", dependencies=_CLIMATE_STACK_DEPENDENCIES),
    "AlbedoHeuristicNode": HeuristicSpec("Albedo", "albedo", "albedo_rgb", "terrain_albedo", overlay_kind="rgb", needs_deposition=True, needs_rock_map=True, dependencies=_CLIMATE_STACK_DEPENDENCIES),
    "ContinuousAlbedoHeuristicNode": HeuristicSpec("Continuous Albedo", "albedo_continuous", "albedo_continuous_rgb", "terrain_albedo_continuous", overlay_kind="rgb", dependencies=_CLIMATE_STACK_DEPENDENCIES),
    "FoliageColorHeuristicNode": HeuristicSpec("Foliage Color", "foliage", "foliage_rgb", "foliage_color", overlay_kind="rgb", dependencies=_CLIMATE_STACK_DEPENDENCIES),
    "ForestDensityHeuristicNode": HeuristicSpec("Forest Density", "forest_density", "forest_density", "forest_density", dependencies=_CLIMATE_STACK_DEPENDENCIES),
    "GroundcoverDensityHeuristicNode": HeuristicSpec("Groundcover Density", "groundcover_density", "groundcover_density", "groundcover_density", dependencies=_CLIMATE_STACK_DEPENDENCIES),
}


def _build_heuristic_node_class(class_name: str, spec: HeuristicSpec):
    class _Node(HeuristicMapNode):
        NODE_NAME = spec.node_name
        SPEC = spec

        def __init__(self):
            super().__init__()

        def _build_settings(self, dependency_overlays: Optional[Dict[str, MapOverlayData]] = None) -> Dict[str, Any]:
            settings = super()._build_settings(dependency_overlays)
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
                    needs_deposition=spec.needs_deposition,
                    needs_rock_map=spec.needs_rock_map,
                    supports_flow_override=spec.supports_flow_override,
                    uses_temperature_settings=spec.uses_temperature_settings,
                    uses_precipitation_settings=spec.uses_precipitation_settings,
                    dependencies=spec.dependencies,
                )
            return spec

    _Node.__name__ = class_name
    return _Node


globals().update({name: _build_heuristic_node_class(name, spec) for name, spec in _HEURISTIC_SPECS.items()})

__all__ = list(_HEURISTIC_SPECS.keys()) + ["HeuristicMapNode", "HeuristicSpec"]
