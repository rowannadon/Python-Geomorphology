"""Heuristic map nodes built on top of the shared heuristic engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...heuristics import HeuristicEngine, HeuristicSettings, qimage_to_rgba
from .base_nodes import TerrainBaseNode, _parse_float, _parse_int
from .contracts import (
    ClimateBundleData,
    HeightfieldData,
    MapOverlayData,
    PORT_TYPE_CLIMATE_BUNDLE,
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


@dataclass(frozen=True)
class GroupedHeuristicOutputSpec:
    """Description of a single output exposed by a grouped heuristic node."""

    port_name: str
    display_name: str
    selection_key: str
    array_key: str
    image_key: str
    overlay_kind: str = "scalar"


def _copy_array(value: np.ndarray, dtype: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())


def _copy_optional_array(value: Optional[np.ndarray], dtype: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return _copy_array(value, dtype)


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
        "deposition_map": (PORT_TYPE_MAP_OVERLAY,),
        "rock_map": (PORT_TYPE_MAP_OVERLAY,),
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
        self.add_input("deposition_map", color=(180, 180, 120))
        self.add_input("rock_map", color=(180, 180, 120))
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
        deposition_map = self.get_input_overlay("deposition_map", required=False)
        rock_map = self.get_input_overlay("rock_map", required=False)

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
        cached_overlay = self.context.get_cached_heuristic(cache_key)
        if cached_overlay is not None:
            return cached_overlay

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {spec.node_name}")
        engine.prepare(heightfield.array, engine_settings)
        self.check_cancelled()
        self._seed_engine_from_dependencies(engine, dependency_overlays)
        if flow_override is not None:
            engine.qt_engine.cache["acc"] = np.asarray(flow_override, dtype=np.float32).copy()
            engine.qt_engine.params["flowacc_texture"] = None
        if deposition_map is not None:
            engine.inject_deposition_map(deposition_map)
        if spec.needs_rock_map and rock_map is not None:
            engine.inject_rock_map(rock_map, rock_types, rock_colors)
        self.emit_progress(0.5, f"Computing {spec.node_name}")
        images, arrays = engine.compute([selection_key], cancel_callback=self.is_cancelled)
        self.check_cancelled()
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
        self.check_cancelled()
        self.context.set_cached_heuristic(cache_key, overlay)
        return overlay

    def execute(self):
        overlay = self._compute_overlay()
        self.emit_progress(1.0, f"{self._base_name} ready")
        self.set_output_data(overlay)
        return overlay


class GroupedHeuristicMapNode(TerrainBaseNode):
    """Base class for grouped multi-output heuristic nodes."""

    INPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}
    OUTPUT_SPECS: Tuple[GroupedHeuristicOutputSpec, ...] = ()

    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self._base_name = self.NODE_NAME
        self.add_input("terrain_bundle", color=(140, 200, 210))
        for spec in self.OUTPUT_SPECS:
            self.add_output(spec.port_name, color=(180, 180, 120))
        self.add_text_input("cellsize_override", "Cell Size Override (m)", text="0")

    def _resolve_bundle(self) -> TerrainBundleData:
        bundle = self.get_input_data(
            "terrain_bundle",
            expected_types=(PORT_TYPE_TERRAIN_BUNDLE,),
        )
        if bundle is None:
            raise ValueError(f"{self._base_name} requires a terrain bundle input.")
        return bundle

    def current_output_specs(self) -> Tuple[GroupedHeuristicOutputSpec, ...]:
        return self.OUTPUT_SPECS

    def _build_settings(self) -> Dict[str, Any]:
        world_settings = HeuristicSettings().__dict__.copy()
        world_settings.update(self.context.get_world_settings())
        cellsize_override = _parse_float(self.get_property("cellsize_override"), 0.0)
        if cellsize_override > 0.0:
            world_settings["terrain_size_km"] = 0.0
            world_settings["cellsize"] = cellsize_override
        world_settings.pop("use_simulated_flow", None)
        return world_settings

    def _resolve_optional_maps(
        self,
        bundle: TerrainBundleData,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[str, ...]], Optional[Tuple[Optional[Tuple[int, int, int]], ...]]]:
        flow_array = None
        if bundle.river_volume is not None and self.context.get_world_settings().get("use_simulated_flow", True):
            flow_array = np.asarray(bundle.river_volume, dtype=np.float32)

        deposition_array = None
        if bundle.deposition_map is not None:
            deposition_array = np.asarray(bundle.deposition_map, dtype=np.float32)

        rock_array = None
        rock_types = None
        rock_colors = None
        if bundle.rock_map is not None:
            rock_array = np.asarray(bundle.rock_map, dtype=np.int32)
            rock_types = bundle.rock_types or None
            rock_colors = bundle.rock_colors or None

        return flow_array, deposition_array, rock_array, rock_types, rock_colors

    def _cache_key(
        self,
        bundle: TerrainBundleData,
        settings: Dict[str, Any],
        flow_override: Optional[np.ndarray],
        deposition_map: Optional[np.ndarray],
        rock_map: Optional[np.ndarray],
    ) -> Tuple[str, str]:
        extras = [
            self.__class__.__name__,
            repr(sorted(settings.items())),
            payload_identity_hash(bundle.heightfield),
            "none" if bundle.land_mask is None else payload_identity_hash(bundle.land_mask),
            repr(bundle.rock_types),
            repr(bundle.rock_colors),
        ]
        for arr in (flow_override, deposition_map, rock_map):
            if arr is None:
                extras.append("none")
            else:
                extras.append(f"{arr.shape}:{hash(arr.tobytes())}")
        return ("heuristic_group", "|".join(extras))

    def _build_overlay(
        self,
        spec: GroupedHeuristicOutputSpec,
        *,
        bundle: TerrainBundleData,
        settings: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
        images: Dict[str, Any],
    ) -> MapOverlayData:
        if spec.array_key not in arrays:
            raise ValueError(f"Heuristic result '{spec.array_key}' missing for {spec.display_name}.")
        if spec.image_key not in images:
            raise ValueError(f"Heuristic preview '{spec.image_key}' missing for {spec.display_name}.")

        heightfield = bundle.heightfield
        land_mask = bundle.land_mask.array if bundle.land_mask is not None else None
        rgba = qimage_to_rgba(images[spec.image_key])
        array = arrays[spec.array_key]

        if spec.overlay_kind == "rgb":
            overlay = overlay_from_rgb(spec.image_key, spec.display_name, array, heightfield)
        elif spec.overlay_kind == "deposition":
            overlay = overlay_from_deposition(spec.image_key, spec.display_name, array, heightfield, land_mask=land_mask)
        else:
            overlay = overlay_from_scalar(spec.image_key, spec.display_name, array, heightfield, land_mask=land_mask)

        return MapOverlayData(
            key=overlay.key,
            display_name=overlay.display_name,
            array=overlay.array,
            rgba=rgba,
            base_heightfield=overlay.base_heightfield,
            overlay_kind=overlay.overlay_kind,
            metadata={
                "selection_key": spec.selection_key,
                "array_key": spec.array_key,
                "source_settings": dict(settings),
            },
        )

    def execute(self):
        bundle = self._resolve_bundle()
        settings = self._build_settings()
        output_specs = self.current_output_specs()
        flow_override, deposition_map, rock_map, rock_types, rock_colors = self._resolve_optional_maps(bundle)
        cache_key = self._cache_key(bundle, settings, flow_override, deposition_map, rock_map)
        cached_output = self.context.get_cached_heuristic(cache_key)
        if cached_output is not None:
            self.set_output_data(cached_output)
            return cached_output

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {self.NODE_NAME}")
        engine.prepare(bundle.heightfield.array, engine_settings)
        self.check_cancelled()
        if flow_override is not None:
            engine.qt_engine.cache["acc"] = np.asarray(flow_override, dtype=np.float32).copy()
            engine.qt_engine.params["flowacc_texture"] = None
        if deposition_map is not None:
            engine.inject_deposition_map(deposition_map)
        if rock_map is not None:
            engine.inject_rock_map(rock_map, rock_types, rock_colors)

        selections = list(dict.fromkeys(spec.selection_key for spec in output_specs))
        self.emit_progress(0.5, f"Computing {self.NODE_NAME}")
        images, arrays = engine.compute(selections, cancel_callback=self.is_cancelled)
        self.check_cancelled()

        outputs = {
            spec.port_name: self._build_overlay(
                spec,
                bundle=bundle,
                settings=settings,
                arrays=arrays,
                images=images,
            )
            for spec in output_specs
        }
        self.context.set_cached_heuristic(cache_key, outputs)
        self.emit_progress(1.0, f"{self.NODE_NAME} ready")
        self.set_output_data(outputs)
        return outputs


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


class ClimateHeuristicBundleNode(GroupedHeuristicMapNode):
    """Compute the climate-related heuristic stack from one terrain bundle."""

    NODE_NAME = "Climate Maps"
    OUTPUT_SPECS = (
        GroupedHeuristicOutputSpec("flowacc", "Flow Accumulation", "flowacc", "flowacc", "flowacc_log"),
        GroupedHeuristicOutputSpec("temperature", "Temperature", "climate", "temp_c", "temp_c"),
        GroupedHeuristicOutputSpec("precipitation", "Precipitation", "climate", "precip_mm", "precip_mm"),
        GroupedHeuristicOutputSpec("twi", "TWI", "twi", "twi", "twi"),
        GroupedHeuristicOutputSpec("aet", "AET", "climate", "AET", "AET"),
        GroupedHeuristicOutputSpec("aridity", "Aridity", "climate", "AI", "AI"),
    )
    OUTPUT_TYPES = {
        **{spec.port_name: (PORT_TYPE_MAP_OVERLAY,) for spec in OUTPUT_SPECS},
        "climate_bundle": (PORT_TYPE_CLIMATE_BUNDLE,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(88, 118, 176)
        self.add_output("climate_bundle", color=(120, 170, 220))

    def _build_climate_bundle(
        self,
        *,
        bundle: TerrainBundleData,
        settings: Dict[str, Any],
        engine: HeuristicEngine,
    ) -> ClimateBundleData:
        cache = engine.qt_engine.cache
        slope_aspect = cache.get("slope_aspect")
        grad = cache.get("grad")
        return ClimateBundleData(
            terrain_bundle=bundle,
            flowacc=_copy_array(cache["acc"], np.float32),
            twi=_copy_array(cache["twi"], np.float32),
            temp_c=_copy_array(cache["temp_c"], np.float32),
            precip_mm=_copy_array(cache["P_mm"], np.float32),
            pet=_copy_array(cache["PET"], np.float32),
            aet=_copy_array(cache["AET"], np.float32),
            aridity=_copy_array(cache["AI"], np.float32),
            ocean=_copy_optional_array(cache.get("ocean"), bool),
            coastline=_copy_optional_array(cache.get("coastline"), bool),
            distance_to_coast_m=_copy_optional_array(cache.get("d2coast_m"), np.float32),
            slope_deg=_copy_optional_array(slope_aspect[0] if slope_aspect is not None else None, np.float32),
            aspect_deg=_copy_optional_array(slope_aspect[1] if slope_aspect is not None else None, np.float32),
            grad_x=_copy_optional_array(grad[0] if grad is not None else None, np.float32),
            grad_y=_copy_optional_array(grad[1] if grad is not None else None, np.float32),
            wind_u=_copy_optional_array(cache.get("wind_u"), np.float32),
            wind_v=_copy_optional_array(cache.get("wind_v"), np.float32),
            dir_s=_copy_optional_array(cache.get("dir_s"), np.float32),
            metadata={"source_settings": dict(settings)},
        )

    def execute(self):
        bundle = self._resolve_bundle()
        settings = self._build_settings()
        output_specs = self.current_output_specs()
        flow_override, deposition_map, rock_map, rock_types, rock_colors = self._resolve_optional_maps(bundle)
        cache_key = self._cache_key(bundle, settings, flow_override, deposition_map, rock_map)
        cached_output = self.context.get_cached_heuristic(cache_key)
        if cached_output is not None:
            self.set_output_data(cached_output)
            return cached_output

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {self.NODE_NAME}")
        engine.prepare(bundle.heightfield.array, engine_settings)
        self.check_cancelled()
        if flow_override is not None:
            engine.qt_engine.cache["acc"] = np.asarray(flow_override, dtype=np.float32).copy()
            engine.qt_engine.params["flowacc_texture"] = None
        if deposition_map is not None:
            engine.inject_deposition_map(deposition_map)
        if rock_map is not None:
            engine.inject_rock_map(rock_map, rock_types, rock_colors)

        selections = list(dict.fromkeys(spec.selection_key for spec in output_specs))
        self.emit_progress(0.5, f"Computing {self.NODE_NAME}")
        images, arrays = engine.compute(selections, cancel_callback=self.is_cancelled)
        self.check_cancelled()

        outputs = {
            spec.port_name: self._build_overlay(
                spec,
                bundle=bundle,
                settings=settings,
                arrays=arrays,
                images=images,
            )
            for spec in output_specs
        }
        outputs["climate_bundle"] = self._build_climate_bundle(bundle=bundle, settings=settings, engine=engine)
        self.context.set_cached_heuristic(cache_key, outputs)
        self.emit_progress(1.0, f"{self.NODE_NAME} ready")
        self.set_output_data(outputs)
        return outputs


class TopographicHeuristicBundleNode(GroupedHeuristicMapNode):
    """Compute the topographic heuristic stack from one terrain bundle."""

    NODE_NAME = "Topographic Maps"
    OUTPUT_SPECS = (
        GroupedHeuristicOutputSpec("slope", "Slope", "slope", "slope_deg", "slope_deg"),
        GroupedHeuristicOutputSpec("aspect", "Aspect", "aspect", "aspect_deg", "aspect_deg"),
        GroupedHeuristicOutputSpec("normals", "Normals", "normal", "normal", "normal", overlay_kind="rgb"),
        GroupedHeuristicOutputSpec("curvature", "Curvature", "curvature", "curvature", "curvature"),
        GroupedHeuristicOutputSpec("tpi", "TPI", "tpi@25.0", "tpi_25m", "tpi_25m"),
        GroupedHeuristicOutputSpec("svf", "Sky View Factor", "svf", "svf", "svf"),
    )
    OUTPUT_TYPES = {spec.port_name: (PORT_TYPE_MAP_OVERLAY,) for spec in OUTPUT_SPECS}

    def __init__(self):
        super().__init__()
        self.set_color(92, 126, 166)
        self.add_text_input("tpi_radius_m", "TPI Radius (m)", text="25.0")
        self.add_text_input("svf_dirs", "SVF Directions", text="16")
        self.add_text_input("svf_radius", "SVF Radius (m)", text="100.0")

    def _build_settings(self) -> Dict[str, Any]:
        settings = super()._build_settings()
        tpi_radius = _parse_float(self.get_property("tpi_radius_m"), 25.0)
        settings["tpi_radii"] = (max(tpi_radius, 1.0),)
        settings["svf_dirs"] = max(1, _parse_int(self.get_property("svf_dirs"), 16))
        settings["svf_radius"] = max(_parse_float(self.get_property("svf_radius"), 100.0), 1.0)
        return settings

    def current_output_specs(self) -> Tuple[GroupedHeuristicOutputSpec, ...]:
        radius = max(_parse_float(self.get_property("tpi_radius_m"), 25.0), 1.0)
        tpi_key = f"tpi_{int(radius)}m"
        return (
            GroupedHeuristicOutputSpec("slope", "Slope", "slope", "slope_deg", "slope_deg"),
            GroupedHeuristicOutputSpec("aspect", "Aspect", "aspect", "aspect_deg", "aspect_deg"),
            GroupedHeuristicOutputSpec("normals", "Normals", "normal", "normal", "normal", overlay_kind="rgb"),
            GroupedHeuristicOutputSpec("curvature", "Curvature", "curvature", "curvature", "curvature"),
            GroupedHeuristicOutputSpec("tpi", f"TPI ({int(radius)} m)", f"tpi@{float(radius)}", tpi_key, tpi_key),
            GroupedHeuristicOutputSpec("svf", "Sky View Factor", "svf", "svf", "svf"),
        )


class BiomeHeuristicBundleNode(GroupedHeuristicMapNode):
    """Compute biome, albedo, and vegetation overlays from one terrain bundle."""

    NODE_NAME = "Biome & Cover Maps"
    INPUT_TYPES = {
        "terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,),
        "climate_bundle": (PORT_TYPE_CLIMATE_BUNDLE,),
    }
    OUTPUT_SPECS = (
        GroupedHeuristicOutputSpec("biome", "Biome", "biome", "biome_rgb", "biome_map", overlay_kind="rgb"),
        GroupedHeuristicOutputSpec("albedo", "Albedo", "albedo", "albedo_rgb", "terrain_albedo", overlay_kind="rgb"),
        GroupedHeuristicOutputSpec(
            "continuous_albedo",
            "Continuous Albedo",
            "albedo_continuous",
            "albedo_continuous_rgb",
            "terrain_albedo_continuous",
            overlay_kind="rgb",
        ),
        GroupedHeuristicOutputSpec("foliage_color", "Foliage Color", "foliage", "foliage_rgb", "foliage_color", overlay_kind="rgb"),
        GroupedHeuristicOutputSpec("forest_density", "Forest Density", "forest_density", "forest_density", "forest_density"),
        GroupedHeuristicOutputSpec(
            "groundcover_density",
            "Groundcover Density",
            "groundcover_density",
            "groundcover_density",
            "groundcover_density",
        ),
    )
    OUTPUT_TYPES = {spec.port_name: (PORT_TYPE_MAP_OVERLAY,) for spec in OUTPUT_SPECS}

    def __init__(self):
        super().__init__()
        self.set_color(96, 130, 102)
        self.add_input("climate_bundle", color=(120, 170, 220))

    def _resolve_biome_sources(self) -> Tuple[TerrainBundleData, Optional[ClimateBundleData]]:
        climate_bundle = self.get_input_data(
            "climate_bundle",
            required=False,
            expected_types=(PORT_TYPE_CLIMATE_BUNDLE,),
        )
        if climate_bundle is not None:
            return climate_bundle.terrain_bundle, climate_bundle
        return self._resolve_bundle(), None

    @staticmethod
    def _settings_match_climate_bundle(settings: Dict[str, Any], climate_bundle: ClimateBundleData) -> bool:
        source_settings = climate_bundle.metadata.get("source_settings")
        if not isinstance(source_settings, dict):
            return False
        return source_settings == dict(settings)

    @staticmethod
    def _seed_engine_from_climate_bundle(engine: HeuristicEngine, climate_bundle: ClimateBundleData):
        cache = engine.qt_engine.cache
        cache["acc"] = _copy_array(climate_bundle.flowacc, np.float32)
        cache["twi"] = _copy_array(climate_bundle.twi, np.float32)
        cache["temp_c"] = _copy_array(climate_bundle.temp_c, np.float32)
        cache["P_mm"] = _copy_array(climate_bundle.precip_mm, np.float32)
        cache["PET"] = _copy_array(climate_bundle.pet, np.float32)
        cache["AET"] = _copy_array(climate_bundle.aet, np.float32)
        cache["AI"] = _copy_array(climate_bundle.aridity, np.float32)
        engine.qt_engine.params["flowacc_texture"] = None
        if climate_bundle.ocean is not None:
            cache["ocean"] = _copy_array(climate_bundle.ocean, bool)
        if climate_bundle.coastline is not None:
            cache["coastline"] = _copy_array(climate_bundle.coastline, bool)
        if climate_bundle.distance_to_coast_m is not None:
            cache["d2coast_m"] = _copy_array(climate_bundle.distance_to_coast_m, np.float32)
        if climate_bundle.slope_deg is not None and climate_bundle.aspect_deg is not None:
            cache["slope_aspect"] = (
                _copy_array(climate_bundle.slope_deg, np.float32),
                _copy_array(climate_bundle.aspect_deg, np.float32),
            )
        if climate_bundle.grad_x is not None and climate_bundle.grad_y is not None:
            cache["grad"] = (
                _copy_array(climate_bundle.grad_x, np.float32),
                _copy_array(climate_bundle.grad_y, np.float32),
            )
        if climate_bundle.wind_u is not None:
            cache["wind_u"] = _copy_array(climate_bundle.wind_u, np.float32)
        if climate_bundle.wind_v is not None:
            cache["wind_v"] = _copy_array(climate_bundle.wind_v, np.float32)
        if climate_bundle.dir_s is not None:
            cache["dir_s"] = _copy_array(climate_bundle.dir_s, np.float32)

    def execute(self):
        bundle, climate_bundle = self._resolve_biome_sources()
        settings = self._build_settings()
        flow_override, deposition_map, rock_map, rock_types, rock_colors = self._resolve_optional_maps(bundle)
        cache_key = self._cache_key(bundle, settings, flow_override, deposition_map, rock_map)
        if climate_bundle is not None:
            cache_key = (cache_key[0], f"{cache_key[1]}|climate:{payload_identity_hash(climate_bundle)}")
        cached_output = self.context.get_cached_heuristic(cache_key)
        if cached_output is not None:
            self.set_output_data(cached_output)
            return cached_output

        engine = HeuristicEngine()
        engine_settings = HeuristicSettings(**settings)
        self.emit_progress(0.15, f"Preparing {self.NODE_NAME}")
        engine.prepare(bundle.heightfield.array, engine_settings)
        self.check_cancelled()
        if climate_bundle is not None and self._settings_match_climate_bundle(settings, climate_bundle):
            self._seed_engine_from_climate_bundle(engine, climate_bundle)
        if flow_override is not None:
            engine.qt_engine.cache["acc"] = np.asarray(flow_override, dtype=np.float32).copy()
            engine.qt_engine.params["flowacc_texture"] = None
        if deposition_map is not None:
            engine.inject_deposition_map(deposition_map)
        if rock_map is not None:
            engine.inject_rock_map(rock_map, rock_types, rock_colors)

        selections = list(dict.fromkeys(spec.selection_key for spec in self.OUTPUT_SPECS))
        self.emit_progress(0.5, f"Computing {self.NODE_NAME}")
        images, arrays = engine.compute(selections, cancel_callback=self.is_cancelled)
        self.check_cancelled()

        outputs = {
            spec.port_name: self._build_overlay(
                spec,
                bundle=bundle,
                settings=settings,
                arrays=arrays,
                images=images,
            )
            for spec in self.OUTPUT_SPECS
        }
        self.context.set_cached_heuristic(cache_key, outputs)
        self.emit_progress(1.0, f"{self.NODE_NAME} ready")
        self.set_output_data(outputs)
        return outputs


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

__all__ = list(_HEURISTIC_SPECS.keys()) + [
    "GroupedHeuristicOutputSpec",
    "GroupedHeuristicMapNode",
    "ClimateHeuristicBundleNode",
    "TopographicHeuristicBundleNode",
    "BiomeHeuristicBundleNode",
    "HeuristicMapNode",
    "HeuristicSpec",
]
