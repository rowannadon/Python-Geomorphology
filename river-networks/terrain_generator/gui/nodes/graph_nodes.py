"""Graph, river, erosion, and raster bundle nodes."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.tri as mtri
import numpy as np

from ...config import RockLayerConfig, normalize_layer_inputs
from ...core import RiverGenerator, TerrainGenerator, TerrainParameters, normalize
from ...core.particle_erosion import ParticleErosion
from ...core.utils import gaussian_gradient, render_triangulation
from .base_nodes import TerrainBaseNode, _parse_float, _parse_int, _parse_points_text
from .contracts import (
    HeightfieldData,
    MaskData,
    PORT_TYPE_MAP_OVERLAY,
    PORT_TYPE_HEIGHTFIELD,
    PORT_TYPE_MASK,
    PORT_TYPE_RIVER_NETWORK,
    PORT_TYPE_TERRAIN_BUNDLE,
    PORT_TYPE_TERRAIN_GRAPH,
    RiverNetworkData,
    TerrainBundleData,
    TerrainGraphData,
    overlay_from_labels,
)


def _make_generator(**kwargs: Any) -> TerrainGenerator:
    params = TerrainParameters(**kwargs)
    return TerrainGenerator(params)


def _triangulation_to_matplotlib(triangulation: Any) -> mtri.Triangulation:
    if isinstance(triangulation, mtri.Triangulation):
        return triangulation
    if hasattr(triangulation, "points") and hasattr(triangulation, "simplices"):
        return mtri.Triangulation(
            triangulation.points[:, 0],
            triangulation.points[:, 1],
            triangulation.simplices,
        )
    raise ValueError(f"Unsupported triangulation type: {type(triangulation)!r}")


def _render_categorical_map(graph: TerrainGraphData, values: np.ndarray) -> np.ndarray:
    generator = _make_generator(dimension=graph.dimension)
    return generator._render_map(graph.points, np.asarray(values), (graph.dimension, graph.dimension))


def _rasterize_graph_height(graph: TerrainGraphData, values: np.ndarray) -> np.ndarray:
    tri = _triangulation_to_matplotlib(graph.triangulation)
    return render_triangulation((graph.dimension, graph.dimension), tri, np.asarray(values, dtype=np.float64), triangulation=tri)


def _heightfield_from_graph(graph: TerrainGraphData, *, name: str) -> HeightfieldData:
    if graph.point_height is None:
        raise ValueError("Graph has no point heights to rasterize.")
    arr = _rasterize_graph_height(graph, graph.point_height).astype(np.float32)
    return HeightfieldData(array=arr, name=name)


def _coerce_layers_payload(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("rock_layers")
    if not isinstance(payload, list):
        raise ValueError("Rock layer JSON must be a list or an object containing a 'rock_layers' list.")
    return payload


def _load_rock_layers_property(raw_value: Any) -> list[RockLayerConfig]:
    text = str(raw_value or "").strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid rock layer JSON: {exc}") from exc
        return normalize_layer_inputs(_coerce_layers_payload(payload))

    source_path = Path(text).expanduser()
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(f"Failed to read rock layer JSON '{source_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rock layer JSON file '{source_path}': {exc}") from exc

    return normalize_layer_inputs(_coerce_layers_payload(payload), base_path=source_path.parent)


class SampleTerrainGraphNode(TerrainBaseNode):
    """Sample a raster seed terrain into a graph topology."""

    NODE_NAME = "Sample Terrain Graph"
    INPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "land_mask": (PORT_TYPE_MASK,),
    }
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(120, 80, 150)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_input("land_mask", color=(120, 180, 120))
        self.add_output("terrain_graph", color=(180, 120, 200))
        self.add_text_input("disc_radius", "Point Spacing", text="2.0")
        self.add_text_input("seed", "Seed", text="42")

    def execute(self):
        heightfield = self.get_input_heightfield("heightfield")
        land_mask = self.get_input_mask("land_mask")
        if heightfield.array.shape != land_mask.array.shape:
            raise ValueError("Sample Terrain Graph requires heightfield and land mask to match.")
        dim = self.context.get_resolution()
        generator = _make_generator(
            dimension=dim,
            disc_radius=_parse_float(self.get_property("disc_radius"), 2.0),
            seed=_parse_int(self.get_property("seed"), self.context.get_seed()),
        )
        shape = heightfield.array.shape
        self.emit_progress(0.15, "Computing gradients")
        deltas = normalize(np.abs(gaussian_gradient(heightfield.array)))
        self.emit_progress(0.35, "Sampling graph points")
        points, tri, neighbors, edge_weights = generator._create_triangulation(shape)
        coords = generator._points_to_indices(points, shape)
        sampled_land = np.asarray(land_mask.array[coords[:, 0], coords[:, 1]], dtype=bool)
        sampled_deltas = np.asarray(deltas[coords[:, 0], coords[:, 1]], dtype=np.float32)
        graph = TerrainGraphData(
            points=points,
            neighbors=tuple(neighbors),
            edge_weights=tuple(edge_weights),
            triangulation=tri,
            dimension=dim,
            source_heightfield=heightfield,
            land_mask=land_mask,
            sampled_deltas=sampled_deltas,
            sampled_land_mask=sampled_land,
        )
        self.emit_progress(1.0, "Graph sampled")
        self.set_output_data(graph)
        self.signals.execution_finished.emit(self)
        return graph


class SolveBaseGraphElevationNode(TerrainBaseNode):
    """Solve initial graph elevations with Dijkstra."""

    NODE_NAME = "Solve Base Graph Elevation"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(125, 90, 155)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("terrain_graph", color=(180, 120, 200))
        self.add_text_input("max_delta", "Max Delta", text="0.05")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        if graph.sampled_deltas is None:
            raise ValueError("Terrain graph is missing sampled deltas.")
        generator = _make_generator(
            dimension=graph.dimension,
            max_delta=_parse_float(self.get_property("max_delta"), 0.05),
        )
        self.emit_progress(0.2, "Solving base graph heights")
        point_height = generator._compute_height(graph.points, list(graph.neighbors), list(graph.edge_weights), graph.sampled_deltas.astype(np.float64))
        updated = graph.with_updates(point_height=point_height)
        self.emit_progress(1.0, "Base graph ready")
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class TerraceMaxDeltaNode(TerrainBaseNode):
    """Attach variable max-delta modulation to a graph."""

    NODE_NAME = "Terrace/Max Delta Modulation"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(150, 100, 160)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("terrain_graph", color=(180, 120, 200))
        self.add_combo_menu("enabled_curve", "Use Curve", items=["False", "True"])
        self.set_property("enabled_curve", "False")
        self.add_text_input("base_max_delta", "Base Max Delta", text="0.05")
        self.add_text_input("curve_points", "Curve Points", text="0.0:1.0, 1.0:1.0")
        self.add_text_input("terrace_count", "Terrace Count", text="3")
        self.add_text_input("terrace_thickness", "Terrace Thickness", text="0.5")
        self.add_text_input("terrace_flat_delta", "Flat Delta", text="0.01")
        self.add_text_input("terrace_steep_delta", "Steep Delta", text="0.07")
        self.add_text_input("terrace_strength_scale", "Strength Scale", text="-1.7")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        if graph.point_height is None:
            raise ValueError("Terrace node requires graph point heights.")
        if graph.points.size == 0:
            updated = graph.with_updates(variable_max_delta=np.zeros(0, dtype=np.float64))
            self.set_output_data(updated)
            self.signals.execution_finished.emit(self)
            return updated
        normalized_heights = normalize(np.asarray(graph.point_height, dtype=np.float64), bounds=(0, 1))
        generator = _make_generator(
            dimension=graph.dimension,
            max_delta=_parse_float(self.get_property("base_max_delta"), 0.05),
            use_variable_max_delta=True,
            terrace_count=_parse_int(self.get_property("terrace_count"), 3),
            terrace_thickness=_parse_float(self.get_property("terrace_thickness"), 0.5),
            terrace_flat_delta=_parse_float(self.get_property("terrace_flat_delta"), 0.01),
            terrace_steep_delta=_parse_float(self.get_property("terrace_steep_delta"), 0.07),
            terrace_strength_scale=_parse_float(self.get_property("terrace_strength_scale"), -1.7),
        )
        coords = generator._points_to_indices(graph.points, (graph.dimension, graph.dimension))
        variable_max_delta = np.full(normalized_heights.shape, generator.params.max_delta, dtype=np.float64)
        if generator.params.use_variable_max_delta:
            self.emit_progress(0.35, "Generating terrace modulation")
            variable_max_delta = generator._generate_variable_max_delta((graph.dimension, graph.dimension), coords, normalized_heights)
        if self.get_property("enabled_curve") == "True":
            self.emit_progress(0.65, "Applying max-delta curve")
            curve_points = _parse_points_text(str(self.get_property("curve_points") or ""), [(0.0, 1.0), (1.0, 1.0)])
            factors = generator._evaluate_curve(curve_points, normalized_heights)
            variable_max_delta = np.asarray(variable_max_delta, dtype=np.float64) * np.asarray(factors, dtype=np.float64)
        updated = graph.with_updates(variable_max_delta=np.asarray(variable_max_delta, dtype=np.float64))
        self.emit_progress(1.0, "Max-delta modulation ready")
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class RockStackWarpNode(TerrainBaseNode):
    """Attach a rock-stack shift field to a graph."""

    NODE_NAME = "Rock Stack Warp"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(150, 120, 95)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("terrain_graph", color=(180, 120, 200))
        self.add_text_input("rock_warp_strength", "Warp Strength", text="0.0")
        self.add_text_input("rock_warp_scale", "FBM Scale", text="-2.0")
        self.add_text_input("rock_warp_lower", "Lower Bound", text="1.0")
        self.add_text_input("rock_warp_upper", "Upper Bound", text="inf")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        generator = _make_generator(
            dimension=graph.dimension,
            rock_warp_strength=_parse_float(self.get_property("rock_warp_strength"), 0.0),
            rock_warp_scale=_parse_float(self.get_property("rock_warp_scale"), -2.0),
            rock_warp_lower=_parse_float(self.get_property("rock_warp_lower"), 1.0),
            rock_warp_upper=_parse_float(self.get_property("rock_warp_upper"), float("inf")),
        )
        shifts_field = generator._compute_rock_stack_shift((graph.dimension, graph.dimension))
        coords = generator._points_to_indices(graph.points, (graph.dimension, graph.dimension))
        shifts = shifts_field[coords[:, 0], coords[:, 1]]
        updates: Dict[str, Any] = {
            "rock_stack_shifts": np.asarray(shifts, dtype=np.float32),
        }
        if graph.point_height is not None and graph.rock_layers:
            normalized_heights = normalize(np.asarray(graph.point_height, dtype=np.float64), bounds=(0, 1))
            assignments = generator._assign_rock_layers(
                normalized_heights,
                list(graph.rock_layers),
                np.asarray(shifts, dtype=np.float32),
            )
            updates["rock_assignments"] = np.asarray(assignments, dtype=np.int32)
        updated = graph.with_updates(**updates)
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class AssignRockLayersNode(TerrainBaseNode):
    """Resolve and assign rock layers across graph samples."""

    NODE_NAME = "Assign Rock Layers"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(160, 110, 75)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("terrain_graph", color=(180, 120, 200))
        presets_dir = Path(__file__).resolve().parents[2] / "presets"
        self.add_file_input(
            "layers_json",
            "Layers JSON",
            text="",
            placeholder_text="Select rock layers JSON",
            dialog_caption="Select Rock Layers JSON",
            file_filter="JSON Files (*.json);;All Files (*)",
            directory=str(presets_dir),
            tooltip="Choose a JSON file containing a 'rock_layers' array.",
        )
        self.add_text_input("rock_warp_strength", "Warp Strength", text="0.0")
        self.add_text_input("rock_warp_scale", "FBM Scale", text="-2.0")
        self.add_text_input("rock_warp_lower", "Lower Bound", text="1.0")
        self.add_text_input("rock_warp_upper", "Upper Bound", text="inf")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        if graph.point_height is None:
            raise ValueError("Assign Rock Layers requires graph point heights.")
        layers = _load_rock_layers_property(self.get_property("layers_json"))
        if not layers:
            layers = [RockLayerConfig(name="Default", thickness=float("inf"))]
        generator = _make_generator(
            dimension=graph.dimension,
            rock_layers=layers,
            rock_warp_strength=_parse_float(self.get_property("rock_warp_strength"), 0.0),
            rock_warp_scale=_parse_float(self.get_property("rock_warp_scale"), -2.0),
            rock_warp_lower=_parse_float(self.get_property("rock_warp_lower"), 1.0),
            rock_warp_upper=_parse_float(self.get_property("rock_warp_upper"), float("inf")),
        )
        normalized_heights = normalize(np.asarray(graph.point_height, dtype=np.float64), bounds=(0, 1))
        rock_layers, rock_parameters, rock_colors = generator._resolve_rock_layers()
        stack_shifts = graph.rock_stack_shifts
        if stack_shifts is None:
            shift_field = generator._compute_rock_stack_shift((graph.dimension, graph.dimension))
            coords = generator._points_to_indices(graph.points, (graph.dimension, graph.dimension))
            stack_shifts = shift_field[coords[:, 0], coords[:, 1]]
        assignments = generator._assign_rock_layers(normalized_heights, rock_layers, stack_shifts)
        updated = graph.with_updates(
            rock_layers=tuple(rock_layers),
            rock_parameters=tuple(dict(item) for item in rock_parameters),
            rock_colors=tuple(rock_colors),
            rock_stack_shifts=np.asarray(stack_shifts, dtype=np.float32),
            rock_assignments=np.asarray(assignments, dtype=np.int32),
        )
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class ComputeRiverNetworkNode(TerrainBaseNode):
    """Compute a river network on a sampled terrain graph."""

    NODE_NAME = "Compute River Network"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"river_network": (PORT_TYPE_RIVER_NETWORK,)}

    def __init__(self):
        super().__init__()
        self.set_color(85, 120, 170)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("river_network", color=(100, 160, 220))
        self.add_text_input("directional_inertia", "Directional Inertia", text="0.2")
        self.add_text_input("default_water_level", "Default Water Level", text="0.8")
        self.add_text_input("evaporation_rate", "Evaporation Rate", text="0.3")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        if graph.point_height is None or graph.sampled_land_mask is None:
            raise ValueError("River network computation requires graph heights and land mask.")
        river_generator = RiverGenerator(
            directional_inertia=_parse_float(self.get_property("directional_inertia"), 0.2),
            default_water_level=_parse_float(self.get_property("default_water_level"), 0.8),
            evaporation_rate=_parse_float(self.get_property("evaporation_rate"), 0.3),
        )
        normalized_heights = normalize(np.asarray(graph.point_height, dtype=np.float64), bounds=(0, 1))
        self.emit_progress(0.2, "Computing river flow directions")
        network = river_generator.compute_network(
            graph.points,
            [list(item) for item in graph.neighbors],
            normalized_heights,
            np.asarray(graph.sampled_land_mask, dtype=bool),
        )
        payload = RiverNetworkData(
            upstream=network.upstream,
            downstream=network.downstream,
            volume=network.volume,
            watershed=network.watershed,
            point_land_mask=np.asarray(graph.sampled_land_mask, dtype=bool),
        )
        self.emit_progress(1.0, "River network ready")
        self.set_output_data(payload)
        self.signals.execution_finished.emit(self)
        return payload


class ApplyRiverDowncuttingNode(TerrainBaseNode):
    """Apply river-driven downcutting to graph elevations."""

    NODE_NAME = "Apply River Downcutting"
    INPUT_TYPES = {
        "terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,),
        "river_network": (PORT_TYPE_RIVER_NETWORK,),
    }
    OUTPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}

    def __init__(self):
        super().__init__()
        self.set_color(100, 110, 180)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_input("river_network", color=(100, 160, 220))
        self.add_output("terrain_graph", color=(180, 120, 200))
        self.add_text_input("river_downcutting", "River Downcutting", text="1.7")
        self.add_text_input("max_delta", "Base Max Delta", text="0.05")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        river_network = self.get_input_data("river_network", expected_types=(PORT_TYPE_RIVER_NETWORK,))
        if graph.point_height is None or graph.sampled_deltas is None:
            raise ValueError("River downcutting requires base graph heights and deltas.")
        terrain_generator = _make_generator(
            dimension=graph.dimension,
            river_downcutting=_parse_float(self.get_property("river_downcutting"), 1.7),
            max_delta=_parse_float(self.get_property("max_delta"), 0.05),
        )
        network_payload = type("CompatRiverNetwork", (), {
            "upstream": river_network.upstream,
            "downstream": river_network.downstream,
            "volume": river_network.volume,
            "watershed": river_network.watershed,
        })()
        point_height = terrain_generator._compute_final_height(
            graph.points,
            list(graph.neighbors),
            list(graph.edge_weights),
            np.asarray(graph.sampled_deltas, dtype=np.float64),
            network_payload,
            variable_max_delta=graph.variable_max_delta,
            rock_assignments=graph.rock_assignments,
            rock_parameters=list(graph.rock_parameters) if graph.rock_parameters else None,
        )
        updated = graph.with_updates(point_height=np.asarray(point_height, dtype=np.float64))
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class RasterizeGraphFieldNode(TerrainBaseNode):
    """Rasterize a field from a graph payload."""

    NODE_NAME = "Rasterize Graph Field"
    INPUT_TYPES = {
        "terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,),
        "river_network": (PORT_TYPE_RIVER_NETWORK,),
    }
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(145, 85, 145)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_input("river_network", color=(100, 160, 220))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_combo_menu("field_name", "Field", items=["point_height", "variable_max_delta", "rock_assignments", "river_volume", "watershed"])
        self.set_property("field_name", "point_height")

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        field_name = str(self.get_property("field_name") or "point_height")
        if field_name == "point_height":
            if graph.point_height is None:
                raise ValueError("Graph has no point heights to rasterize.")
            arr = _rasterize_graph_height(graph, graph.point_height)
        elif field_name == "variable_max_delta":
            if graph.variable_max_delta is None:
                raise ValueError("Graph has no variable max-delta field.")
            arr = _rasterize_graph_height(graph, graph.variable_max_delta)
        elif field_name == "rock_assignments":
            if graph.rock_assignments is None:
                raise ValueError("Graph has no rock assignments.")
            arr = _render_categorical_map(graph, graph.rock_assignments).astype(np.float32)
        elif field_name == "river_volume":
            river_network = self.get_input_data("river_network", expected_types=(PORT_TYPE_RIVER_NETWORK,))
            arr = _rasterize_graph_height(graph, river_network.volume)
        elif field_name == "watershed":
            river_network = self.get_input_data("river_network", expected_types=(PORT_TYPE_RIVER_NETWORK,))
            arr = _render_categorical_map(graph, river_network.watershed).astype(np.float32)
        else:
            raise ValueError(f"Unsupported field '{field_name}'.")
        payload = HeightfieldData(array=np.asarray(arr, dtype=np.float32), name=self._base_name)
        self.set_output_data(payload)
        self.signals.execution_finished.emit(self)
        return payload


class RockLayerOverlayNode(TerrainBaseNode):
    """Rasterize rock assignments and display them as a terrain overlay."""

    NODE_NAME = "Rock Layer Overlay"
    INPUT_TYPES = {"terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,)}
    OUTPUT_TYPES = {"map_overlay": (PORT_TYPE_MAP_OVERLAY,)}

    def __init__(self):
        super().__init__()
        self.set_color(155, 105, 80)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_output("map_overlay", color=(180, 180, 120))

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        if graph.rock_assignments is None:
            raise ValueError("Graph has no rock assignments.")
        base_heightfield = _heightfield_from_graph(graph, name="Rock Layer Base Height")
        rock_map = _render_categorical_map(graph, graph.rock_assignments).astype(np.int32)
        emergent_land = np.asarray(base_heightfield.array > 0.0, dtype=bool)
        overlay = overlay_from_labels(
            "rock_assignments",
            self._base_name,
            rock_map,
            base_heightfield,
            land_mask=emergent_land,
        )
        preview_bundle = TerrainBundleData(
            heightfield=base_heightfield,
            land_mask=MaskData(array=emergent_land, name="Preview Land Mask"),
            rock_map=rock_map,
            rock_types=tuple(layer.name for layer in graph.rock_layers),
            rock_colors=tuple(graph.rock_colors),
        )
        overlay.metadata["preview_bundle"] = preview_bundle
        self.set_output_data(overlay)
        self.signals.execution_finished.emit(self)
        return overlay


class BundleTerrainOutputsNode(TerrainBaseNode):
    """Bundle graph outputs into raster terrain products."""

    NODE_NAME = "Bundle Terrain Outputs"
    INPUT_TYPES = {
        "terrain_graph": (PORT_TYPE_TERRAIN_GRAPH,),
        "land_mask": (PORT_TYPE_MASK,),
        "river_network": (PORT_TYPE_RIVER_NETWORK,),
    }
    OUTPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}

    def __init__(self):
        super().__init__()
        self.set_color(90, 115, 145)
        self.add_input("terrain_graph", color=(180, 120, 200))
        self.add_input("land_mask", color=(120, 180, 120))
        self.add_input("river_network", color=(100, 160, 220))
        self.add_output("terrain_bundle", color=(140, 200, 210))

    def execute(self):
        graph = self.get_input_data("terrain_graph", expected_types=(PORT_TYPE_TERRAIN_GRAPH,))
        land_mask = self.get_input_mask("land_mask", required=False)
        river_network = self.get_input_data("river_network", required=False, expected_types=(PORT_TYPE_RIVER_NETWORK,))
        if graph.point_height is None:
            raise ValueError("Bundle Terrain Outputs requires graph point heights.")
        self.emit_progress(0.2, "Rasterizing graph height")
        height_arr = _rasterize_graph_height(graph, graph.point_height).astype(np.float32)
        heightfield = HeightfieldData(array=height_arr, name="Terrain Height")
        if land_mask is None:
            if graph.land_mask is not None:
                land_mask = graph.land_mask
            else:
                land_mask = MaskData(array=height_arr > 0.0, name="Land Mask")
        river_volume = None
        watershed_mask = None
        if river_network is not None:
            self.emit_progress(0.45, "Rasterizing river network")
            river_volume = _rasterize_graph_height(graph, river_network.volume).astype(np.float32)
            watershed_mask = _render_categorical_map(graph, river_network.watershed)
        rock_map = None
        if graph.rock_assignments is not None:
            self.emit_progress(0.65, "Rasterizing rock assignments")
            rock_map = _render_categorical_map(graph, graph.rock_assignments)
        bundle = TerrainBundleData(
            heightfield=heightfield,
            land_mask=land_mask,
            river_volume=river_volume,
            watershed_mask=watershed_mask,
            deposition_map=np.zeros(height_arr.shape, dtype=np.float32),
            rock_map=rock_map,
            rock_types=tuple(layer.name for layer in graph.rock_layers),
            rock_colors=tuple(graph.rock_colors),
            metadata={
                "rock_parameters": [dict(item) for item in graph.rock_parameters],
            },
        )
        self.emit_progress(1.0, "Terrain bundle ready")
        self.set_output_data(bundle)
        self.signals.execution_finished.emit(self)
        return bundle


class BuildErosionParameterMapsNode(TerrainBaseNode):
    """Create per-cell parameter maps for erosion from a rock map."""

    NODE_NAME = "Build Erosion Parameter Maps"
    INPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}
    OUTPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}

    def __init__(self):
        super().__init__()
        self.set_color(150, 130, 75)
        self.add_input("terrain_bundle", color=(140, 200, 210))
        self.add_output("terrain_bundle", color=(140, 200, 210))

    def execute(self):
        bundle = self.get_input_data("terrain_bundle", expected_types=(PORT_TYPE_TERRAIN_BUNDLE,))
        rock_map = bundle.rock_map
        rock_parameters = bundle.metadata.get("rock_parameters", []) if bundle.metadata else []
        if rock_map is None or not rock_parameters:
            self.set_output_data(bundle)
            self.signals.execution_finished.emit(self)
            return bundle
        generator = _make_generator(dimension=bundle.heightfield.array.shape[0])
        erosion_maps = generator._build_parameter_maps(np.asarray(rock_map, dtype=np.int32), list(rock_parameters))
        updated = bundle.with_updates(erosion_parameter_maps=erosion_maps)
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


class ParticleErosionNode(TerrainBaseNode):
    """Apply particle erosion to a raster terrain bundle."""

    NODE_NAME = "Particle Erosion"
    INPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}
    OUTPUT_TYPES = {"terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,)}

    def __init__(self):
        super().__init__()
        self.set_color(160, 100, 60)
        self.add_input("terrain_bundle", color=(140, 200, 210))
        self.add_output("terrain_bundle", color=(140, 200, 210))
        self.add_text_input("erosion_iterations", "Droplet Count", text="80000")
        self.add_text_input("erosion_inertia", "Flow Inertia", text="0.3")
        self.add_text_input("erosion_capacity", "Capacity", text="8.0")
        self.add_text_input("erosion_deposition_rate", "Deposition Rate", text="0.2")
        self.add_text_input("erosion_rate", "Erosion Rate", text="0.4")
        self.add_text_input("erosion_evaporation", "Water Retention", text="0.98")
        self.add_text_input("erosion_gravity", "Gravity", text="10.0")
        self.add_text_input("erosion_max_lifetime", "Lifetime", text="60")
        self.add_text_input("erosion_step_size", "Step Size", text="0.3")
        self.add_text_input("erosion_blur_iterations", "Blur Iterations", text="1")
        self.add_text_input("max_delta", "Max Delta", text="0.05")

    def execute(self):
        bundle = self.get_input_data("terrain_bundle", expected_types=(PORT_TYPE_TERRAIN_BUNDLE,))
        heightfield = np.asarray(bundle.heightfield.array, dtype=np.float32)
        max_height = float(heightfield.max())
        if max_height <= 0.0:
            self.set_output_data(bundle)
            self.signals.execution_finished.emit(self)
            return bundle
        erosion_maps = dict(bundle.erosion_parameter_maps)
        if not erosion_maps and bundle.rock_map is not None:
            rock_parameters = bundle.metadata.get("rock_parameters", []) if bundle.metadata else []
            if rock_parameters:
                generator = _make_generator(dimension=heightfield.shape[0])
                erosion_maps = generator._build_parameter_maps(np.asarray(bundle.rock_map, dtype=np.int32), list(rock_parameters))
        erosion = ParticleErosion(
            iterations=_parse_int(self.get_property("erosion_iterations"), 80000),
            inertia=_parse_float(self.get_property("erosion_inertia"), 0.3),
            capacity_const=_parse_float(self.get_property("erosion_capacity"), 8.0),
            deposition_const=_parse_float(self.get_property("erosion_deposition_rate"), 0.2),
            erosion_const=_parse_float(self.get_property("erosion_rate"), 0.4),
            evaporation_const=_parse_float(self.get_property("erosion_evaporation"), 0.98),
            gravity=_parse_float(self.get_property("erosion_gravity"), 10.0),
            max_lifetime=_parse_int(self.get_property("erosion_max_lifetime"), 60),
            step_size=_parse_float(self.get_property("erosion_step_size"), 0.3),
            max_delta=_parse_float(self.get_property("max_delta"), 0.05),
            blur_iterations=_parse_int(self.get_property("erosion_blur_iterations"), 1),
        )
        normalized_terrain = np.asarray(heightfield / max_height, dtype=np.float32)

        def progress_callback(percent: int, message: str):
            progress = min(max(percent / 100.0, 0.0), 1.0)
            self.emit_progress(progress, message)

        eroded, deposition_map = erosion.erode(normalized_terrain, parameter_maps=erosion_maps, progress_callback=progress_callback)
        eroded_height = np.asarray(eroded * max_height, dtype=np.float32)
        land_mask = bundle.land_mask or MaskData(array=eroded_height > 0.001, name="Land Mask")
        land_mask = MaskData(array=np.asarray(eroded_height > 0.001, dtype=bool), name=land_mask.name)
        updated = bundle.with_updates(
            heightfield=bundle.heightfield.with_array(eroded_height, name=bundle.heightfield.name),
            land_mask=land_mask,
            deposition_map=np.asarray(deposition_map * max_height, dtype=np.float32),
            erosion_parameter_maps=erosion_maps,
        )
        self.emit_progress(1.0, "Particle erosion complete")
        self.set_output_data(updated)
        self.signals.execution_finished.emit(self)
        return updated


TerrainGraph = TerrainGraphData
BuildTerrainNode = SampleTerrainGraphNode
