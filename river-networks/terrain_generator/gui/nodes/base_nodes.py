"""Typed base nodes for the terrain graph editor."""

from __future__ import annotations

import math
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from NodeGraphQt import BaseNode
from NodeGraphQt.constants import NodePropWidgetEnum
try:
    from numba import njit, prange
    _NUMBA = True
except Exception:
    _NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

    def prange(*args):
        return range(*args)

from PyQt5.QtCore import QObject, pyqtSignal

from ...core import ConsistentFBMNoise, gaussian_blur
from ...core.utils import connect_inland_seas
from ...io import HeightmapImporter
from ..curves_widget import DEFAULT_LINEAR_CURVE, apply_curve_points, parse_curve_points
from .context import ExecutionCancellationToken, NodeExecutionCancelled, get_global_context
from .node_widgets import (
    CurveEditorNodeWidget,
    FilePathNodeWidget,
    FbmPreviewNodeWidget,
    FloatSliderNodeWidget,
    PolygonEditorNodeWidget,
    parse_polygon_points,
    regular_polygon_points,
    serialize_polygon_points,
)
from .contracts import (
    HeightfieldData,
    MapOverlayData,
    MaskData,
    PORT_TYPE_HEIGHTFIELD,
    PORT_TYPE_MAP_OVERLAY,
    PORT_TYPE_MASK,
    PORT_TYPE_SETTINGS,
    PORT_TYPE_TERRAIN_BUNDLE,
    SettingsData,
    TerrainBundleData,
    port_type_for_payload,
)
from .custom_node_view import CustomNodeItem


@njit(cache=True)
def _finite_min_max_2d(src: np.ndarray) -> tuple[float, float]:
    h, w = src.shape
    min_value = np.inf
    max_value = -np.inf
    for y in range(h):
        for x in range(w):
            value = src[y, x]
            if value < min_value:
                min_value = value
            if value > max_value:
                max_value = value
    return min_value, max_value


@njit(cache=True, parallel=True, fastmath=True)
def _domain_warp_sample_numba(src: np.ndarray, off_x: np.ndarray, off_y: np.ndarray, out: np.ndarray):
    h, w = src.shape
    for y in prange(h):
        for x in range(w):
            fx = x - off_x[y, x]
            fy = y - off_y[y, x]

            fx_floor = np.floor(fx)
            fy_floor = np.floor(fy)

            x0 = int(fx_floor) % w
            y0 = int(fy_floor) % h
            x1 = (x0 + 1) % w
            y1 = (y0 + 1) % h

            tx = fx - fx_floor
            ty = fy - fy_floor

            s00 = src[y0, x0]
            s10 = src[y0, x1]
            s01 = src[y1, x0]
            s11 = src[y1, x1]

            a0 = s00 + (s10 - s00) * tx
            a1 = s01 + (s11 - s01) * tx
            out[y, x] = a0 + (a1 - a0) * ty


@njit(cache=True, parallel=True, fastmath=True)
def _threshold_flood_numba(src: np.ndarray, sea_level: float, flooded: np.ndarray, land_mask: np.ndarray):
    h, w = src.shape
    for y in prange(h):
        for x in range(w):
            value = src[y, x] - sea_level
            if value > 0.0:
                flooded[y, x] = value
                land_mask[y, x] = value > 0.001
            else:
                flooded[y, x] = 0.0
                land_mask[y, x] = False


@njit(cache=True, parallel=True, fastmath=True)
def _land_mask_numba(src: np.ndarray, sea_level: float, out: np.ndarray):
    h, w = src.shape
    for y in prange(h):
        for x in range(w):
            out[y, x] = src[y, x] > sea_level


@njit(cache=True, parallel=True, fastmath=True)
def _invert_normalized_numba(src: np.ndarray, out: np.ndarray):
    h, w = src.shape
    for y in prange(h):
        for x in range(w):
            out[y, x] = 1.0 - src[y, x]


@njit(cache=True, parallel=True, fastmath=True)
def _invert_range_numba(src: np.ndarray, min_value: float, max_value: float, out: np.ndarray):
    h, w = src.shape
    for y in prange(h):
        for x in range(w):
            out[y, x] = max_value - src[y, x] + min_value


@njit(cache=True, parallel=True, fastmath=True)
def _normalize_or_clamp_numba(src: np.ndarray,
                              clamp_mode: bool,
                              clamp_min: float,
                              clamp_max: float,
                              out: np.ndarray):
    h, w = src.shape
    if clamp_mode:
        for y in prange(h):
            for x in range(w):
                value = src[y, x]
                if value < clamp_min:
                    value = clamp_min
                elif value > clamp_max:
                    value = clamp_max
                out[y, x] = value
        return

    min_value, max_value = _finite_min_max_2d(src)
    if max_value <= min_value:
        for y in prange(h):
            for x in range(w):
                out[y, x] = 0.0
        return

    scale = 1.0 / (max_value - min_value)
    for y in prange(h):
        for x in range(w):
            out[y, x] = (src[y, x] - min_value) * scale


@njit(cache=True, parallel=True, fastmath=True)
def _combine_heightfields_numba(a: np.ndarray,
                                b: np.ndarray,
                                mask: np.ndarray,
                                op_code: int,
                                fade: float,
                                smooth: float,
                                eps: float,
                                out: np.ndarray):
    h, w = a.shape
    for y in prange(h):
        for x in range(w):
            av = a[y, x]
            bv = b[y, x]

            if op_code == 0:
                combined = (1.0 - fade) * av + fade * bv
            elif op_code == 1:
                combined = av + bv
            elif op_code == 2:
                combined = av - bv
            elif op_code == 3:
                combined = av * bv
            elif op_code == 4:
                safe = bv
                if abs(safe) < eps:
                    safe = eps if safe >= 0.0 else -eps
                combined = av / safe
            elif op_code == 5:
                lhs = smooth * av
                rhs = smooth * bv
                mx = lhs if lhs > rhs else rhs
                combined = (mx + math.log(math.exp(lhs - mx) + math.exp(rhs - mx))) / smooth
            elif op_code == 6:
                lhs = -smooth * av
                rhs = -smooth * bv
                mx = lhs if lhs > rhs else rhs
                combined = -(mx + math.log(math.exp(lhs - mx) + math.exp(rhs - mx))) / smooth
            else:
                base = av if av > 1e-6 else 1e-6
                combined = math.pow(base, bv)

            out[y, x] = av + mask[y, x] * (combined - av)


def _parse_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _parse_points_text(raw_text: str, fallback: Sequence[Tuple[float, float]]) -> list[Tuple[float, float]]:
    text = (raw_text or "").strip()
    if not text:
        return list(fallback)
    result: list[Tuple[float, float]] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        left, right = item.split(":", 1)
        try:
            result.append((float(left), float(right)))
        except ValueError:
            continue
    return result or list(fallback)


def _distance_km_to_cells(value_km: float, terrain_size_km: float, resolution: int) -> float:
    """Convert a world-space distance in kilometers into raster cells."""
    return max(float(value_km), 0.0) / max(float(terrain_size_km) / max(int(resolution), 1), 1e-9)


def _legacy_area_to_pixels(value: float, resolution: int, legacy_resolution: float = 1024.0) -> int:
    """Convert legacy pixel-count areas into the active raster resolution."""
    scale = max(int(resolution), 1) / legacy_resolution
    return max(1, int(round(max(float(value), 0.0) * scale * scale)))


class NodeSignals(QObject):
    """Signals emitted by nodes as they execute."""

    progress_updated = pyqtSignal(object, float, str)
    state_changed = pyqtSignal(object, str)
    error_emitted = pyqtSignal(object, str)


class TerrainBaseNode(BaseNode):
    """Base class for all typed terrain nodes."""

    __identifier__ = "terrain"
    INPUT_TYPES: Dict[str, Tuple[str, ...]] = {}
    OUTPUT_TYPES: Dict[str, Tuple[str, ...]] = {}

    def __init__(self):
        super().__init__(qgraphics_item=CustomNodeItem)
        self.signals = NodeSignals()
        self.context = get_global_context()
        self._cached_output = None
        self._is_dirty = True
        self._last_error = None
        self._execution_token: Optional[ExecutionCancellationToken] = None
        self._base_name = getattr(self, "NODE_NAME", self.__class__.__name__)
        self._serializable_properties: list[str] = []
        self._path_properties: set[str] = set()
        self.set_name(self._base_name)
        self.set_color(80, 80, 120)

    def create_property(self, name: str, value: Any = None, *args, **kwargs):  # type: ignore[override]
        if name not in self._serializable_properties and not name.startswith("_"):
            self._serializable_properties.append(name)
        return super().create_property(name, value, *args, **kwargs)

    def set_property(self, name: str, value: Any, **kwargs):  # type: ignore[override]
        try:
            old_value = self.get_property(name)
        except Exception:
            old_value = None
        result = super().set_property(name, value, **kwargs)
        if old_value != value and not name.startswith("_") and name not in {"name", "selected", "pos", "color", "disabled", "visible"}:
            self.mark_dirty()
        return result

    @staticmethod
    def _is_inline_json_text(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return stripped.startswith("{") or stripped.startswith("[")

    @staticmethod
    def _is_absolute_path_text(value: str) -> bool:
        return Path(value).expanduser().is_absolute() or PureWindowsPath(value).is_absolute()

    @staticmethod
    def _relativize_path_text(value: str, *, base_path: Path) -> str:
        stripped = value.strip()
        if not stripped:
            return value
        if not TerrainBaseNode._is_absolute_path_text(stripped):
            return Path(stripped).as_posix()
        try:
            relative = os.path.relpath(stripped, os.fspath(base_path))
        except ValueError:
            return stripped
        return Path(relative).as_posix()

    @staticmethod
    def _resolve_path_text(value: str, *, base_path: Path) -> str:
        stripped = value.strip()
        if not stripped or TerrainBaseNode._is_absolute_path_text(stripped):
            return stripped
        return str((base_path / Path(stripped)).resolve(strict=False))

    def register_path_property(self, name: str):
        if name and not name.startswith("_"):
            self._path_properties.add(name)

    def serializable_properties(self, *, base_path: Optional[Path] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in self._serializable_properties:
            value = self.get_property(name)
            if (
                base_path is not None
                and name in self._path_properties
                and isinstance(value, str)
                and not self._is_inline_json_text(value)
            ):
                value = self._relativize_path_text(value, base_path=base_path)
            result[name] = value
        return result

    def restore_serialized_properties(
        self,
        properties: Optional[Dict[str, Any]],
        *,
        base_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        restored = dict(properties or {})
        if base_path is None:
            return restored
        for name in self._path_properties:
            value = restored.get(name)
            if isinstance(value, str) and not self._is_inline_json_text(value):
                restored[name] = self._resolve_path_text(value, base_path=base_path)
        return restored

    def emit_progress(self, progress: float, message: str):
        self.check_cancelled()
        self.signals.progress_updated.emit(self, max(0.0, min(1.0, float(progress))), str(message))

    def set_execution_state(self, state: str):
        self.signals.state_changed.emit(self, state)

    def emit_error(self, message: str):
        self._last_error = str(message)
        self.signals.error_emitted.emit(self, self._last_error)

    def bind_execution_token(self, token: Optional[ExecutionCancellationToken]):
        self._execution_token = token

    def clear_execution_token(self, token: Optional[ExecutionCancellationToken] = None):
        if token is None or self._execution_token is token:
            self._execution_token = None

    def is_cancelled(self) -> bool:
        return self._execution_token is not None and self._execution_token.is_cancelled()

    def check_cancelled(self):
        token = self._execution_token
        if token is not None:
            token.raise_if_cancelled()

    def mark_dirty(self):
        self._cached_output = None
        self._is_dirty = True
        self.set_execution_state("dirty")
        try:
            for output_port in self.output_ports():
                for connected_port in output_port.connected_ports():
                    node = connected_port.node()
                    if isinstance(node, TerrainBaseNode) and not node._is_dirty:
                        node.mark_dirty()
        except Exception:
            pass

    def get_output_data(self):
        return self._cached_output

    def set_output_data(self, data: Any):
        self.check_cancelled()
        self._cached_output = data
        self._is_dirty = False
        self._last_error = None

    def expected_input_types(self, port_name: str) -> Tuple[str, ...]:
        return tuple(self.INPUT_TYPES.get(port_name, ()))

    def output_types(self, port_name: str) -> Tuple[str, ...]:
        return tuple(self.OUTPUT_TYPES.get(port_name, ()))

    @staticmethod
    def _port_name(port) -> str:
        name_attr = getattr(port, "name", None)
        if callable(name_attr):
            return str(name_attr())
        if name_attr is not None:
            return str(name_attr)
        model = getattr(port, "model", None)
        if model is not None and hasattr(model, "name"):
            return str(model.name)
        return ""

    def _connected_source(self, port_name: str):
        port = self.inputs().get(port_name)
        if port is None:
            raise ValueError(f"Port '{port_name}' not found on node '{self.name()}'.")
        connected = port.connected_ports()
        if not connected:
            return None
        source_port = connected[0]
        return source_port.node(), self._port_name(source_port)

    def get_input_data(self, port_name: str, *, required: bool = True, expected_types: Optional[Tuple[str, ...]] = None):
        connection = self._connected_source(port_name)
        if connection is None:
            if required:
                raise ValueError(f"Input '{port_name}' is not connected.")
            return None
        node, source_port_name = connection
        if isinstance(node, TerrainBaseNode):
            data = node.get_output_data()
        else:
            data = getattr(node, "get_output_data", lambda: None)()
        if isinstance(data, dict):
            data = data.get(source_port_name)
        if data is None:
            if required:
                raise ValueError(f"Input '{port_name}' does not have data available.")
            return None
        allowed_types = expected_types or self.expected_input_types(port_name)
        if allowed_types:
            actual_type = port_type_for_payload(data)
            if actual_type not in allowed_types:
                raise TypeError(
                    f"Port '{port_name}' expected {allowed_types}, received '{actual_type}'."
                )
        return data

    def get_input_heightfield(self, port_name: str = "heightfield", *, required: bool = True) -> Optional[HeightfieldData]:
        data = self.get_input_data(port_name, required=required, expected_types=(PORT_TYPE_HEIGHTFIELD,))
        return data

    def get_input_mask(self, port_name: str = "mask", *, required: bool = True) -> Optional[MaskData]:
        data = self.get_input_data(port_name, required=required, expected_types=(PORT_TYPE_MASK,))
        return data

    def get_input_overlay(self, port_name: str = "map_overlay", *, required: bool = True) -> Optional[MapOverlayData]:
        data = self.get_input_data(port_name, required=required, expected_types=(PORT_TYPE_MAP_OVERLAY,))
        return data

    def get_visualization_payload(self):
        data = self.get_output_data()
        if isinstance(data, dict):
            for key in ("terrain_bundle", "map_overlay", "heightfield", "land_mask"):
                if key in data:
                    return data[key]
            return next(iter(data.values()), None)
        return data

    def _graph_object(self):
        """Return the owning graph for NodeGraphQt API variants."""
        graph_attr = getattr(self, "graph", None)
        if callable(graph_attr):
            try:
                return graph_attr()
            except TypeError:
                return None
        return graph_attr

    def execute(self):
        raise NotImplementedError

    def add_file_input(
        self,
        name: str,
        label: str = "",
        *,
        text: str = "",
        placeholder_text: str = "",
        dialog_caption: str = "Select File",
        file_filter: str = "All Files (*)",
        directory: Optional[str] = None,
        tooltip: Optional[str] = None,
        tab: Optional[str] = None,
    ):
        """Embed a file picker widget into the node and persist its value."""
        self.register_path_property(name)
        widget = FilePathNodeWidget(
            self.view,
            name,
            label,
            text=text,
            placeholder_text=placeholder_text,
            dialog_caption=dialog_caption,
            file_filter=file_filter,
            directory=directory,
        )
        if tooltip:
            widget.setToolTip(tooltip)
        self.add_custom_widget(widget, widget_type=NodePropWidgetEnum.FILE_OPEN.value, tab=tab)


class FBMPreviewMixin:
    """Shared inline preview support for nodes driven by FBM parameters."""

    FBM_PREVIEW_PROPERTY_NAMES = frozenset({"scale", "octaves", "persistence", "lacunarity", "lower", "upper", "seed"})
    FBM_PREVIEW_DEFAULTS: Dict[str, float | int] = {
        "scale": -6.0,
        "octaves": 6,
        "persistence": 0.5,
        "lacunarity": 2.0,
        "lower": 2.0,
        "upper": float("inf"),
        "seed": 42,
    }
    FBM_PREVIEW_LABEL = "Noise Preview"
    FBM_PREVIEW_RENDER_RESOLUTION = 96
    FBM_PREVIEW_WIDGET_SIZE = 144
    FBM_PREVIEW_SEED_OFFSET = 0
    FBM_PREVIEW_MIN_OCTAVES = 0

    def set_property(self, name: str, value: Any, **kwargs):  # type: ignore[override]
        result = super().set_property(name, value, **kwargs)
        if name in self.FBM_PREVIEW_PROPERTY_NAMES:
            self._refresh_fbm_preview()
        return result

    def _setup_fbm_preview(self):
        self._fbm_preview_widget = FbmPreviewNodeWidget(
            self.view,
            "_fbm_preview",
            self.FBM_PREVIEW_LABEL,
            preview_size=self.FBM_PREVIEW_WIDGET_SIZE,
        )
        self.add_custom_widget(self._fbm_preview_widget)
        self._refresh_fbm_preview()

    def _fbm_preview_parameters(self) -> Dict[str, float | int]:
        defaults = self.FBM_PREVIEW_DEFAULTS
        octaves = _parse_int(self.get_property("octaves"), int(defaults["octaves"]))
        if self.FBM_PREVIEW_MIN_OCTAVES > 0:
            octaves = max(self.FBM_PREVIEW_MIN_OCTAVES, octaves)
        return {
            "scale": _parse_float(self.get_property("scale"), float(defaults["scale"])),
            "octaves": octaves,
            "persistence": _parse_float(self.get_property("persistence"), float(defaults["persistence"])),
            "lacunarity": _parse_float(self.get_property("lacunarity"), float(defaults["lacunarity"])),
            "lower": _parse_float(self.get_property("lower"), float(defaults["lower"])),
            "upper": _parse_float(self.get_property("upper"), float(defaults["upper"])),
            "seed": _parse_int(self.get_property("seed"), self.context.get_seed()),
        }

    def _fbm_preview_status_text(self, parameters: Dict[str, float | int]) -> str:
        scale = float(parameters["scale"])
        return f"Scale {scale:.3g}, {int(parameters['octaves'])} oct, seed {int(parameters['seed'])}"

    def _refresh_fbm_preview(self):
        widget = getattr(self, "_fbm_preview_widget", None)
        if widget is None:
            return

        try:
            parameters = self._fbm_preview_parameters()
            generator = ConsistentFBMNoise(
                scale=float(parameters["scale"]),
                octaves=int(parameters["octaves"]),
                persistence=float(parameters["persistence"]),
                lacunarity=float(parameters["lacunarity"]),
                lower=float(parameters["lower"]),
                upper=float(parameters["upper"]),
                seed_offset=self.FBM_PREVIEW_SEED_OFFSET,
                base_seed=int(parameters["seed"]),
            )
            preview = generator.generate((self.FBM_PREVIEW_RENDER_RESOLUTION, self.FBM_PREVIEW_RENDER_RESOLUTION))
        except Exception as exc:
            widget.set_preview_array(None, status=f"Preview unavailable: {exc}")
            return

        widget.set_preview_array(preview, status=self._fbm_preview_status_text(parameters))


class ViewerNode(TerrainBaseNode):
    """Sink node that controls what is shown in the terrain viewport."""

    NODE_NAME = "Viewer"
    DEFAULT_OVERLAY_OPACITY = 0.7
    DEFAULT_HEIGHT_MULTIPLIER = 1.0
    INPUT_TYPES = {
        "terrain_bundle": (PORT_TYPE_TERRAIN_BUNDLE,),
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "map_overlay": (PORT_TYPE_MAP_OVERLAY,),
        "land_mask": (PORT_TYPE_MASK,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(70, 120, 110)
        self.add_input("terrain_bundle", color=(140, 200, 210))
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_input("map_overlay", color=(180, 180, 120))
        self.add_input("land_mask", color=(120, 180, 120))
        height_multiplier_widget = FloatSliderNodeWidget(
            self.view,
            "height_multiplier",
            "Height Multiplier",
            value=self.DEFAULT_HEIGHT_MULTIPLIER,
            min_value=1.0,
            max_value=10.0,
            step=0.1,
        )
        self.add_custom_widget(height_multiplier_widget)
        overlay_opacity_widget = FloatSliderNodeWidget(
            self.view,
            "overlay_opacity",
            "Overlay Opacity",
            value=self.DEFAULT_OVERLAY_OPACITY,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            display_multiplier=100.0,
            display_suffix="%",
        )
        self.add_custom_widget(overlay_opacity_widget)

    @staticmethod
    def _terrain_shape(payload: HeightfieldData | TerrainBundleData) -> tuple[int, int]:
        if isinstance(payload, TerrainBundleData):
            return payload.heightfield.array.shape
        return payload.array.shape

    def _resolve_terrain_source(self, overlay: Optional[MapOverlayData]) -> HeightfieldData | TerrainBundleData | None:
        bundle = self.get_input_data("terrain_bundle", required=False, expected_types=(PORT_TYPE_TERRAIN_BUNDLE,))
        if isinstance(bundle, TerrainBundleData):
            return bundle
        heightfield = self.get_input_heightfield("heightfield", required=False)
        if isinstance(heightfield, HeightfieldData):
            return heightfield
        return None

    def _get_overlay_opacity(self) -> float:
        return max(0.0, min(1.0, _parse_float(self.get_property("overlay_opacity"), self.DEFAULT_OVERLAY_OPACITY)))

    def get_height_multiplier(self) -> float:
        return max(1.0, min(10.0, _parse_float(self.get_property("height_multiplier"), self.DEFAULT_HEIGHT_MULTIPLIER)))

    def execute(self):
        overlay = self.get_input_overlay(required=False)
        terrain_source = self._resolve_terrain_source(overlay)
        land_mask = self.get_input_mask("land_mask", required=False)
        if terrain_source is None and overlay is None and land_mask is None:
            raise ValueError("Viewer requires a terrain bundle, heightfield, map overlay, or land mask input.")

        if overlay is None and land_mask is not None:
            metadata = dict(land_mask.metadata)
            metadata["viewer_mode"] = "2d"
            payload = MaskData(
                array=land_mask.array,
                name=land_mask.name,
                mask_kind=land_mask.mask_kind,
                metadata=metadata,
            )
            self.set_output_data(payload)
            return payload

        if overlay is None:
            self.set_output_data(terrain_source)
            return terrain_source

        if terrain_source is None:
            metadata = dict(overlay.metadata)
            metadata["viewer_mode"] = "2d"
            payload = MapOverlayData(
                key=overlay.key,
                display_name=overlay.display_name,
                array=overlay.array,
                rgba=overlay.rgba,
                base_heightfield=overlay.base_heightfield,
                overlay_kind=overlay.overlay_kind,
                metadata=metadata,
            )
            self.set_output_data(payload)
            return payload

        overlay_shape = overlay.rgba.shape[:2]
        terrain_shape = self._terrain_shape(terrain_source)
        if terrain_shape != overlay_shape:
            raise ValueError(
                f"Viewer overlay shape {overlay_shape} does not match terrain shape {terrain_shape}."
            )

        metadata = dict(overlay.metadata)
        metadata["overlay_opacity"] = self._get_overlay_opacity()
        if isinstance(terrain_source, TerrainBundleData):
            metadata["preview_bundle"] = terrain_source
            base_heightfield = terrain_source.heightfield
        else:
            metadata.pop("preview_bundle", None)
            base_heightfield = terrain_source

        payload = MapOverlayData(
            key=overlay.key,
            display_name=overlay.display_name,
            array=overlay.array,
            rgba=overlay.rgba,
            base_heightfield=base_heightfield,
            overlay_kind=overlay.overlay_kind,
            metadata=metadata,
        )
        self.set_output_data(payload)
        return payload


class ProjectSettingsNode(TerrainBaseNode):
    """Global project settings for the node graph."""

    NODE_NAME = "Project Settings"
    OUTPUT_TYPES = {"settings": (PORT_TYPE_SETTINGS,)}

    def __init__(self):
        super().__init__()
        self.set_color(70, 110, 70)
        self.add_output("settings", color=(120, 180, 120))
        self.add_combo_menu("dimension", "Dimension", items=["512", "1024", "2048", "4096"])
        self.set_property("dimension", "1024")
        self.add_text_input("seed", "Seed", text="42")
        self.context.set_project_settings_node(self)

    def collect_settings(self) -> Dict[str, Any]:
        return {
            "dimension": _parse_int(self.get_property("dimension"), 1024),
            "seed": _parse_int(self.get_property("seed"), 42),
        }

    def mark_dirty(self):
        super().mark_dirty()
        graph = self._graph_object()
        if graph is None:
            return
        for node in graph.all_nodes():
            if isinstance(node, TerrainBaseNode) and node is not self and not node._is_dirty:
                node.mark_dirty()

    def execute(self):
        settings = SettingsData(values=self.collect_settings(), scope="project")
        self.set_output_data(settings)
        return settings


class WorldSettingsNode(TerrainBaseNode):
    """Global world and heuristic defaults."""

    NODE_NAME = "World Settings"
    OUTPUT_TYPES = {"settings": (PORT_TYPE_SETTINGS,)}

    def __init__(self):
        super().__init__()
        self.set_color(70, 90, 130)
        self.add_output("settings", color=(120, 160, 220))
        self.add_text_input("terrain_size_km", "Terrain Size (km)", text="0")
        self.add_text_input("cellsize", "Cell Size Fallback (m)", text="1500")
        self.add_text_input("z_min", "Elevation Min", text="0")
        self.add_text_input("z_max", "Elevation Max", text="6000")
        self.add_text_input("sea_level_m", "Sea Level (m)", text="0")
        self.add_text_input("lapse_rate_c_per_km", "Temp Lapse Rate (C/km)", text="6.5")
        self.add_text_input("t_equator_c", "Temp @ Equator (C)", text="30.0")
        self.add_text_input("t_pole_c", "Temp @ Poles (C)", text="0.0")
        self.add_text_input("coast_decay_km", "Coast Decay (km)", text="1.75")
        self.add_text_input("orographic_alpha", "Orographic Alpha", text="4.0")
        self.add_text_input("shadow_max_distance_km", "Shadow Max Distance (km)", text="400.0")
        self.add_text_input("shadow_decay_km", "Shadow Decay (km)", text="150.0")
        self.add_text_input("shadow_height_threshold_m", "Shadow Height Threshold (m)", text="150.0")
        self.add_text_input("shadow_strength", "Shadow Strength", text="1.0")
        self.add_text_input("biome_mixing", "Biome Mixing Radius (km)", text="1.5")
        self.add_combo_menu("temperature_pattern", "Temperature Pattern", items=["polar", "equatorial", "gradient"])
        self.set_property("temperature_pattern", "polar")
        self.add_text_input("temperature_gradient_azimuth_deg", "Temp Gradient Azimuth (deg)", text="0.0")
        self.add_combo_menu("precip_lat_pattern", "Precipitation Pattern", items=["two_bands", "single_band", "uniform", "gradient"])
        self.set_property("precip_lat_pattern", "two_bands")
        self.add_text_input("precip_gradient_azimuth_deg", "Precip Gradient Azimuth (deg)", text="0.0")
        self.add_combo_menu("prevailing_wind_model", "Wind Model", items=["three_cell", "constant"])
        self.set_property("prevailing_wind_model", "three_cell")
        self.add_text_input("constant_wind_azimuth_deg", "Constant Wind Azimuth (deg)", text="25.0")
        self.add_combo_menu("use_random_biomes", "Random Biomes", items=["False", "True"])
        self.set_property("use_random_biomes", "False")
        self.add_combo_menu("use_simulated_flow", "Use Sim Flow", items=["True", "False"])
        self.set_property("use_simulated_flow", "True")
        self.context.set_world_settings_node(self)

    def collect_settings(self) -> Dict[str, Any]:
        resolution = self.context.get_resolution()
        terrain_size_km = _parse_float(self.get_property("terrain_size_km"), 0.0)
        legacy_cellsize = _parse_float(self.get_property("cellsize"), 1500.0)
        if terrain_size_km <= 0.0:
            terrain_size_km = max(legacy_cellsize, 1e-6) * resolution / 1000.0
        resolved_cellsize = terrain_size_km * 1000.0 / max(resolution, 1)
        return {
            "terrain_size_km": terrain_size_km,
            "cellsize": resolved_cellsize,
            "z_min": _parse_float(self.get_property("z_min"), 0.0),
            "z_max": _parse_float(self.get_property("z_max"), 6000.0),
            "sea_level_m": _parse_float(self.get_property("sea_level_m"), 0.0),
            "lapse_rate_c_per_km": _parse_float(self.get_property("lapse_rate_c_per_km"), 6.5),
            "t_equator_c": _parse_float(self.get_property("t_equator_c"), 30.0),
            "t_pole_c": _parse_float(self.get_property("t_pole_c"), 0.0),
            "coast_decay_km": _parse_float(self.get_property("coast_decay_km"), 1.75),
            "orographic_alpha": _parse_float(self.get_property("orographic_alpha"), 4.0),
            "shadow_max_distance_km": _parse_float(self.get_property("shadow_max_distance_km"), 400.0),
            "shadow_decay_km": _parse_float(self.get_property("shadow_decay_km"), 150.0),
            "shadow_height_threshold_m": _parse_float(self.get_property("shadow_height_threshold_m"), 150.0),
            "shadow_strength": _parse_float(self.get_property("shadow_strength"), 1.0),
            "biome_mixing": max(_parse_float(self.get_property("biome_mixing"), 1.5), 0.0),
            "temperature_pattern": str(self.get_property("temperature_pattern") or "polar"),
            "temperature_gradient_azimuth_deg": _parse_float(self.get_property("temperature_gradient_azimuth_deg"), 0.0),
            "precip_lat_pattern": str(self.get_property("precip_lat_pattern") or "two_bands"),
            "precip_gradient_azimuth_deg": _parse_float(self.get_property("precip_gradient_azimuth_deg"), 0.0),
            "prevailing_wind_model": str(self.get_property("prevailing_wind_model") or "three_cell"),
            "constant_wind_azimuth_deg": _parse_float(self.get_property("constant_wind_azimuth_deg"), 25.0),
            "use_random_biomes": (self.get_property("use_random_biomes") == "True"),
            "use_simulated_flow": (self.get_property("use_simulated_flow") == "True"),
        }

    def mark_dirty(self):
        super().mark_dirty()
        graph = self._graph_object()
        if graph is None:
            return
        for node in graph.all_nodes():
            if isinstance(node, TerrainBaseNode) and node is not self and not node._is_dirty:
                node.mark_dirty()

    def execute(self):
        settings = SettingsData(values=self.collect_settings(), scope="world")
        self.set_output_data(settings)
        return settings


class ConstantNode(TerrainBaseNode):
    """Emit a constant-valued heightfield."""

    NODE_NAME = "Constant"
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(100, 80, 120)
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_text_input("value", "Value", text="0.5")

    def execute(self):
        dim = self.context.get_resolution()
        value = _parse_float(self.get_property("value"), 0.5)
        arr = np.full((dim, dim), value, dtype=np.float32)
        payload = HeightfieldData(array=arr, name=self._base_name, metadata={"node": self._base_name})
        self.set_output_data(payload)
        return payload


class FBMNode(FBMPreviewMixin, TerrainBaseNode):
    """Generate FBM noise."""

    NODE_NAME = "FBM Noise"
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(80, 120, 150)
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_text_input("scale", "Scale", text="-6.0")
        self.add_text_input("octaves", "Octaves", text="6")
        self.add_text_input("persistence", "Persistence", text="0.5")
        self.add_text_input("lacunarity", "Lacunarity", text="2.0")
        self.add_text_input("lower", "Lower Bound", text="2.0")
        self.add_text_input("upper", "Upper Bound", text="inf")
        self.add_text_input("seed", "Seed", text="42")
        self._setup_fbm_preview()

    def execute(self):
        dim = self.context.get_resolution()
        generator = ConsistentFBMNoise(
            scale=_parse_float(self.get_property("scale"), -6.0),
            octaves=_parse_int(self.get_property("octaves"), 6),
            persistence=_parse_float(self.get_property("persistence"), 0.5),
            lacunarity=_parse_float(self.get_property("lacunarity"), 2.0),
            lower=_parse_float(self.get_property("lower"), 2.0),
            upper=_parse_float(self.get_property("upper"), float("inf")),
            seed_offset=0,
            base_seed=_parse_int(self.get_property("seed"), self.context.get_seed()),
        )
        self.emit_progress(0.1, "Generating FBM noise")
        arr = generator.generate((dim, dim))
        self.check_cancelled()
        payload = HeightfieldData(array=arr, name=self._base_name, metadata={"node": self._base_name})
        self.emit_progress(1.0, "FBM noise ready")
        self.set_output_data(payload)
        return payload


class ImportHeightmapNode(TerrainBaseNode):
    """Load a heightmap from disk."""

    NODE_NAME = "Import Heightmap"
    OUTPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "land_mask": (PORT_TYPE_MASK,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(110, 130, 80)
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_output("land_mask", color=(120, 180, 120))
        self.add_text_input("file_path", "File Path", text="")
        self.register_path_property("file_path")

    def execute(self):
        file_path = str(self.get_property("file_path") or "").strip()
        if not file_path:
            raise ValueError("Import Heightmap node requires a file path.")
        dim = self.context.get_resolution()
        self.emit_progress(0.1, "Loading heightmap")
        heightfield, land_mask = HeightmapImporter.load_heightmap(file_path, (dim, dim))
        self.check_cancelled()
        payload = HeightfieldData(array=heightfield, name=self._base_name, metadata={"source_path": file_path})
        mask_payload = MaskData(array=land_mask, name=f"{self._base_name} Mask")
        self.emit_progress(1.0, "Heightmap loaded")
        self.set_output_data({"heightfield": payload, "land_mask": mask_payload})
        return self._cached_output

    def get_output_data(self):
        return self._cached_output


class ShapeNode(TerrainBaseNode):
    """Generate a geometric mask shape."""

    NODE_NAME = "Shape Mask"
    OUTPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "mask": (PORT_TYPE_MASK,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(180, 120, 90)
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_output("mask", color=(120, 180, 120))
        self.add_combo_menu("shape_type", "Shape", items=["Circle", "Square", "Triangle", "Rounded Square", "Polygon"])
        self.set_property("shape_type", "Circle")
        self.add_text_input("scale", "Scale", text="1.0")
        self.add_text_input("offset_x", "Offset X", text="0.0")
        self.add_text_input("offset_y", "Offset Y", text="0.0")
        self.add_text_input("falloff", "Falloff", text="0.1")
        self.add_text_input("polygon_point_count", "Polygon Points", text="5")
        self._polygon_widget = PolygonEditorNodeWidget(
            self.view,
            "polygon_points",
            "Polygon",
            value=serialize_polygon_points(regular_polygon_points(5)),
            point_count=5,
        )
        self.add_custom_widget(self._polygon_widget)
        self._update_polygon_editor_state()

    @staticmethod
    def _polygon_vertices(
        raw_points: Any,
        *,
        point_count: int,
        scale: float,
        offset_x: float,
        offset_y: float,
    ) -> np.ndarray:
        points = parse_polygon_points(
            str(raw_points or ""),
            regular_polygon_points(point_count),
            point_count=max(3, int(point_count)),
        )
        vertices = np.asarray(points, dtype=np.float32)
        vertices[:, 0] = ((vertices[:, 0] * 2.0) - 1.0) * scale + offset_x
        vertices[:, 1] = ((vertices[:, 1] * 2.0) - 1.0) * scale + offset_y
        return vertices

    @staticmethod
    def _polygon_signed_distance(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        if vertices.shape[0] < 3:
            raise ValueError("Polygon masks require at least 3 vertices.")
        inside = np.zeros(x.shape, dtype=bool)
        min_dist_sq = np.full(x.shape, np.inf, dtype=np.float32)
        for idx in range(vertices.shape[0]):
            x1, y1 = float(vertices[idx, 0]), float(vertices[idx, 1])
            x2, y2 = float(vertices[(idx + 1) % vertices.shape[0], 0]), float(vertices[(idx + 1) % vertices.shape[0], 1])

            dy = y2 - y1
            if abs(dy) > 1e-8:
                x_intersection = ((x2 - x1) * (y - y1) / dy) + x1
                inside ^= ((y1 > y) != (y2 > y)) & (x < x_intersection)

            edge_x = x2 - x1
            edge_y = y2 - y1
            edge_len_sq = edge_x * edge_x + edge_y * edge_y
            if edge_len_sq <= 1e-12:
                dist_sq = (x - x1) ** 2 + (y - y1) ** 2
            else:
                t = np.clip(((x - x1) * edge_x + (y - y1) * edge_y) / edge_len_sq, 0.0, 1.0)
                proj_x = x1 + t * edge_x
                proj_y = y1 + t * edge_y
                dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            min_dist_sq = np.minimum(min_dist_sq, dist_sq.astype(np.float32, copy=False))
        distance = np.sqrt(min_dist_sq).astype(np.float32, copy=False)
        return np.where(inside, -distance, distance)

    def _update_polygon_editor_state(self):
        enabled = str(self.get_property("shape_type") or "").strip().lower() == "polygon"
        if hasattr(self, "_polygon_widget"):
            self._polygon_widget.widget().setEnabled(enabled)

    def set_property(self, name: str, value: Any, **kwargs):  # type: ignore[override]
        result = super().set_property(name, value, **kwargs)
        if name == "shape_type":
            self._update_polygon_editor_state()
        elif name == "polygon_point_count" and hasattr(self, "_polygon_widget"):
            self._polygon_widget.set_point_count(max(_parse_int(value, 5), 3))
        return result

    def restore_serialized_properties(
        self,
        properties: Optional[Dict[str, Any]],
        *,
        base_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        restored = super().restore_serialized_properties(properties, base_path=base_path)
        polygon_points = restored.get("polygon_points")
        if polygon_points is not None:
            parsed_points = parse_polygon_points(str(polygon_points or ""), regular_polygon_points(5))
            restored["polygon_point_count"] = str(max(3, len(parsed_points)))
        return restored

    def execute(self):
        dim = self.context.get_resolution()
        shape_type = str(self.get_property("shape_type") or "Circle").strip().lower()
        scale = _parse_float(self.get_property("scale"), 1.0)
        offset_x = _parse_float(self.get_property("offset_x"), 0.0)
        offset_y = _parse_float(self.get_property("offset_y"), 0.0)
        falloff = max(_parse_float(self.get_property("falloff"), 0.1), 0.001)
        y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, dim),
            np.linspace(-1.0, 1.0, dim),
            indexing="ij",
        )
        signed_distance = None
        x = x - offset_x
        y = y - offset_y
        if shape_type == "circle":
            dist = np.sqrt(x**2 + y**2)
            radius = scale
        elif shape_type == "square":
            dist = np.maximum(np.abs(x), np.abs(y))
            radius = scale
        elif shape_type == "triangle":
            d1 = y + scale
            d2 = (-np.sqrt(3.0) * (x - scale) - (y + scale)) / 2.0
            d3 = (np.sqrt(3.0) * (x + scale) - (y + scale)) / 2.0
            dist = -np.minimum(np.minimum(d1, d2), d3)
            radius = 0.0
        elif shape_type == "rounded square":
            p = 4.0
            dist = (np.abs(x) ** p + np.abs(y) ** p) ** (1.0 / p)
            radius = scale
        elif shape_type == "polygon":
            point_count = max(_parse_int(self.get_property("polygon_point_count"), 5), 3)
            vertices = self._polygon_vertices(
                self.get_property("polygon_points"),
                point_count=point_count,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            signed_distance = self._polygon_signed_distance(x + offset_x, y + offset_y, vertices)
        else:
            raise ValueError(f"Unknown shape '{shape_type}'.")
        self.check_cancelled()
        if signed_distance is None:
            edge_dist = np.clip((dist - radius) / falloff, 0.0, 1.0)
        else:
            edge_dist = np.clip(signed_distance / falloff, 0.0, 1.0)
        mask_arr = 1.0 - (3.0 * edge_dist**2 - 2.0 * edge_dist**3)
        heightfield = HeightfieldData(array=mask_arr.astype(np.float32), name=self._base_name)
        mask = MaskData(array=mask_arr >= 0.5, name=f"{self._base_name} Mask")
        self.set_output_data({"heightfield": heightfield, "mask": mask})
        return self._cached_output


class CombineNode(TerrainBaseNode):
    """Blend or combine two heightfields."""

    NODE_NAME = "Combine"
    INPUT_TYPES = {
        "heightfield_a": (PORT_TYPE_HEIGHTFIELD,),
        "heightfield_b": (PORT_TYPE_HEIGHTFIELD,),
        "mask": (PORT_TYPE_MASK, PORT_TYPE_HEIGHTFIELD),
    }
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(140, 90, 130)
        self.add_input("heightfield_a", color=(150, 200, 150))
        self.add_input("heightfield_b", color=(150, 200, 150))
        self.add_input("mask", color=(200, 200, 200))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_combo_menu("operation", "Operation", items=["Fade", "Add", "Subtract", "Multiply", "Divide", "Smooth Max", "Smooth Min", "Pow"])
        self.set_property("operation", "Fade")
        self.add_text_input("fade_amount", "Fade Amount", text="0.5")
        self.add_text_input("smoothness", "Smoothness", text="5.0")
        self.add_text_input("divide_epsilon", "Divide Epsilon", text="1e-5")

    def execute(self):
        a_data = self.get_input_data("heightfield_a", expected_types=(PORT_TYPE_HEIGHTFIELD,))
        b_data = self.get_input_data("heightfield_b", expected_types=(PORT_TYPE_HEIGHTFIELD,))
        mask_input = self.get_input_data("mask", required=False, expected_types=(PORT_TYPE_MASK, PORT_TYPE_HEIGHTFIELD))
        a = a_data.array.astype(np.float32, copy=False)
        b = b_data.array.astype(np.float32, copy=False)
        if a.shape != b.shape:
            raise ValueError("Combine node inputs must have matching shapes.")
        if isinstance(mask_input, MaskData):
            if mask_input.mask_kind == "boolean":
                mask = np.asarray(mask_input.array, dtype=np.float32)
            else:
                mask = np.clip(np.asarray(mask_input.array, dtype=np.float32), 0.0, 1.0)
        elif isinstance(mask_input, HeightfieldData):
            mask = np.clip(mask_input.array.astype(np.float32), 0.0, 1.0)
        else:
            mask = np.ones_like(a, dtype=np.float32)
        operation = str(self.get_property("operation") or "Fade").strip().lower()
        if _NUMBA:
            operation_map = {
                "fade": 0,
                "add": 1,
                "subtract": 2,
                "multiply": 3,
                "divide": 4,
                "smooth max": 5,
                "smooth min": 6,
                "pow": 7,
            }
            op_code = operation_map.get(operation)
            if op_code is None:
                raise ValueError(f"Unsupported combine operation '{operation}'.")
            result = np.empty_like(a, dtype=np.float32)
            _combine_heightfields_numba(
                np.ascontiguousarray(a),
                np.ascontiguousarray(b),
                np.ascontiguousarray(mask.astype(np.float32, copy=False)),
                op_code,
                np.clip(_parse_float(self.get_property("fade_amount"), 0.5), 0.0, 1.0),
                max(abs(_parse_float(self.get_property("smoothness"), 5.0)), 1e-6),
                max(abs(_parse_float(self.get_property("divide_epsilon"), 1e-5)), 1e-6),
                result,
            )
            self.check_cancelled()
        elif operation == "fade":
            fade = np.clip(_parse_float(self.get_property("fade_amount"), 0.5), 0.0, 1.0)
            combined = (1.0 - fade) * a + fade * b
            result = a + mask * (combined - a)
        elif operation == "add":
            combined = a + b
            result = a + mask * (combined - a)
        elif operation == "subtract":
            combined = a - b
            result = a + mask * (combined - a)
        elif operation == "multiply":
            combined = a * b
            result = a + mask * (combined - a)
        elif operation == "divide":
            eps = max(abs(_parse_float(self.get_property("divide_epsilon"), 1e-5)), 1e-6)
            safe = np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)
            combined = a / safe
            result = a + mask * (combined - a)
        elif operation == "smooth max":
            smooth = max(abs(_parse_float(self.get_property("smoothness"), 5.0)), 1e-6)
            combined = np.logaddexp(smooth * a, smooth * b) / smooth
            result = a + mask * (combined - a)
        elif operation == "smooth min":
            smooth = max(abs(_parse_float(self.get_property("smoothness"), 5.0)), 1e-6)
            combined = -np.logaddexp(-smooth * a, -smooth * b) / smooth
            result = a + mask * (combined - a)
        elif operation == "pow":
            combined = np.power(np.clip(a, 1e-6, None), b)
            result = a + mask * (combined - a)
        else:
            raise ValueError(f"Unsupported combine operation '{operation}'.")
        payload = a_data.with_array(result.astype(np.float32), name=self._base_name)
        self.set_output_data(payload)
        return payload


class DomainWarpNode(TerrainBaseNode):
    """Apply domain warping to a heightfield."""

    NODE_NAME = "Domain Warp"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(150, 100, 80)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_text_input("offset_scale", "Offset Scale", text="-5.0")
        self.add_text_input("offset_lower", "Offset Lower", text="1.5")
        self.add_text_input("offset_upper", "Offset Upper", text="inf")
        self.add_text_input("offset_amplitude", "Warp Strength (km)", text="225.0")
        self.add_text_input("seed", "Seed", text="42")

    @staticmethod
    def _sample(array: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if _NUMBA:
            src = np.ascontiguousarray(array.astype(np.float32, copy=False))
            off_x = np.ascontiguousarray(np.asarray(offset.real, dtype=np.float32))
            off_y = np.ascontiguousarray(np.asarray(offset.imag, dtype=np.float32))
            out = np.empty_like(src)
            _domain_warp_sample_numba(src, off_x, off_y, out)
            return out
        shape = np.array(array.shape)
        delta = np.array((offset.real, offset.imag))
        coords = np.array(np.meshgrid(*map(range, shape), indexing="ij")) - delta
        lower = np.floor(coords).astype(int)
        upper = lower + 1
        blend = coords - lower
        lower[0] %= shape[0]
        lower[1] %= shape[1]
        upper[0] %= shape[0]
        upper[1] %= shape[1]

        def lerp(lhs, rhs, t):
            return lhs * (1.0 - t) + rhs * t

        return lerp(
            lerp(array[lower[0], lower[1]], array[lower[0], upper[1]], blend[1]),
            lerp(array[upper[0], lower[1]], array[upper[0], upper[1]], blend[1]),
            blend[0],
        )

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        dim = self.context.get_resolution()
        seed = _parse_int(self.get_property("seed"), self.context.get_seed())
        scale = _parse_float(self.get_property("offset_scale"), -5.0)
        lower = _parse_float(self.get_property("offset_lower"), 1.5)
        upper = _parse_float(self.get_property("offset_upper"), float("inf"))
        amplitude = _distance_km_to_cells(
            _parse_float(self.get_property("offset_amplitude"), 225.0),
            self.context.get_terrain_size_km(),
            dim,
        )
        fbm_x = ConsistentFBMNoise(
            scale=scale,
            octaves=6,
            persistence=0.5,
            lacunarity=2.0,
            lower=lower,
            upper=upper,
            seed_offset=1000,
            base_seed=seed,
        )
        fbm_y = ConsistentFBMNoise(
            scale=scale,
            octaves=6,
            persistence=0.5,
            lacunarity=2.0,
            lower=lower,
            upper=upper,
            seed_offset=2000,
            base_seed=seed,
        )
        offset_x = fbm_x.generate((dim, dim))
        offset_y = fbm_y.generate((dim, dim))
        self.check_cancelled()
        warped = self._sample(source.array, amplitude * (offset_x + 1j * offset_y))
        self.check_cancelled()
        payload = source.with_array(warped.astype(np.float32), name=self._base_name)
        self.set_output_data(payload)
        return payload


class CurveRemapNode(TerrainBaseNode):
    """Remap a heightfield through configurable control points."""

    NODE_NAME = "Curve Remap"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(160, 120, 70)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        curve_widget = CurveEditorNodeWidget(
            self.view,
            "control_points",
            "Curve",
            value="0.0:0.0, 1.0:1.0",
        )
        self.add_custom_widget(curve_widget)

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        points = parse_curve_points(self.get_property("control_points"), DEFAULT_LINEAR_CURVE)
        input_arr = source.array.astype(np.float32)
        normalized = input_arr
        source_min = float(input_arr.min())
        source_max = float(input_arr.max())
        if source_max > source_min:
            normalized = (input_arr - source_min) / (source_max - source_min)
        remapped = apply_curve_points(normalized, points).astype(np.float32)
        payload = source.with_array(remapped, name=self._base_name)
        self.set_output_data(payload)
        return payload


class ThresholdFloodNode(TerrainBaseNode):
    """Threshold a heightfield into land/water and flatten submerged areas."""

    NODE_NAME = "Threshold/Flood"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "land_mask": (PORT_TYPE_MASK,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(90, 140, 110)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_output("land_mask", color=(120, 180, 120))
        self.add_text_input("sea_level", "Sea Level", text="0.3")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        sea_level = _parse_float(self.get_property("sea_level"), 0.0)
        if _NUMBA:
            flooded = np.empty_like(source.array, dtype=np.float32)
            land_mask = np.empty(source.array.shape, dtype=np.bool_)
            _threshold_flood_numba(
                np.ascontiguousarray(source.array.astype(np.float32, copy=False)),
                sea_level,
                flooded,
                land_mask,
            )
            self.check_cancelled()
        else:
            flooded = np.where(source.array > sea_level, source.array - sea_level, 0.0).astype(np.float32)
            land_mask = flooded > 0.001
        payload = source.with_array(flooded, name=self._base_name)
        mask = MaskData(array=land_mask, name=f"{self._base_name} Mask")
        self.set_output_data({"heightfield": payload, "land_mask": mask})
        return self._cached_output


class GaussianBlurNode(TerrainBaseNode):
    """Blur a heightfield with a Gaussian kernel."""

    NODE_NAME = "Gaussian Blur"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(120, 100, 90)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_text_input("sigma", "Sigma (km)", text="3.75")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        sigma = max(
            _distance_km_to_cells(
                _parse_float(self.get_property("sigma"), 0.0),
                self.context.get_terrain_size_km(),
                source.array.shape[0],
            ),
            0.0,
        )
        if sigma <= 0.0:
            payload = source.with_array(source.array.copy(), name=self._base_name)
        else:
            payload = source.with_array(gaussian_blur(source.array, sigma=sigma).astype(np.float32), name=self._base_name)
        self.set_output_data(payload)
        return payload


class ConnectInlandSeasNode(TerrainBaseNode):
    """Apply legacy inland-lake removal and sea-level channel cutting."""

    NODE_NAME = "Connect Inland Lakes"
    INPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "land_mask": (PORT_TYPE_MASK,),
    }
    OUTPUT_TYPES = {
        "heightfield": (PORT_TYPE_HEIGHTFIELD,),
        "land_mask": (PORT_TYPE_MASK,),
    }

    def __init__(self):
        super().__init__()
        self.set_color(85, 130, 130)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_input("land_mask", color=(120, 180, 120))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_output("land_mask", color=(120, 180, 120))
        self.add_text_input("sea_level", "Sea Level", text="0.0")
        self.add_text_input("min_lake_size", "Min Lake Size", text="30")
        self.add_text_input("carve_depth", "Carve Depth", text="0.1")
        self.add_text_input("channel_width", "Channel Width (km)", text="7.5")
        self.add_text_input("channel_falloff", "Channel Falloff", text="1.2")
        self.add_text_input("slope_falloff", "Slope Falloff", text="1.2")
        self.add_text_input("fill_height", "Fill Height", text="0.01")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        land_mask_input = self.get_input_mask("land_mask", required=False)
        sea_level = _parse_float(self.get_property("sea_level"), 0.0)
        if land_mask_input is None:
            land_mask_array = np.asarray(source.array > sea_level, dtype=bool)
            land_mask_name = f"{self._base_name} Mask"
        else:
            land_mask_array = np.asarray(land_mask_input.array, dtype=bool)
            land_mask_name = land_mask_input.name
        channel_falloff = max(_parse_float(self.get_property("channel_falloff"), 1.2), 0.05)
        slope_falloff = _parse_float(self.get_property("slope_falloff"), channel_falloff)
        if slope_falloff <= 0.0:
            slope_falloff = channel_falloff
        adjusted_height, adjusted_land = connect_inland_seas(
            source.array,
            land_mask_array,
            min_sea_size=_legacy_area_to_pixels(_parse_int(self.get_property("min_lake_size"), 30), source.array.shape[0]),
            carve_depth=max(_parse_float(self.get_property("carve_depth"), 0.1), 0.0),
            channel_width=max(
                _distance_km_to_cells(
                    _parse_float(self.get_property("channel_width"), 7.5),
                    self.context.get_terrain_size_km(),
                    source.array.shape[0],
                ),
                0.0,
            ),
            channel_falloff=channel_falloff,
            slope_falloff=max(slope_falloff, 0.05),
            fill_height=max(_parse_float(self.get_property("fill_height"), 0.01), 0.0),
            water_level=sea_level,
        )
        self.check_cancelled()
        payload = source.with_array(np.asarray(adjusted_height, dtype=np.float32), name=self._base_name)
        mask = MaskData(array=np.asarray(adjusted_land, dtype=bool), name=land_mask_name)
        self.set_output_data({"heightfield": payload, "land_mask": mask})
        return self._cached_output


class InvertNode(TerrainBaseNode):
    """Invert a heightfield."""

    NODE_NAME = "Invert"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(130, 100, 140)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_combo_menu("mode", "Mode", items=["Normalized (1 - x)", "Range Flip (max - x + min)"])
        self.set_property("mode", "Normalized (1 - x)")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        mode = str(self.get_property("mode") or "Normalized (1 - x)")
        array = source.array.astype(np.float32, copy=False)
        if _NUMBA and "Normalized" in mode:
            result = np.empty_like(array, dtype=np.float32)
            _invert_normalized_numba(np.ascontiguousarray(array), result)
        elif _NUMBA:
            result = np.empty_like(array, dtype=np.float32)
            min_value, max_value = _finite_min_max_2d(np.ascontiguousarray(array))
            _invert_range_numba(np.ascontiguousarray(array), min_value, max_value, result)
        elif "Normalized" in mode:
            result = 1.0 - array
        else:
            min_value = float(array.min())
            max_value = float(array.max())
            result = max_value - array + min_value
        payload = source.with_array(result.astype(np.float32), name=self._base_name)
        self.set_output_data(payload)
        return payload


class NormalizeClampNode(TerrainBaseNode):
    """Normalize and clamp a heightfield."""

    NODE_NAME = "Normalize/Clamp"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}

    def __init__(self):
        super().__init__()
        self.set_color(110, 110, 150)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("heightfield", color=(150, 200, 150))
        self.add_combo_menu("mode", "Mode", items=["Normalize", "Clamp"])
        self.set_property("mode", "Normalize")
        self.add_text_input("clamp_min", "Clamp Min", text="0.0")
        self.add_text_input("clamp_max", "Clamp Max", text="1.0")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        array = source.array.astype(np.float32, copy=False)
        mode = str(self.get_property("mode") or "Normalize")
        if _NUMBA:
            out = np.empty_like(array, dtype=np.float32)
            _normalize_or_clamp_numba(
                np.ascontiguousarray(array),
                mode == "Clamp",
                _parse_float(self.get_property("clamp_min"), 0.0),
                _parse_float(self.get_property("clamp_max"), 1.0),
                out,
            )
        elif mode == "Clamp":
            out = np.clip(array, _parse_float(self.get_property("clamp_min"), 0.0), _parse_float(self.get_property("clamp_max"), 1.0))
        else:
            min_value = float(array.min())
            max_value = float(array.max())
            if max_value > min_value:
                out = (array - min_value) / (max_value - min_value)
            else:
                out = np.zeros_like(array, dtype=np.float32)
        payload = source.with_array(out.astype(np.float32), name=self._base_name)
        self.set_output_data(payload)
        return payload


class LandMaskNode(TerrainBaseNode):
    """Build a land mask from a heightfield."""

    NODE_NAME = "Land Mask"
    INPUT_TYPES = {"heightfield": (PORT_TYPE_HEIGHTFIELD,)}
    OUTPUT_TYPES = {"land_mask": (PORT_TYPE_MASK,)}

    def __init__(self):
        super().__init__()
        self.set_color(90, 140, 90)
        self.add_input("heightfield", color=(150, 200, 150))
        self.add_output("land_mask", color=(120, 180, 120))
        self.add_text_input("sea_level", "Sea Level", text="0.0")

    def execute(self):
        source = self.get_input_heightfield("heightfield")
        sea_level = _parse_float(self.get_property("sea_level"), 0.0)
        if _NUMBA:
            mask_array = np.empty(source.array.shape, dtype=np.bool_)
            _land_mask_numba(
                np.ascontiguousarray(source.array.astype(np.float32, copy=False)),
                sea_level,
                mask_array,
            )
        else:
            mask_array = np.asarray(source.array > sea_level, dtype=bool)
        mask = MaskData(array=mask_array, name=self._base_name)
        self.set_output_data(mask)
        return mask
