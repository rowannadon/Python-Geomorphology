"""Typed payload contracts and visualization adapters for the node editor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ...config import RockLayerConfig
from ...core import (
    TerrainData,
    _build_palette_u8,
    _deposition_to_rgba,
    _gray_to_rgba_norm,
    _labels_to_rgba,
)


PORT_TYPE_HEIGHTFIELD = "heightfield"
PORT_TYPE_MASK = "mask"
PORT_TYPE_TERRAIN_GRAPH = "terrain_graph"
PORT_TYPE_RIVER_NETWORK = "river_network"
PORT_TYPE_TERRAIN_BUNDLE = "terrain_bundle"
PORT_TYPE_MAP_OVERLAY = "map_overlay"
PORT_TYPE_SETTINGS = "settings"


def _as_float32_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return np.ascontiguousarray(arr)


def _as_bool_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return np.ascontiguousarray(arr.astype(bool, copy=False))


def _rgba_from_scalar(array: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
    height = _as_float32_2d(array)
    mask = None if land_mask is None else _as_bool_2d(land_mask)
    out = np.zeros((height.shape[0], height.shape[1], 4), dtype=np.uint8)
    if mask is None:
        mask = np.ones(height.shape, dtype=bool)
    _gray_to_rgba_norm(height, mask, out)
    return out


def _rgba_from_labels(array: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
    labels = np.asarray(array, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError("Expected a 2D label map.")
    if land_mask is None:
        mask = np.ones(labels.shape, dtype=bool)
    else:
        mask = _as_bool_2d(land_mask)
    palette = _build_palette_u8(int(labels.max()) + 1 if labels.size else 1)
    out = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    _labels_to_rgba(labels, mask, palette, out)
    return out


def _rgba_from_deposition(array: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
    deposition = _as_float32_2d(array)
    if land_mask is None:
        mask = np.ones(deposition.shape, dtype=bool)
    else:
        mask = _as_bool_2d(land_mask)
    out = np.zeros((deposition.shape[0], deposition.shape[1], 4), dtype=np.uint8)
    _deposition_to_rgba(deposition, mask, out)
    return out


@dataclass(frozen=True)
class NodePayload:
    """Base class for payloads exchanged between nodes."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def port_type(self) -> str:
        raise NotImplementedError

    def identity_hash(self) -> str:
        return payload_identity_hash(self)


@dataclass(frozen=True)
class HeightfieldData(NodePayload):
    """A typed heightfield payload."""

    array: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    name: str = "Heightfield"
    value_range: Optional[Tuple[float, float]] = None
    normalized_range: Optional[Tuple[float, float]] = (0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "array", _as_float32_2d(self.array))
        if self.value_range is None:
            arr = self.array
            object.__setattr__(self, "value_range", (float(arr.min()), float(arr.max())))

    @property
    def port_type(self) -> str:
        return PORT_TYPE_HEIGHTFIELD

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    def with_array(self, array: np.ndarray, **kwargs: Any) -> "HeightfieldData":
        metadata = dict(self.metadata)
        if "metadata" in kwargs:
            metadata.update(kwargs.pop("metadata"))
        return replace(self, array=_as_float32_2d(array), metadata=metadata, **kwargs)


@dataclass(frozen=True)
class MaskData(NodePayload):
    """A typed raster mask payload."""

    array: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))
    name: str = "Mask"
    mask_kind: str = "boolean"

    def __post_init__(self):
        arr = np.asarray(self.array)
        if arr.ndim != 2:
            raise ValueError("MaskData expects a 2D array.")
        if self.mask_kind == "boolean":
            arr = arr.astype(bool, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)
        object.__setattr__(self, "array", np.ascontiguousarray(arr))

    @property
    def port_type(self) -> str:
        return PORT_TYPE_MASK


@dataclass(frozen=True)
class RiverNetworkData(NodePayload):
    """A typed river-network payload."""

    upstream: Iterable[Any] = field(default_factory=list)
    downstream: Iterable[Any] = field(default_factory=list)
    volume: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    watershed: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    point_land_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, "volume", np.ascontiguousarray(np.asarray(self.volume, dtype=np.float32)))
        object.__setattr__(self, "watershed", np.ascontiguousarray(np.asarray(self.watershed, dtype=np.int32)))
        if self.point_land_mask is not None:
            object.__setattr__(
                self,
                "point_land_mask",
                np.ascontiguousarray(np.asarray(self.point_land_mask, dtype=bool)),
            )

    @property
    def port_type(self) -> str:
        return PORT_TYPE_RIVER_NETWORK


@dataclass(frozen=True)
class TerrainGraphData(NodePayload):
    """A typed graph-based terrain payload."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    neighbors: Tuple[np.ndarray, ...] = field(default_factory=tuple)
    edge_weights: Tuple[np.ndarray, ...] = field(default_factory=tuple)
    triangulation: Any = None
    dimension: int = 256
    source_heightfield: Optional[HeightfieldData] = None
    land_mask: Optional[MaskData] = None
    sampled_deltas: Optional[np.ndarray] = None
    sampled_land_mask: Optional[np.ndarray] = None
    point_height: Optional[np.ndarray] = None
    variable_max_delta: Optional[np.ndarray] = None
    rock_stack_shifts: Optional[np.ndarray] = None
    rock_assignments: Optional[np.ndarray] = None
    rock_layers: Tuple[RockLayerConfig, ...] = field(default_factory=tuple)
    rock_parameters: Tuple[Dict[str, float], ...] = field(default_factory=tuple)
    rock_colors: Tuple[Optional[Tuple[int, int, int]], ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "points", np.ascontiguousarray(np.asarray(self.points, dtype=np.float64)))
        object.__setattr__(self, "neighbors", tuple(np.ascontiguousarray(np.asarray(item, dtype=np.int64)) for item in self.neighbors))
        object.__setattr__(self, "edge_weights", tuple(np.ascontiguousarray(np.asarray(item, dtype=np.float64)) for item in self.edge_weights))
        for attr_name, dtype in (
            ("sampled_deltas", np.float32),
            ("sampled_land_mask", bool),
            ("point_height", np.float64),
            ("variable_max_delta", np.float64),
            ("rock_stack_shifts", np.float32),
            ("rock_assignments", np.int32),
        ):
            value = getattr(self, attr_name)
            if value is not None:
                object.__setattr__(self, attr_name, np.ascontiguousarray(np.asarray(value, dtype=dtype)))
        if self.rock_layers:
            object.__setattr__(self, "rock_layers", tuple(self.rock_layers))
        if self.rock_parameters:
            object.__setattr__(self, "rock_parameters", tuple(dict(item) for item in self.rock_parameters))

    @property
    def port_type(self) -> str:
        return PORT_TYPE_TERRAIN_GRAPH

    def with_updates(self, **kwargs: Any) -> "TerrainGraphData":
        metadata = dict(self.metadata)
        if "metadata" in kwargs:
            metadata.update(kwargs.pop("metadata"))
        return replace(self, metadata=metadata, **kwargs)


@dataclass(frozen=True)
class TerrainBundleData(NodePayload):
    """A typed container for raster terrain outputs."""

    heightfield: HeightfieldData = field(default_factory=HeightfieldData)
    land_mask: Optional[MaskData] = None
    river_volume: Optional[np.ndarray] = None
    watershed_mask: Optional[np.ndarray] = None
    deposition_map: Optional[np.ndarray] = None
    rock_map: Optional[np.ndarray] = None
    rock_types: Tuple[str, ...] = field(default_factory=tuple)
    rock_colors: Tuple[Optional[Tuple[int, int, int]], ...] = field(default_factory=tuple)
    erosion_parameter_maps: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for attr_name, dtype in (
            ("river_volume", np.float32),
            ("watershed_mask", np.int32),
            ("deposition_map", np.float32),
            ("rock_map", np.int32),
        ):
            value = getattr(self, attr_name)
            if value is not None:
                object.__setattr__(self, attr_name, np.ascontiguousarray(np.asarray(value, dtype=dtype)))
        if self.erosion_parameter_maps:
            normalized = {
                str(key): np.ascontiguousarray(np.asarray(value, dtype=np.float64))
                for key, value in self.erosion_parameter_maps.items()
            }
            object.__setattr__(self, "erosion_parameter_maps", normalized)

    @property
    def port_type(self) -> str:
        return PORT_TYPE_TERRAIN_BUNDLE

    def with_updates(self, **kwargs: Any) -> "TerrainBundleData":
        metadata = dict(self.metadata)
        if "metadata" in kwargs:
            metadata.update(kwargs.pop("metadata"))
        return replace(self, metadata=metadata, **kwargs)


@dataclass(frozen=True)
class MapOverlayData(NodePayload):
    """A typed overlay-map payload."""

    key: str = "overlay"
    display_name: str = "Overlay"
    array: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    rgba: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 4), dtype=np.uint8))
    base_heightfield: HeightfieldData = field(default_factory=HeightfieldData)
    overlay_kind: str = "scalar"

    def __post_init__(self):
        arr = np.asarray(self.array)
        if arr.ndim not in (2, 3):
            raise ValueError("MapOverlayData expects a 2D or 3D array.")
        rgba = np.asarray(self.rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("MapOverlayData expects an RGBA image.")
        object.__setattr__(self, "array", np.ascontiguousarray(arr))
        object.__setattr__(self, "rgba", np.ascontiguousarray(rgba))

    @property
    def port_type(self) -> str:
        return PORT_TYPE_MAP_OVERLAY


@dataclass(frozen=True)
class SettingsData(NodePayload):
    """A typed settings payload exposed by global config nodes."""

    values: Dict[str, Any] = field(default_factory=dict)
    scope: str = "settings"

    @property
    def port_type(self) -> str:
        return PORT_TYPE_SETTINGS

    def merged_with(self, override_values: Dict[str, Any]) -> "SettingsData":
        merged = dict(self.values)
        merged.update(override_values)
        return SettingsData(values=merged, scope=self.scope, metadata=dict(self.metadata))


def payload_identity_hash(payload: NodePayload) -> str:
    """Create a stable identity hash for payload cache keys."""
    digest = hashlib.sha1()
    digest.update(payload.port_type.encode("utf-8"))
    if isinstance(payload, HeightfieldData):
        digest.update(payload.array.tobytes())
        digest.update(str(payload.array.shape).encode("utf-8"))
    elif isinstance(payload, MaskData):
        digest.update(np.asarray(payload.array).tobytes())
        digest.update(str(np.asarray(payload.array).shape).encode("utf-8"))
    elif isinstance(payload, TerrainBundleData):
        digest.update(payload.heightfield.array.tobytes())
        if payload.deposition_map is not None:
            digest.update(payload.deposition_map.tobytes())
        if payload.rock_map is not None:
            digest.update(payload.rock_map.tobytes())
    elif isinstance(payload, TerrainGraphData):
        digest.update(payload.points.tobytes())
        if payload.point_height is not None:
            digest.update(payload.point_height.tobytes())
    elif isinstance(payload, MapOverlayData):
        digest.update(payload.rgba.tobytes())
    elif isinstance(payload, SettingsData):
        digest.update(repr(sorted(payload.values.items())).encode("utf-8"))
    else:
        digest.update(repr(payload).encode("utf-8"))
    return digest.hexdigest()


def port_type_for_payload(payload: Any) -> str:
    """Return the logical port type for the supplied payload."""
    if isinstance(payload, NodePayload):
        return payload.port_type
    return "unknown"


def terrain_data_from_heightfield(heightfield: HeightfieldData) -> TerrainData:
    """Create a minimal TerrainData preview from a heightfield payload."""
    arr = heightfield.array
    dim_y, dim_x = arr.shape
    land_mask = arr > 0.0
    terrain = TerrainData(
        heightmap=arr,
        land_mask=land_mask,
        river_volume=np.zeros((dim_y, dim_x), dtype=np.float32),
        watershed_mask=np.zeros((dim_y, dim_x), dtype=np.int32),
        deposition_map=np.zeros((dim_y, dim_x), dtype=np.float32),
        rock_map=np.zeros((dim_y, dim_x), dtype=np.int32),
        triangulation=None,
        rock_types=None,
        rock_albedo=None,
        points=None,
        neighbors=None,
    )
    return terrain


def terrain_data_from_bundle(bundle: TerrainBundleData) -> TerrainData:
    """Create a TerrainData preview from a raster terrain bundle."""
    heightfield = bundle.heightfield.array
    land_mask = (
        bundle.land_mask.array
        if bundle.land_mask is not None
        else np.ones(heightfield.shape, dtype=bool)
    )
    return TerrainData(
        heightmap=heightfield,
        land_mask=np.asarray(land_mask, dtype=bool),
        river_volume=(
            np.asarray(bundle.river_volume, dtype=np.float32)
            if bundle.river_volume is not None
            else np.zeros(heightfield.shape, dtype=np.float32)
        ),
        watershed_mask=(
            np.asarray(bundle.watershed_mask, dtype=np.int32)
            if bundle.watershed_mask is not None
            else np.zeros(heightfield.shape, dtype=np.int32)
        ),
        deposition_map=(
            np.asarray(bundle.deposition_map, dtype=np.float32)
            if bundle.deposition_map is not None
            else np.zeros(heightfield.shape, dtype=np.float32)
        ),
        rock_map=(
            np.asarray(bundle.rock_map, dtype=np.int32)
            if bundle.rock_map is not None
            else np.zeros(heightfield.shape, dtype=np.int32)
        ),
        triangulation=None,
        rock_types=list(bundle.rock_types) if bundle.rock_types else None,
        rock_albedo=list(bundle.rock_colors) if bundle.rock_colors else None,
        points=None,
        neighbors=None,
    )


def overlay_from_scalar(
    key: str,
    display_name: str,
    array: np.ndarray,
    base_heightfield: HeightfieldData,
    *,
    land_mask: Optional[np.ndarray] = None,
) -> MapOverlayData:
    """Build a scalar overlay payload."""
    rgba = _rgba_from_scalar(array, land_mask=land_mask)
    return MapOverlayData(
        key=key,
        display_name=display_name,
        array=np.asarray(array),
        rgba=rgba,
        base_heightfield=base_heightfield,
        overlay_kind="scalar",
    )


def overlay_from_labels(
    key: str,
    display_name: str,
    array: np.ndarray,
    base_heightfield: HeightfieldData,
    *,
    land_mask: Optional[np.ndarray] = None,
) -> MapOverlayData:
    """Build a categorical overlay payload."""
    rgba = _rgba_from_labels(array, land_mask=land_mask)
    return MapOverlayData(
        key=key,
        display_name=display_name,
        array=np.asarray(array),
        rgba=rgba,
        base_heightfield=base_heightfield,
        overlay_kind="label",
    )


def overlay_from_deposition(
    key: str,
    display_name: str,
    array: np.ndarray,
    base_heightfield: HeightfieldData,
    *,
    land_mask: Optional[np.ndarray] = None,
) -> MapOverlayData:
    """Build a deposition overlay payload."""
    rgba = _rgba_from_deposition(array, land_mask=land_mask)
    return MapOverlayData(
        key=key,
        display_name=display_name,
        array=np.asarray(array),
        rgba=rgba,
        base_heightfield=base_heightfield,
        overlay_kind="deposition",
    )


def overlay_from_rgb(
    key: str,
    display_name: str,
    rgb: np.ndarray,
    base_heightfield: HeightfieldData,
) -> MapOverlayData:
    """Build an RGB overlay payload from a 3-channel image."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB array.")
    alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
    rgba = np.concatenate([arr, alpha], axis=2)
    return MapOverlayData(
        key=key,
        display_name=display_name,
        array=arr,
        rgba=rgba,
        base_heightfield=base_heightfield,
        overlay_kind="rgb",
    )
