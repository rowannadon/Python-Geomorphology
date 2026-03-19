"""Serialization helpers for erosion parameter presets and rock layer settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


NUMBER_FIELDS = {
    'river_downcutting': float,
    'max_delta': float,
    'erosion_iterations': int,
    'erosion_inertia': float,
    'erosion_capacity': float,
    'erosion_deposition_rate': float,
    'erosion_rate': float,
    'erosion_evaporation': float,
    'erosion_gravity': float,
    'erosion_max_lifetime': int,
    'erosion_step_size': float,
    'erosion_blur_iterations': int,
}


@dataclass
class ErosionParameterSet:
    """Container for a set of erosion-related parameters."""

    name: str = 'Unnamed Erosion Set'
    values: Dict[str, Union[int, float]] = field(default_factory=dict)
    source_path: Optional[Path] = None
    base_albedo_rgb: Optional[Tuple[int, int, int]] = None
    distance_unit: str = 'km'

    def resolve(self, defaults: Mapping[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """Return a mapping containing values with defaults filled in."""
        resolved: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key in self.values:
                resolved[key] = caster(self.values[key])
            else:
                resolved[key] = caster(defaults.get(key, caster()))
        return resolved

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        payload: Dict[str, Any] = {'name': self.name}
        payload.update(self.values)
        payload['distance_units'] = self.distance_unit
        if self.base_albedo_rgb is not None:
            payload['base_albedo_rgb'] = list(self.base_albedo_rgb)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, fallback_name: str = 'Unnamed Erosion Set') -> 'ErosionParameterSet':
        name = str(payload.get('name', fallback_name))
        distance_unit = str(payload.get('distance_units', 'legacy_cells')).strip().lower() or 'legacy_cells'
        values: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key not in payload:
                continue
            raw_value = payload[key]
            try:
                values[key] = caster(raw_value)
            except (TypeError, ValueError):
                continue
        base_albedo: Optional[Tuple[int, int, int]] = None
        color_payload: Optional[Sequence[Any]] = None
        if 'base_albedo_rgb' in payload:
            color_payload = payload['base_albedo_rgb']
        elif 'albedo_rgb' in payload:
            color_payload = payload['albedo_rgb']
        if color_payload is not None:
            try:
                components = [int(float(c)) for c in color_payload]
                if len(components) >= 3:
                    base_albedo = tuple(max(0, min(255, comp)) for comp in components[:3])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                base_albedo = None
        return cls(name=name, values=values, base_albedo_rgb=base_albedo, distance_unit=distance_unit)

    @classmethod
    def from_defaults(cls, defaults: Mapping[str, Union[int, float]], *, name: str = 'Current Erosion Settings') -> 'ErosionParameterSet':
        values: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key not in defaults:
                continue
            try:
                values[key] = caster(defaults[key])
            except (TypeError, ValueError):
                continue
        return cls(name=name, values=values, distance_unit='km')


def load_erosion_parameters(path: Union[str, Path]) -> ErosionParameterSet:
    """Load erosion parameters from a JSON file."""
    target = Path(path)
    with target.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'Erosion parameter file {target} does not contain an object')
    result = ErosionParameterSet.from_mapping(payload, fallback_name=target.stem)
    result.source_path = target
    return result


def save_erosion_parameters(path: Union[str, Path], parameter_set: ErosionParameterSet) -> Path:
    """Persist an erosion parameter set to disk."""
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix('.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(parameter_set.to_json_dict(), handle, indent=2)
    return target


@dataclass
class RockLayerConfig:
    """Configuration of a single rock layer entry."""

    name: str = 'Layer'
    thickness: float = 0.25
    erosion_params_path: Optional[str] = None
    source_directory: Optional[Path] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_path: Optional[Union[str, Path]] = None,
    ) -> 'RockLayerConfig':
        name = str(payload.get('name', 'Layer'))
        thickness_value = payload.get('thickness', 0.25)
        try:
            thickness = float(thickness_value)
        except (TypeError, ValueError):
            thickness = 0.25
        path = payload.get('erosion_params_path') or payload.get('parameters_path')
        if path is not None:
            path = str(path)
        source_directory = Path(base_path) if base_path is not None else None
        return cls(
            name=name,
            thickness=thickness,
            erosion_params_path=path,
            source_directory=source_directory,
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Convert to a JSON-compatible mapping."""
        result: Dict[str, Any] = {
            'name': self.name,
            'thickness': float(self.thickness),
        }
        if self.erosion_params_path:
            result['erosion_params_path'] = self.erosion_params_path
        return result

    def load_parameter_set(self) -> Optional[ErosionParameterSet]:
        """Load the erosion parameter set referenced by this layer, if any."""
        if not self.erosion_params_path:
            return None
        target = Path(self.erosion_params_path).expanduser()
        if not target.is_absolute() and self.source_directory is not None:
            target = self.source_directory / target
        return load_erosion_parameters(target)


def normalize_layer_inputs(
    layers: Iterable[Union[RockLayerConfig, Mapping[str, Any]]],
    *,
    base_path: Optional[Union[str, Path]] = None,
) -> list:
    """Convert arbitrary layer inputs into RockLayerConfig instances."""
    result: list = []
    source_directory = Path(base_path) if base_path is not None else None
    for item in layers:
        if isinstance(item, RockLayerConfig):
            if source_directory is not None and item.source_directory is None:
                item.source_directory = source_directory
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(RockLayerConfig.from_mapping(item, base_path=source_directory))
    return result
