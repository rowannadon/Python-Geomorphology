"""Preset management for terrain and heuristic settings."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class PresetError(RuntimeError):
    """Raised when a preset file cannot be read or written."""


class PresetManager:
    """Handle serialization of combined terrain and heuristic presets."""

    PRESET_VERSION = 1

    def __init__(self, preset_directory: Optional[Union[str, Path]] = None):
        if preset_directory is None:
            preset_directory = Path(__file__).resolve().parent.parent / 'presets'
        self._preset_directory = Path(preset_directory)
        self._preset_directory.mkdir(parents=True, exist_ok=True)

    def default_directory(self) -> Path:
        """Return the default directory used for presets."""
        return self._preset_directory

    @classmethod
    def _ensure_path(cls, path: Union[str, Path]) -> Path:
        result = Path(path)
        if result.is_dir():
            raise PresetError(f"Preset path {result} is a directory; expected a file path.")
        if not result.suffix:
            result = result.with_suffix('.json')
        return result

    @classmethod
    def _normalize_for_json(cls, value: Any) -> Any:
        """Convert arbitrary Python/numpy values into JSON-serialisable forms."""
        if isinstance(value, dict):
            return {str(key): cls._normalize_for_json(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize_for_json(val) for val in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return cls._normalize_for_json(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if math.isinf(value):
                return 'inf' if value > 0 else '-inf'
            if math.isnan(value):
                return None
        return value

    def _build_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'preset_version': self.PRESET_VERSION,
        }
        if metadata:
            base.update({str(k): self._normalize_for_json(v) for k, v in metadata.items()})
        return base

    def save_preset(
        self,
        file_path: Union[str, Path],
        *,
        terrain_state: Dict[str, Any],
        heuristics_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the provided state snapshot to disk."""
        target = self._ensure_path(file_path)
        payload = {
            'version': self.PRESET_VERSION,
            'terrain': self._normalize_for_json(terrain_state or {}),
            'heuristics': self._normalize_for_json(heuristics_state or {}),
            'metadata': self._build_metadata(metadata),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            raise PresetError(f"Failed to write preset: {exc}") from exc
        return target

    def load_preset(self, file_path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load a preset from disk and return terrain, heuristic, metadata mappings."""
        path = Path(file_path)
        if not path.exists():
            raise PresetError(f"Preset not found: {path}")
        try:
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PresetError(f"Failed to read preset: {exc}") from exc

        if not isinstance(data, dict):
            raise PresetError("Invalid preset format: expected a JSON object.")

        terrain = data.get('terrain', {})
        heuristics = data.get('heuristics', {})
        metadata = data.get('metadata', {})

        if not isinstance(terrain, dict) or not isinstance(heuristics, dict):
            raise PresetError("Preset is missing terrain or heuristic data.")
        if not isinstance(metadata, dict):
            metadata = {}

        return terrain, heuristics, metadata
