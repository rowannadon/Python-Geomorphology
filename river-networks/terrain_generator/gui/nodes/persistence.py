"""Persistence helpers for node graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


GRAPH_SCHEMA_VERSION = 2
GRAPH_AUTOSAVE_FILENAME = "autosaved.json"


def default_graph_preset_directory() -> Path:
    """Return the directory used for persisted graph presets."""
    return Path(__file__).resolve().parents[2] / "presets"


def default_graph_autosave_path() -> Path:
    """Return the default autosave file path for the node graph."""
    return default_graph_preset_directory() / GRAPH_AUTOSAVE_FILENAME


def save_graph_payload(path: str, payload: Dict[str, Any]) -> Path:
    """Persist a graph payload to disk."""
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(".terrain_graph.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def load_graph_payload(path: str) -> Dict[str, Any]:
    """Load a persisted graph payload from disk."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Graph file must contain a JSON object.")
    return payload


def build_graph_payload(
    *,
    nodes: Iterable[Dict[str, Any]],
    connections: Iterable[Dict[str, Any]],
    pinned_node_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a serialized graph payload."""
    return {
        "version": GRAPH_SCHEMA_VERSION,
        "nodes": list(nodes),
        "connections": list(connections),
        "pinned_node_id": pinned_node_id,
        "metadata": metadata or {},
    }
