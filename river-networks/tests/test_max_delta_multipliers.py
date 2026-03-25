import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_terrain_module():
    package_root = Path(__file__).resolve().parents[1] / "terrain_generator"

    terrain_pkg = sys.modules.setdefault("terrain_generator", types.ModuleType("terrain_generator"))
    terrain_pkg.__path__ = [str(package_root)]

    core_pkg = sys.modules.setdefault("terrain_generator.core", types.ModuleType("terrain_generator.core"))
    core_pkg.__path__ = [str(package_root / "core")]

    spec = importlib.util.spec_from_file_location(
        "terrain_generator.core.terrain",
        package_root / "core" / "terrain.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graph_max_delta_multipliers_scale_layer_constraints():
    terrain_module = _load_terrain_module()
    generator = terrain_module.TerrainGenerator(
        terrain_module.TerrainParameters(max_delta=0.5, river_downcutting=1.8)
    )

    node_max_delta, downcut_power = generator._resolve_graph_erosion_parameters(
        3,
        rock_assignments=np.array([0, 1, 1], dtype=np.int32),
        rock_parameters=[
            {"max_delta": 0.4, "river_downcutting": 1.2},
            {"max_delta": 0.2, "river_downcutting": 1.6},
        ],
        max_delta_multipliers=np.array([0.5, 0.25, 1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(node_max_delta, np.array([0.2, 0.05, 0.2], dtype=np.float64))
    np.testing.assert_allclose(downcut_power, np.array([1.2, 1.6, 1.6], dtype=np.float64))


def test_graph_max_delta_multipliers_apply_before_variable_cap():
    terrain_module = _load_terrain_module()
    generator = terrain_module.TerrainGenerator(
        terrain_module.TerrainParameters(max_delta=0.5, river_downcutting=1.8)
    )

    node_max_delta, _ = generator._resolve_graph_erosion_parameters(
        3,
        rock_assignments=np.array([0, 1, 1], dtype=np.int32),
        rock_parameters=[
            {"max_delta": 0.4, "river_downcutting": 1.2},
            {"max_delta": 0.2, "river_downcutting": 1.6},
        ],
        max_delta_multipliers=np.array([1.0, 1.0, 0.5], dtype=np.float64),
        variable_max_delta=np.array([0.3, 0.1, 0.15], dtype=np.float64),
    )

    np.testing.assert_allclose(node_max_delta, np.array([0.3, 0.1, 0.1], dtype=np.float64))
