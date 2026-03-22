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


def test_points_to_indices_converts_xy_points_to_row_col_indices():
    terrain_module = _load_terrain_module()
    generator = terrain_module.TerrainGenerator(terrain_module.TerrainParameters(dimension=8))
    points = np.array(
        [
            [1.2, 3.8],
            [6.9, 0.4],
            [10.0, -2.0],
        ],
        dtype=np.float64,
    )

    coords = generator._points_to_indices(points, (5, 7))

    np.testing.assert_array_equal(
        coords,
        np.array(
            [
                [3, 1],
                [0, 6],
                [0, 6],
            ],
            dtype=np.int64,
        ),
    )
