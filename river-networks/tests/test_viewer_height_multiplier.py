import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_contracts_module():
    package_root = Path(__file__).resolve().parents[1] / "terrain_generator"

    terrain_pkg = sys.modules.setdefault("terrain_generator", types.ModuleType("terrain_generator"))
    terrain_pkg.__path__ = [str(package_root)]

    gui_pkg = sys.modules.setdefault("terrain_generator.gui", types.ModuleType("terrain_generator.gui"))
    gui_pkg.__path__ = [str(package_root / "gui")]

    nodes_pkg = sys.modules.setdefault("terrain_generator.gui.nodes", types.ModuleType("terrain_generator.gui.nodes"))
    nodes_pkg.__path__ = [str(package_root / "gui" / "nodes")]

    spec = importlib.util.spec_from_file_location(
        "terrain_generator.gui.nodes.contracts",
        package_root / "gui" / "nodes" / "contracts.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_heightfield_preview_height_multiplier_scales_display_only():
    contracts = _load_contracts_module()
    source = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    heightfield = contracts.HeightfieldData(array=source, name="Preview")

    preview = contracts.terrain_data_from_heightfield(heightfield, height_multiplier=3.0)

    np.testing.assert_allclose(preview.heightmap, source * 3.0)
    np.testing.assert_allclose(heightfield.array, source)


def test_bundle_preview_height_multiplier_scales_display_only():
    contracts = _load_contracts_module()
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    heightfield = contracts.HeightfieldData(array=source, name="Bundle")
    bundle = contracts.TerrainBundleData(heightfield=heightfield)

    preview = contracts.terrain_data_from_bundle(bundle, height_multiplier=2.5)

    np.testing.assert_allclose(preview.heightmap, source * 2.5)
    np.testing.assert_allclose(bundle.heightfield.array, source)
