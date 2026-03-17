"""3D Terrain Generator with River Networks."""

from .core import (
    TerrainGenerator, 
    TerrainParameters, 
    TerrainData,
    ConsistentFBMNoise,
    RiverGenerator
)
from .gui.main_window import TerrainGeneratorWindow
from .visualization import TerrainViewport

__version__ = "1.0.0"

__all__ = [
    'TerrainGenerator',
    'TerrainParameters', 
    'TerrainData',
    'TerrainGeneratorWindow',
    'TerrainViewport',
    'FBMNoise',
    'RiverGenerator'
]