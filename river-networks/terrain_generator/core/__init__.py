"""Core terrain generation modules."""

from .terrain import TerrainGenerator, TerrainParameters, TerrainData
from .noise import ConsistentFBMNoise
from .rivers import RiverGenerator, RiverNetwork
from .utils import normalize, gaussian_blur, gaussian_gradient, _gray_to_rgba_norm, _deposition_to_rgba, _build_palette_u8, _labels_to_rgba

__all__ = [
    'TerrainGenerator', 'TerrainParameters', 'TerrainData',
    'ConsistentFBMNoise',
    'RiverGenerator', 'RiverNetwork',
    'normalize', 'gaussian_blur', 'gaussian_gradient', '_gray_to_rgba_norm', 
    '_deposition_to_rgba', '_build_palette_u8', '_labels_to_rgba'
]