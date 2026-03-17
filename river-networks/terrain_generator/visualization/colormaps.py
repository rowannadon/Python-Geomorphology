"""Color schemes for terrain visualization."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict

class TerrainColormap:
    """Manages terrain color schemes."""
    
    @staticmethod
    def get_terrain() -> LinearSegmentedColormap:
        """Returns the terrain colormap."""
        return LinearSegmentedColormap.from_list('terrain', [
            (0.00, (0.0, 0.1, 0.3)),     # Dark blue
            (0.03, (0.9, 0.8, 0.6)),     # Sand
            (0.05, (0.10, 0.2, 0.10)),   # Dark green
            (0.25, (0.3, 0.45, 0.3)),    # Green
            (0.50, (0.5, 0.5, 0.35)),    # Brown
            (0.80, (0.4, 0.36, 0.33)),   # Rocky
            (1.00, (1.0, 1.0, 1.0)),     # Snow
        ])
    
    @staticmethod
    def get_grayscale() -> LinearSegmentedColormap:
        """Returns a grayscale colormap."""
        return LinearSegmentedColormap.from_list('grayscale', [
            (0.0, (0.0, 0.0, 0.0)),
            (1.0, (1.0, 1.0, 1.0)),
        ])
    
    @staticmethod
    def get_topographic() -> LinearSegmentedColormap:
        """Returns a topographic colormap with contour-like bands."""
        return LinearSegmentedColormap.from_list('topographic', [
            (0.0, (0.0, 0.0, 0.0)),
            (0.05, (0.6, 0.0, 1.0)),
            (0.10, (0.0, 0.0, 1.0)),
            (0.25, (0.0, 0.9, 1.0)),
            (0.4, (0.0, 1.0, 0.0)),
            (0.7, (1.0, 1.0, 0.0)),
            (1.0, (1.0, 0.0, 0.0)),
        ])
    
    @staticmethod
    def get_all() -> Dict[str, LinearSegmentedColormap]:
        """Get all available colormaps."""
        return {
            'terrain': TerrainColormap.get_terrain(),
            'grayscale': TerrainColormap.get_grayscale(),
            'topographic': TerrainColormap.get_topographic()
        }