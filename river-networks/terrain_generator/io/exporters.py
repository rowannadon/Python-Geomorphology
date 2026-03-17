"""Export functionality for terrain data."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Iterable

from ..core.utils import normalize

class TerrainExporter:
    """Handles terrain data export."""
    
    @staticmethod
    def export_heightmap(heightmap: np.ndarray, filepath: str, 
                        format: str = "PNG_8"):
        """Export heightmap to image file."""
        exporters = {
            "PNG_8": TerrainExporter._export_png_8bit,
            "PNG_16": TerrainExporter._export_png_16bit,
            "TIFF_32": TerrainExporter._export_tiff_32bit
        }
        
        if format in exporters:
            exporters[format](heightmap, Path(filepath))
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    @staticmethod
    def export_flow_mask(river_volume: np.ndarray, land_mask: np.ndarray,
                        filepath: str, format: str = "PNG_8"):
        """Export flow mask to image file."""
        # Prepare flow data
        flow_data = river_volume.copy()

        # Set non-land areas to 0
        if land_mask is not None:
            flow_data[~land_mask] = 0

        # Normalize to 0-1 range
        if flow_data.max() > 0:
            flow_data = flow_data / flow_data.max()

        # Export using same methods as heightmap
        TerrainExporter.export_heightmap(flow_data, filepath, format)

    @staticmethod
    def export_watershed_mask(watershed_mask: np.ndarray, land_mask: np.ndarray,
                              filepath: str, format: str = "PNG_8"):
        """Export watershed mask to image file."""
        mask_data = watershed_mask.astype(np.float32)

        if land_mask is not None:
            mask_data = mask_data.copy()
            mask_data[~land_mask] = 0

        if mask_data.max() > 0:
            mask_data = mask_data / mask_data.max()

        TerrainExporter.export_heightmap(mask_data, filepath, format)

    @staticmethod
    def export_deposition_mask(deposition_map: np.ndarray, land_mask: np.ndarray,
                               filepath: str, format: str = "PNG_8"):
        """Export deposition mask to image file."""
        # Prepare deposition data
        # Positive values = deposition (bright), negative = erosion (dark)
        deposition_data = deposition_map.copy()
        
        # Normalize to 0-1 range where 0.5 is neutral (no change)
        max_change = max(abs(deposition_data.min()), abs(deposition_data.max()))
        if max_change > 0:
            # Map [-max_change, max_change] to [0, 1] with 0.5 as center
            normalized = (deposition_data / (2 * max_change)) + 0.5
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.full_like(deposition_data, 0.5)
        
        # Export using same methods as heightmap
        TerrainExporter.export_heightmap(normalized, filepath, format)

    @staticmethod
    def export_rock_map(rock_map: np.ndarray,
                        land_mask: Optional[np.ndarray],
                        filepath: str,
                        format: str = "PNG_8",
                        colors: Optional[Iterable[Iterable[float]]] = None):
        """Export the rock layer map as an RGB image."""
        if rock_map is None:
            raise ValueError("Rock map is not available for export.")

        if format != "PNG_8":
            raise ValueError("Rock map export currently supports PNG (8-bit) output.")

        indices = np.asarray(rock_map, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("Rock map is empty.")

        max_index = int(indices.max())
        if max_index < 0:
            max_index = 0

        palette = TerrainExporter._rock_color_palette(max_index + 1, colors)

        h, w = indices.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        clipped_indices = np.clip(indices, 0, max_index)

        for rock_idx, color in enumerate(palette):
            rgb_image[clipped_indices == rock_idx] = color

        if land_mask is not None:
            rgb_image = rgb_image.copy()
            rgb_image[~land_mask] = (0, 0, 0)

        img = Image.fromarray(rgb_image, mode='RGB')
        img.save(Path(filepath))

    @staticmethod
    def _rock_color_palette(count: int,
                            overrides: Optional[Iterable[Iterable[float]]] = None) -> np.ndarray:
        """Generate a stable palette for rock layer visualisation."""
        if overrides is not None:
            manual = []
            for entry in overrides:
                try:
                    r, g, b = entry
                except (TypeError, ValueError):
                    continue
                manual.append([
                    np.clip(int(r), 0, 255),
                    np.clip(int(g), 0, 255),
                    np.clip(int(b), 0, 255),
                ])
            if manual:
                manual_arr = np.asarray(manual, dtype=np.uint8)
                if manual_arr.shape[0] >= count:
                    return manual_arr[:count]
                palette = np.zeros((count, 3), dtype=np.uint8)
                palette[:manual_arr.shape[0]] = manual_arr
                for idx in range(manual_arr.shape[0], count):
                    palette[idx] = TerrainExporter._index_to_rgb(idx)
                return palette

        palette = np.zeros((count, 3), dtype=np.uint8)
        for idx in range(count):
            palette[idx] = TerrainExporter._index_to_rgb(idx)
        return palette

    @staticmethod
    def _index_to_rgb(index: int) -> np.ndarray:
        """Create a deterministic RGB colour from an index."""
        hue = (index * 0.61803398875) % 1.0
        saturation = 0.55 + 0.35 * ((index * 0.37) % 1.0)
        value = 0.70 + 0.25 * ((index * 0.23) % 1.0)
        r, g, b = TerrainExporter._hsv_to_rgb(hue, saturation, value)
        return np.array([r, g, b], dtype=np.uint8)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
        """Convert HSV components (0-1 range) to 8-bit RGB."""
        h = h % 1.0
        s = np.clip(s, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        return (
            int(np.clip(r * 255.0, 0, 255)),
            int(np.clip(g * 255.0, 0, 255)),
            int(np.clip(b * 255.0, 0, 255)),
        )

    @staticmethod
    def _export_png_8bit(data: np.ndarray, filepath: Path):
        """Export as 8-bit PNG."""
        normalized = normalize(data.astype(np.float32), (0, 255))
        # Round to nearest to minimize banding from truncation
        img_data = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        img.save(filepath)
    
    @staticmethod
    def _export_png_16bit(data: np.ndarray, filepath: Path):
        """Export as 16-bit PNG."""
        # Normalize to full 16-bit range using float math, then round
        normalized = normalize(data.astype(np.float32), (0, 65535))
        img_data = np.clip(np.rint(normalized), 0, 65535).astype(np.uint16)
        img = Image.fromarray(img_data, mode='I;16')
        img.save(filepath)
    
    @staticmethod
    def _export_tiff_32bit(data: np.ndarray, filepath: Path):
        """Export as 32-bit float TIFF."""
        img = Image.fromarray(data.astype(np.float32), mode='F')
        img.save(filepath)
