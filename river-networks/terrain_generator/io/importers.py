"""Import functionality for terrain data."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from scipy.ndimage import zoom

from ..core.utils import normalize

class HeightmapImporter:
    """Handles heightmap import and processing."""
    
    @staticmethod
    def load_heightmap(filepath: str, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process a heightmap image.
        
        Returns:
            Tuple of (heightmap, land_mask)
            - heightmap: Normalized height values (0-1)
            - land_mask: Boolean mask where True = land
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Heightmap file not found: {filepath}")
        
        # Load image
        try:
            img = Image.open(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Convert to a numpy array while preserving source precision
        # - If the image is 16-bit (mode 'I;16' or info), keep uint16 and scale later
        # - If the image is 32-bit float ('F') or 32-bit int ('I'), keep precision
        # - Otherwise, convert to 8-bit grayscale
        if img.mode in ('I;16', 'I', 'F'):
            img_array_raw = np.array(img)
        else:
            img_gray = img.convert('L')
            img_array_raw = np.array(img_gray)

        # Normalize raw array to float32 in [0,1] without quantizing unnecessarily
        if img_array_raw.dtype == np.uint16:
            img_array = (img_array_raw.astype(np.float32) / 65535.0)
        elif img_array_raw.dtype == np.uint8:
            img_array = (img_array_raw.astype(np.float32) / 255.0)
        elif img_array_raw.dtype in (np.int32, np.uint32, np.float32, np.float64):
            # Dynamic normalization for generic numeric inputs
            arr = img_array_raw.astype(np.float32, copy=False)
            amax = float(arr.max())
            amin = float(arr.min())
            if amax > amin:
                img_array = (arr - amin) / (amax - amin)
            else:
                img_array = np.zeros_like(arr, dtype=np.float32)
        else:
            # Fallback: ensure float32
            img_array = img_array_raw.astype(np.float32)
        
        # Resize to target dimensions if necessary
        if img_array.shape != target_shape:
            zoom_factors = (target_shape[0] / img_array.shape[0], 
                          target_shape[1] / img_array.shape[1])
            img_array = zoom(img_array, zoom_factors, order=1)
        
        # Ensure final 0-1 range (preserves relative precision)
        heightmap = normalize(img_array.astype(np.float32), bounds=(0, 1))
        
        # Create land mask (anything not pure black is considered land)
        # Using a small threshold to account for compression artifacts
        land_mask = heightmap > 0.01
        
        return heightmap, land_mask
    
    @staticmethod
    def process_for_terrain(heightmap: np.ndarray, land_mask: np.ndarray,
                          blend_with_noise: bool = False,
                          noise_array: Optional[np.ndarray] = None,
                          blend_factor: float = 1.0) -> np.ndarray:
        """
        Process imported heightmap for terrain generation.
        
        Args:
            heightmap: Imported heightmap (0-1)
            land_mask: Boolean mask for land areas
            blend_with_noise: Whether to blend with procedural noise
            noise_array: Optional procedural noise to blend with
            blend_factor: 0 = full noise, 1 = full import
        """
        # Apply land mask
        processed = heightmap * land_mask
        
        # Optionally blend with procedural noise
        if blend_with_noise and noise_array is not None:
            processed = (blend_factor * processed + 
                        (1 - blend_factor) * noise_array)
        
        return processed
