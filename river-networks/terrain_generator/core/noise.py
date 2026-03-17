"""Noise generation functions for terrain synthesis."""

import numpy as np
from scipy.ndimage import zoom
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class NoiseGenerator(ABC):
    """Abstract base class for noise generators."""
    
    @abstractmethod
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise for the given shape."""
        pass

class ConsistentFBMNoise(NoiseGenerator):
    """FBM noise that maintains consistent features across dimensions."""
    
    BASE_RESOLUTION = 128  # Base resolution for coarse features
    
    def __init__(self, scale: float = -2.0, octaves: int = 6, 
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 lower: float = -np.inf, upper: float = np.inf,
                 seed_offset: int = 0, base_seed: int = 42):  # Add base_seed parameter
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.lower = lower
        self.upper = upper
        self.seed_offset = seed_offset  # Unique offset for each noise type
        self.base_seed = base_seed  # Main seed from parameters
    
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise with consistent coarse features."""
        target_dim = int(np.mean(shape))
        
        # Determine octave split based on target dimension
        if target_dim <= self.BASE_RESOLUTION:
            # For low res, generate everything at target resolution
            return self._generate_fbm(shape, self.octaves, start_octave=0)
        
        # For high res, split octaves between base and detail
        # More octaves at base = more consistent large features
        # Adjust this ratio to control consistency vs detail
        ratio = np.clip(self.BASE_RESOLUTION / target_dim, 0.3, 0.7)
        coarse_octaves = max(1, int(self.octaves * ratio))
        detail_octaves = self.octaves - coarse_octaves
        
        # Generate coarse features at base resolution
        base_shape = (self.BASE_RESOLUTION, self.BASE_RESOLUTION)
        coarse_noise = self._generate_fbm(base_shape, coarse_octaves, 
                                         start_octave=0)
        
        # Upsample to target resolution
        zoom_factor = (shape[0] / base_shape[0], shape[1] / base_shape[1])
        coarse_upsampled = zoom(coarse_noise, zoom_factor, order=3)
        
        if detail_octaves > 0:
            # Add high-frequency detail at full resolution
            detail_noise = self._generate_fbm(shape, detail_octaves, 
                                             start_octave=coarse_octaves)
            
            # Combine with appropriate weighting
            detail_weight = self.persistence ** coarse_octaves
            result = coarse_upsampled + detail_noise * detail_weight * 0.5
        else:
            result = coarse_upsampled
        
        return self._normalize(result)
    
    def _generate_fbm(self, shape: Tuple[int, ...], num_octaves: int, 
                      start_octave: int = 0) -> np.ndarray:
        """Generate FBM with specified octaves."""
        result = np.zeros(shape)
        
        # Calculate dimension scaling
        avg_dim = np.mean(shape)
        dim_scale = 256.0 / avg_dim
        
        # Start at the appropriate octave
        amplitude = self.persistence ** start_octave
        frequency = (2.0 ** self.scale) * (self.lacunarity ** start_octave) * dim_scale
        
        for i in range(num_octaves):
            octave_index = start_octave + i
            octave = self._generate_octave(shape, frequency, octave_index)
            result += octave * amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return result
    
    def _generate_octave(self, shape: Tuple[int, ...], frequency: float, 
                        octave_index: int) -> np.ndarray:
        """Generate a single octave with consistent seeding."""
        freqs = tuple(np.fft.fftfreq(n) for n in shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        
        scaled_lower = self.lower * frequency if self.lower != -np.inf else -np.inf
        scaled_upper = self.upper * frequency if self.upper != np.inf else np.inf
        
        in_bounds = (freq_radial > scaled_lower) & (freq_radial < scaled_upper)
        
        envelope = np.zeros_like(freq_radial)
        mask = (freq_radial != 0) & in_bounds
        envelope[mask] = np.power(freq_radial[mask] * frequency, self.scale)
        
        # Incorporate base_seed into the deterministic seeding
        # This ensures different seeds produce different terrain
        # while maintaining consistency across dimensions
        seed = (self.base_seed * 10000 + self.seed_offset * 1000 + octave_index * 100) % (2**32)
        rng = np.random.RandomState(seed)
        phase_noise = np.exp(2j * np.pi * rng.rand(*shape))
        
        octave = np.real(np.fft.ifftn(np.fft.fftn(phase_noise) * envelope))
        
        if octave.max() != octave.min():
            octave = 2 * (octave - octave.min()) / (octave.max() - octave.min()) - 1
        
        return octave
    
    @staticmethod
    def _normalize(array: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Normalize array values to specified bounds."""
        if array.max() == array.min():
            return np.full_like(array, bounds[0])
        return np.interp(array, (array.min(), array.max()), bounds)

# Keep the original FBMNoise class for backward compatibility or if needed
class FBMNoise(ConsistentFBMNoise):
    """Original FBM noise - can be used where consistency isn't needed."""
    pass