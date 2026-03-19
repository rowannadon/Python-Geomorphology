"""Particle-based erosion simulation for heightmaps."""

import numpy as np
from numba import njit, prange
from typing import Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter


@njit(cache=True)
def _sample_cdf_index(cdf: np.ndarray, value: float) -> int:
    """Locate ``value`` in a monotonically increasing CDF."""
    lo = 0
    hi = cdf.size - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if value <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

@njit(cache=True)
def bilinear_interpolate(heightmap: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation of heightmap at position (x, y)."""
    h, w = heightmap.shape
    
    # Clamp to grid bounds with small margin
    x = max(0.0, min(x, w - 1.001))
    y = max(0.0, min(y, h - 1.001))
    
    # Get integer coordinates
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    
    # Compute interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Bilinear interpolation
    return (heightmap[y0, x0] * (1 - wx) * (1 - wy) +
            heightmap[y0, x1] * wx * (1 - wy) +
            heightmap[y1, x0] * (1 - wx) * wy +
            heightmap[y1, x1] * wx * wy)

@njit(cache=True)
def compute_gradient(heightmap: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Compute terrain gradient at position using central differences."""
    h, w = heightmap.shape
    
    # Use smaller step for more accurate gradients
    step = 0.5
    
    # Sample heights for gradient computation with bounds checking
    x_left = max(0.0, x - step)
    x_right = min(w - 1.0, x + step)
    y_up = max(0.0, y - step)
    y_down = min(h - 1.0, y + step)
    
    h_left = bilinear_interpolate(heightmap, x_left, y)
    h_right = bilinear_interpolate(heightmap, x_right, y)
    h_up = bilinear_interpolate(heightmap, x, y_up)
    h_down = bilinear_interpolate(heightmap, x, y_down)
    
    # Compute gradient (points uphill)
    # Normalize by actual step distance
    grad_x = (h_right - h_left) / (x_right - x_left + 1e-10)
    grad_y = (h_down - h_up) / (y_down - y_up + 1e-10)
    
    return grad_x, grad_y

@njit(cache=True)
def deposit_at_position(heightmap: np.ndarray, x: float, y: float, amount: float, radius: float = 1.5):
    """
    Deposit or erode sediment with a smooth kernel.
    Uses a wider kernel than bilinear to reduce grid artifacts.
    """
    h, w = heightmap.shape
    
    # Get bounds for affected region
    x_min = max(0, int(x - radius))
    x_max = min(w - 1, int(x + radius) + 1)
    y_min = max(0, int(y - radius))
    y_max = min(h - 1, int(y + radius) + 1)
    
    # Compute weights using smooth falloff
    total_weight = 0.0
    for yi in range(y_min, y_max + 1):
        for xi in range(x_min, x_max + 1):
            dist_sq = (xi - x) ** 2 + (yi - y) ** 2
            if dist_sq <= radius * radius:
                # Smooth falloff function (cubic)
                t = np.sqrt(dist_sq) / radius
                weight = max(0.0, 1.0 - t * t * (3.0 - 2.0 * t))
                total_weight += weight
    
    # Apply weighted change
    if total_weight > 0:
        for yi in range(y_min, y_max + 1):
            for xi in range(x_min, x_max + 1):
                dist_sq = (xi - x) ** 2 + (yi - y) ** 2
                if dist_sq <= radius * radius:
                    t = np.sqrt(dist_sq) / radius
                    weight = max(0.0, 1.0 - t * t * (3.0 - 2.0 * t))
                    heightmap[yi, xi] += amount * weight / total_weight

@njit(cache=True)
def simulate_single_droplet(heightmap: np.ndarray,
                           start_x: float,
                           start_y: float,
                           inertia_map: np.ndarray,
                           capacity_map: np.ndarray,
                           deposition_map: np.ndarray,
                           erosion_map: np.ndarray,
                           evaporation_map: np.ndarray,
                           gravity_map: np.ndarray,
                           step_map: np.ndarray,
                           max_delta_map: np.ndarray,
                           max_steps: int,
                           min_slope: float) -> float:
    """Simulate a single water droplet with spatially varying parameters."""
    h, w = heightmap.shape

    # Initialize droplet state
    x, y = start_x, start_y
    dx, dy = 0.0, 0.0  # Direction
    vel = 1.0  # Velocity
    water = 1.0  # Water volume
    sediment = 0.0  # Carried sediment
    
    total_change = 0.0
    
    for step in range(max_steps):
        # Get current position height
        old_height = bilinear_interpolate(heightmap, x, y)
        
        # Stop only if we hit boundary (not ocean)
        if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
            break
        
        # Calculate gradient
        grad_x, grad_y = compute_gradient(heightmap, x, y)

        # Sample local parameter values
        inertia = max(0.0, min(0.999, bilinear_interpolate(inertia_map, x, y)))
        capacity_const = max(0.0, bilinear_interpolate(capacity_map, x, y))
        deposition_const = max(0.0, bilinear_interpolate(deposition_map, x, y))
        erosion_const = max(0.0, bilinear_interpolate(erosion_map, x, y))
        evaporation_const = max(0.0, min(0.999, bilinear_interpolate(evaporation_map, x, y)))
        gravity = max(1e-3, bilinear_interpolate(gravity_map, x, y))
        step_size = max(0.05, bilinear_interpolate(step_map, x, y))
        max_delta = max(1e-6, bilinear_interpolate(max_delta_map, x, y))

        # Update direction with inertia
        new_dx = dx * inertia - grad_x * (1.0 - inertia)
        new_dy = dy * inertia - grad_y * (1.0 - inertia)

        # Normalize direction
        dir_len = np.sqrt(new_dx * new_dx + new_dy * new_dy)
        if dir_len > 1e-6:
            new_dx /= dir_len
            new_dy /= dir_len
        else:
            # Random walk if stuck
            angle = np.random.random() * 2.0 * np.pi
            new_dx = np.cos(angle)
            new_dy = np.sin(angle)
        
        dx, dy = new_dx, new_dy
        
        # Move droplet with adaptive step size
        actual_step = step_size * vel
        new_x = x + dx * actual_step
        new_y = y + dy * actual_step
        
        # Bounds check
        new_x = max(1.0, min(new_x, w - 2.0))
        new_y = max(1.0, min(new_y, h - 2.0))
        
        # Get new height
        new_height = bilinear_interpolate(heightmap, new_x, new_y)
        
        # Calculate height difference
        h_diff = old_height - new_height
        
        # Update velocity based on height change
        # In ocean/flat areas, velocity decreases
        if old_height <= 0.01 and new_height <= 0.01:
            # In ocean - velocity decreases due to water resistance
            vel *= 0.8
        else:
            # On land - normal velocity update
            vel = np.sqrt(vel * vel + h_diff * gravity)
        vel = max(0.1, min(vel, 4.0))  # Clamp velocity
        
        # Calculate carrying capacity
        # Reduced capacity in ocean for delta formation
        if old_height <= 0.01:
            capacity = max(0.0, h_diff) * vel * water * capacity_const * 0.2  # Much lower in ocean
        else:
            capacity = max(0.0, h_diff) * vel * water * capacity_const
        
        # Erosion or deposition
        if capacity > sediment and old_height > 0.01:  # Only erode on land
            # Erosion (only on land, not ocean floor)
            erode_amount = min((capacity - sediment) * erosion_const, h_diff * 0.5)
            
            # Don't erode below sea level
            if erode_amount > 0:
                erode_amount = min(erode_amount, old_height, max_delta)
                deposit_at_position(heightmap, x, y, -erode_amount, radius=1.2)
                sediment += erode_amount
                total_change -= erode_amount
        elif sediment > capacity:
            # Deposition (can happen anywhere, including ocean)
            deposit_amount = (sediment - capacity) * deposition_const
            
            # Increase deposition rate in very shallow water (delta formation)
            if old_height <= 0.05 and old_height >= 0.0:
                deposit_amount *= 2.0  # Enhanced deposition in shallows
            
            if deposit_amount > 0:
                deposit_amount = min(deposit_amount, sediment)
                deposit_at_position(heightmap, x, y, deposit_amount, radius=1.5)
                sediment -= deposit_amount
                total_change += deposit_amount
        
        # Update position
        x, y = new_x, new_y
        
        # Evaporate water (less evaporation in ocean)
        if old_height <= 0.01:
            water *= min(evaporation_const, 0.99)  # Slower evaporation in water
        else:
            water *= evaporation_const

        if water < 0.01:
            break

    return total_change


@njit(cache=True)
def _simulate_droplet_batch(heightmap: np.ndarray,
                            spawn_cdf: np.ndarray,
                            inertia_map: np.ndarray,
                            capacity_map: np.ndarray,
                            deposition_map: np.ndarray,
                            erosion_map: np.ndarray,
                            evaporation_map: np.ndarray,
                            gravity_map: np.ndarray,
                            step_map: np.ndarray,
                            max_delta_map: np.ndarray,
                            droplet_count: int,
                            max_steps: int,
                            min_slope: float):
    """Simulate a batch of droplets entirely inside compiled code."""
    h, w = heightmap.shape
    total_cells = h * w
    for _ in range(droplet_count):
        idx = _sample_cdf_index(spawn_cdf, np.random.random())
        if idx >= total_cells:
            idx = total_cells - 1
        y_start = float(idx // w) + (np.random.random() - 0.5)
        x_start = float(idx % w) + (np.random.random() - 0.5)
        simulate_single_droplet(
            heightmap,
            x_start,
            y_start,
            inertia_map,
            capacity_map,
            deposition_map,
            erosion_map,
            evaporation_map,
            gravity_map,
            step_map,
            max_delta_map,
            max_steps,
            min_slope,
        )


@njit(cache=True)
def _compute_flow_accumulation_numba(heightmap: np.ndarray) -> np.ndarray:
    """Simple D8-style flow accumulation used to bias droplet spawning."""
    h, w = heightmap.shape
    flow = np.zeros_like(heightmap, dtype=np.float64)
    flat = heightmap.reshape(h * w)
    indices = np.argsort(-flat)

    for idx_pos in range(indices.size):
        idx = indices[idx_pos]
        y = idx // w
        x = idx % w

        if heightmap[y, x] <= 0.001:
            continue

        min_h = heightmap[y, x]
        min_y = y
        min_x = x

        for dy in range(-1, 2):
            ny = y + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-1, 2):
                nx = x + dx
                if dx == 0 and dy == 0:
                    continue
                if nx < 0 or nx >= w:
                    continue
                neighbor_h = heightmap[ny, nx]
                if neighbor_h < min_h:
                    min_h = neighbor_h
                    min_y = ny
                    min_x = nx

        if min_x != x or min_y != y:
            flow[min_y, min_x] += flow[y, x] + 1.0

    return flow

class ParticleErosion:
    """Improved particle-based erosion simulation."""
    
    def __init__(self,
                 iterations: int = 80000,  # More droplets for smoother results
                 inertia: float = 0.3,  # Directional inertia
                 capacity_const: float = 8.0,  # Sediment capacity multiplier
                 deposition_const: float = 0.2,  # Deposition rate
                 erosion_const: float = 0.4,  # Erosion rate
                 evaporation_const: float = 0.98,  # Water retention per step
                 gravity: float = 10.0,
                 max_lifetime: int = 60,  # Max steps per droplet
                 step_size: float = 0.3,  # Smaller steps for smoother flow
                 max_delta: float = 0.05,  # Maximum erosion per step
                 min_slope: float = 0.0001,
                 blur_iterations: int = 1):  # Post-process smoothing
        """
        Initialize erosion parameters with improved defaults.
        """
        self.iterations = iterations
        self.inertia = inertia
        self.capacity_const = capacity_const
        self.deposition_const = deposition_const
        self.erosion_const = erosion_const
        self.evaporation_const = evaporation_const
        self.gravity = gravity
        self.max_lifetime = max_lifetime
        self.step_size = step_size
        self.max_delta = max_delta
        self.min_slope = min_slope
        self.blur_iterations = blur_iterations
    
    def erode(self, heightmap: np.ndarray,
            parameter_maps: Optional[Dict[str, np.ndarray]] = None,
            progress_callback: Optional[callable] = None,
            cancel_check: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply particle erosion with ocean deposition allowed.
        """
        h, w = heightmap.shape
        eroded = heightmap.copy().astype(np.float64)
        initial_height = eroded.copy()

        # Pre-compute flow accumulation for weighted spawn points
        flow_acc = self._compute_flow_accumulation(eroded)
        
        # Spawn probability - mostly on land but some in shallow water
        spawn_prob = np.ones_like(eroded)
        
        # Much lower spawn rate in deep ocean
        deep_ocean = eroded <= -0.1
        spawn_prob[deep_ocean] = 0.01
        
        # Higher spawn rate in higher areas and flow accumulation zones
        land_areas = eroded > 0.01
        spawn_prob[land_areas] *= (1.0 + eroded[land_areas] * 2.0)  # More droplets at higher elevations
        spawn_prob *= (1.0 + np.tanh(flow_acc * 0.1))  # And along flow paths
        
        # Normalize probabilities
        if spawn_prob.sum() > 0:
            spawn_prob = spawn_prob / spawn_prob.sum()
        else:
            spawn_prob = np.ones_like(spawn_prob) / spawn_prob.size
        
        # Flatten once and build a CDF so compiled batches can sample directly.
        total_cells = h * w
        spawn_prob_flat = np.ascontiguousarray(spawn_prob.reshape(-1).astype(np.float64, copy=False))
        spawn_cdf = np.cumsum(spawn_prob_flat, dtype=np.float64)
        if spawn_cdf.size == 0 or spawn_cdf[-1] <= 0.0:
            spawn_cdf = np.linspace(1.0 / total_cells, 1.0, total_cells, dtype=np.float64)
        else:
            spawn_cdf /= spawn_cdf[-1]
            spawn_cdf[-1] = 1.0
        
        # Prepare per-cell parameter maps (fallback to uniform values)
        maps = parameter_maps or {}

        def prepare_map(key: str, default: float) -> np.ndarray:
            arr = maps.get(key)
            if arr is None:
                arr = np.full_like(eroded, default, dtype=np.float64)
            else:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != eroded.shape:
                    raise ValueError(f"Parameter map '{key}' has mismatched shape {arr.shape}, expected {eroded.shape}")
            return np.ascontiguousarray(arr)

        inertia_map = prepare_map('erosion_inertia', self.inertia)
        capacity_map = prepare_map('erosion_capacity', self.capacity_const)
        deposition_map = prepare_map('erosion_deposition_rate', self.deposition_const)
        erosion_map = prepare_map('erosion_rate', self.erosion_const)
        evaporation_map = prepare_map('erosion_evaporation', self.evaporation_const)
        gravity_map = prepare_map('erosion_gravity', self.gravity)
        step_map = prepare_map('erosion_step_size', self.step_size)
        max_delta_map = prepare_map('max_delta', self.max_delta)

        # Simulate many droplets in compiled batches to avoid Python call overhead.
        batch_size = min(1000, max(self.iterations, 1))
        num_batches, remainder = divmod(self.iterations, batch_size)
        progress_denominator = max(num_batches + (1 if remainder else 0), 1)

        for batch_idx in range(num_batches):
            if cancel_check and cancel_check():
                raise RuntimeError("Execution cancelled.")
            if progress_callback and batch_idx % 10 == 0:
                progress = int(70 + (batch_idx / progress_denominator) * 20)
                progress_callback(progress, f"Erosion: {batch_idx * batch_size}/{self.iterations} droplets...")

            _simulate_droplet_batch(
                eroded,
                spawn_cdf,
                inertia_map,
                capacity_map,
                deposition_map,
                erosion_map,
                evaporation_map,
                gravity_map,
                step_map,
                max_delta_map,
                batch_size,
                self.max_lifetime,
                self.min_slope,
            )

        if remainder:
            if cancel_check and cancel_check():
                raise RuntimeError("Execution cancelled.")
            if progress_callback:
                progress = int(70 + (num_batches / progress_denominator) * 20)
                progress_callback(progress, f"Erosion: {num_batches * batch_size}/{self.iterations} droplets...")
            _simulate_droplet_batch(
                eroded,
                spawn_cdf,
                inertia_map,
                capacity_map,
                deposition_map,
                erosion_map,
                evaporation_map,
                gravity_map,
                step_map,
                max_delta_map,
                remainder,
                self.max_lifetime,
                self.min_slope,
            )
        
        # Post-processing: smooth to reduce noise while preserving features
        if self.blur_iterations > 0:
            for _ in range(self.blur_iterations):
                if cancel_check and cancel_check():
                    raise RuntimeError("Execution cancelled.")
                # Selective smoothing - less on steep slopes
                grad_mag = np.sqrt(
                    np.gradient(eroded, axis=0)**2 + 
                    np.gradient(eroded, axis=1)**2
                )
                smooth_weight = np.exp(-grad_mag * 10.0)
                
                smoothed = gaussian_filter(eroded, sigma=0.5)
                eroded = eroded * (1 - smooth_weight * 0.3) + smoothed * smooth_weight * 0.3
        
        # Ensure ocean floor doesn't go negative (but can build up with sediment)
        eroded = np.maximum(eroded, 0.0)
        
        # Calculate deposition map
        deposition = eroded - initial_height
        
        return eroded, deposition
    
    def _compute_flow_accumulation(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Simple flow accumulation to identify natural drainage paths.
        """
        return _compute_flow_accumulation_numba(np.asarray(heightmap, dtype=np.float64))
