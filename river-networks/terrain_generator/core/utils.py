"""Utility functions for terrain generation."""

import numpy as np
import scipy.spatial
from typing import Tuple, List
from numba import njit, prange
import matplotlib.tri as mtri
from functools import lru_cache

def normalize(x: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Renormalizes the values of x to bounds."""
    if x.max() == x.min():
        return np.full_like(x, bounds[0])
    return np.interp(x, (x.min(), x.max()), bounds)

def gaussian_blur(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Performs a gaussian blur of a."""
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

def gaussian_gradient(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Returns the gradient of the gaussian blur of a encoded as a complex number."""
    [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    dg = lambda x: g(x) * (x / sigma2)
    
    fa = np.fft.fft2(a)
    dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
    dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
    return 1j * dx + dy

def lerp(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """Linear interpolation of x to y with respect to a."""
    return (1.0 - a) * x + a * y

def make_grid_points(shape: Tuple[int, int]) -> np.ndarray:
    """Returns a list of grid coordinates for every (x, y) position."""
    [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    return grid_points

def bump(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns an array with a bump centered in the middle."""
    [y, x] = np.meshgrid(*map(np.arange, shape))
    r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
    c = min(shape) / 2
    return np.tanh(np.maximum(c - r, 0.0) / sigma)

def dist_to_mask(mask: np.ndarray) -> np.ndarray:
    """Returns distance to nearest False value for all True values in mask."""
    border_mask = (np.maximum.reduce([
        np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
        np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
    border_points = np.column_stack(np.where(border_mask > 0))
    
    if len(border_points) == 0:
        return np.zeros_like(mask, dtype=float)
    
    kdtree = scipy.spatial.cKDTree(border_points)
    grid_points = make_grid_points(mask.shape)
    
    return kdtree.query(grid_points)[0].reshape(mask.shape)

@njit(cache=True, fastmath=True)
def _poisson_disc_numba(H, W, radius, retries, seed):
    if radius <= 0.0:
        return np.empty((0, 2), np.float32)

    cell_size = radius / np.sqrt(2.0)
    if cell_size <= 0.0:
        return np.empty((0, 2), np.float32)

    grid_rows = int(np.ceil(H / cell_size))
    grid_cols = int(np.ceil(W / cell_size))
    grid = -np.ones((grid_rows, grid_cols), np.int32)

    offsets = np.array([
        [ 0,  0], [ 0, -1], [ 0,  1], [-1,  0], [ 1,  0],
        [-1, -1], [-1,  1], [ 1, -1], [ 1,  1],
        [-2,  0], [ 2,  0], [ 0, -2], [ 0,  2]
    ], dtype=np.int32)

    max_pts = grid_rows * grid_cols
    pts = np.empty((max_pts, 2), np.float32)
    active = np.empty(max_pts, np.int32)
    n_pts = 0
    n_active = 0

    r2 = radius * radius
    tau = 2.0 * np.pi
    if seed >= 0:
        np.random.seed(seed)

    # first point
    x = np.random.random() * W
    y = np.random.random() * H
    cx = int(x / cell_size)
    cy = int(y / cell_size)
    grid[cy, cx] = n_pts
    pts[n_pts, 0] = x
    pts[n_pts, 1] = y
    active[n_active] = n_pts
    n_active += 1
    n_pts += 1

    while n_active > 0:
        pidx = active[n_active - 1]
        n_active -= 1
        px = pts[pidx, 0]
        py = pts[pidx, 1]

        left = retries
        while left > 0:
            # single-candidate loop is usually optimal under JIT; batching gives no Python benefit anymore
            ang = np.random.random() * tau
            rr = radius * np.sqrt(1.0 + 3.0 * np.random.random())
            x = px + rr * np.cos(ang)
            y = py + rr * np.sin(ang)

            if not (0.0 <= x < W and 0.0 <= y < H):
                left -= 1
                continue

            cx = int(x / cell_size)
            cy = int(y / cell_size)
            if cx < 0 or cy < 0 or cx >= grid_cols or cy >= grid_rows:
                left -= 1
                continue

            ok = True
            for k in range(offsets.shape[0]):
                nx = cx + offsets[k, 0]
                ny = cy + offsets[k, 1]
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid[ny, nx]
                    if j != -1:
                        dx = pts[j, 0] - x
                        dy = pts[j, 1] - y
                        if dx * dx + dy * dy <= r2:
                            ok = False
                            break

            if ok:
                grid[cy, cx] = n_pts
                pts[n_pts, 0] = x
                pts[n_pts, 1] = y
                active[n_active] = n_pts
                n_active += 1
                n_pts += 1

            left -= 1

    return pts[:n_pts]

def poisson_disc_sampling(shape: Tuple[int, int], radius: float, retries: int = 16, seed: int = -1) -> np.ndarray:
    H, W = int(shape[0]), int(shape[1])
    return _poisson_disc_numba(H, W, float(radius), int(retries), int(seed))
# def render_triangulation(shape: Tuple[int, int], tri, values: np.ndarray) -> np.ndarray:
#     """Renders values for each triangle on an array."""
#     import matplotlib.tri
#     points = make_grid_points(shape)
#     triangulation = matplotlib.tri.Triangulation(
#         tri.points[:,0], tri.points[:,1], tri.simplices)
#     interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
#     return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)

@lru_cache(maxsize=8)
def _grid_xy(shape):
    """Cached X,Y for the given (rows, cols) shape."""
    nrows, ncols = shape
    # Use sparse=True to avoid allocating full X/Y until needed by broadcasting
    X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows), indexing="xy")
    return X, Y

def render_triangulation(shape, tri, values, *, triangulation=None, fill_value=0.0):
    """
    Faster renderer:
      - avoids Nx2 point stack
      - reuses cached grid
      - can reuse a prebuilt Triangulation
    """
    if triangulation is None:
        # Accept either a ready Triangulation or a (points,simplices)-like object
        if isinstance(tri, mtri.Triangulation):
            triangulation = tri
        else:
            triangulation = mtri.Triangulation(tri.points[:, 0], tri.points[:, 1], tri.simplices)

    X, Y = _grid_xy(shape)                # shapes ~ (rows,1) and (1,cols) due to sparse=True
    zi = mtri.LinearTriInterpolator(triangulation, values)(X, Y)  # returns (rows, cols)
    return zi.filled(fill_value)

def identify_inland_seas(land_mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Identify inland seas (water bodies not connected to ocean).
    
    Args:
        land_mask: Boolean array where True = land, False = water
        
    Returns:
        ocean_mask: Boolean mask of the main ocean
        inland_seas: List of boolean masks for each inland sea
    """
    from skimage import measure
    
    # Label water bodies (inverse of land mask)
    water_mask = ~land_mask
    labeled_water = measure.label(water_mask, connectivity=1)
    
    # Find the ocean (largest water body touching the edge)
    edge_labels = set()
    h, w = land_mask.shape
    
    # Collect labels that touch the edges
    edge_labels.update(labeled_water[0, :])    # top
    edge_labels.update(labeled_water[-1, :])   # bottom  
    edge_labels.update(labeled_water[:, 0])    # left
    edge_labels.update(labeled_water[:, -1])   # right
    edge_labels.discard(0)  # Remove background
    
    # Ocean is the largest edge-touching water body
    ocean_label = 0
    ocean_size = 0
    for label in edge_labels:
        size = np.sum(labeled_water == label)
        if size > ocean_size:
            ocean_size = size
            ocean_label = label
    
    # If no edge water, pick the largest water body
    if ocean_label == 0 and labeled_water.max() > 0:
        labels, counts = np.unique(labeled_water[labeled_water > 0], return_counts=True)
        ocean_label = labels[np.argmax(counts)]
    
    ocean_mask = (labeled_water == ocean_label)
    
    # Find all inland seas
    inland_seas = []
    for label in np.unique(labeled_water):
        if label > 0 and label != ocean_label:
            inland_seas.append(labeled_water == label)
    
    return ocean_mask, inland_seas

def _raised_cosine_profile(distance_ratio: float, softness: float = 1.0) -> float:
    """Return a softened raised-cosine profile for a normalized distance."""
    t = min(max(float(distance_ratio), 0.0), 1.0) ** (1.0 / max(float(softness), 0.05))
    return float(0.5 * (np.cos(np.pi * t) + 1.0))


def _channel_carve_profile(distance_ratio: float, channel_softness: float, slope_softness: float) -> float:
    """Blend a base channel profile with a bank-specific falloff near the edges."""
    ratio = min(max(float(distance_ratio), 0.0), 1.0)
    base_profile = _raised_cosine_profile(ratio, channel_softness)
    bank_profile = _raised_cosine_profile(ratio, slope_softness)

    # Keep the channel center driven mostly by the original profile and let the
    # bank control take over progressively toward the edges.
    edge_mix = min(max((ratio - 0.35) / 0.65, 0.0), 1.0)
    edge_mix = edge_mix * edge_mix * (3.0 - 2.0 * edge_mix)
    return float(base_profile * (1.0 - edge_mix) + bank_profile * edge_mix)


def carve_channel_to_ocean(heightmap: np.ndarray, land_mask: np.ndarray,
                          inland_sea_mask: np.ndarray, ocean_mask: np.ndarray,
                          carve_depth: float = 0.05,
                          channel_width: float = 5.0,
                          channel_falloff: float = 1.2,
                          slope_falloff: float = 1.2) -> np.ndarray:
    """
    Carve a channel from inland sea to ocean.
    
    Args:
        heightmap: Current terrain heightmap
        land_mask: Boolean mask of land areas
        inland_sea_mask: Boolean mask of the specific inland sea
        ocean_mask: Boolean mask of the ocean
        carve_depth: How deep to carve the channel
        channel_width: Channel radius in pixels/cells
        channel_falloff: Edge softness for the carved channel profile
        slope_falloff: Bank-specific softness used near the channel edges
        
    Returns:
        Modified heightmap with carved channel
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation
    from skimage.graph import route_through_array
    
    h, w = heightmap.shape
    
    # Find closest points between inland sea and ocean
    inland_dist = distance_transform_edt(~inland_sea_mask)
    ocean_dist = distance_transform_edt(~ocean_mask)
    
    # Find a point on the inland sea edge
    dilated_inland = binary_dilation(inland_sea_mask)
    inland_edge = dilated_inland & ~inland_sea_mask
    if not np.any(inland_edge):
        return heightmap
    
    inland_edge_points = np.argwhere(inland_edge)
    # Pick the point closest to ocean
    min_dist_idx = np.argmin([ocean_dist[p[0], p[1]] for p in inland_edge_points])
    start_point = tuple(inland_edge_points[min_dist_idx])
    
    # Find the closest ocean point
    dilated_ocean = binary_dilation(ocean_mask)
    ocean_edge = dilated_ocean & ~ocean_mask
    if not np.any(ocean_edge):
        return heightmap
    
    ocean_edge_points = np.argwhere(ocean_edge)
    distances = [np.sqrt((p[0]-start_point[0])**2 + (p[1]-start_point[1])**2) 
                 for p in ocean_edge_points]
    end_point = tuple(ocean_edge_points[np.argmin(distances)])
    
    # Create cost array for pathfinding
    # Lower cost for lower terrain, high cost for going uphill
    cost = np.ones_like(heightmap)
    # Normalize heightmap for cost calculation
    norm_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
    cost = 1.0 + norm_height * 10.0  # Prefer lower areas
    
    # Find path using least-cost pathfinding
    try:
        path_indices, _ = route_through_array(
            cost, start_point, end_point, fully_connected=True
        )
    except:
        # Fallback to straight line if pathfinding fails
        num_points = int(np.sqrt((end_point[0]-start_point[0])**2 + 
                                 (end_point[1]-start_point[1])**2))
        if num_points > 0:
            t = np.linspace(0, 1, num_points)
            path_indices = np.array([
                (int(start_point[0] * (1-ti) + end_point[0] * ti),
                 int(start_point[1] * (1-ti) + end_point[1] * ti))
                for ti in t
            ])
        else:
            return heightmap
    
    # Carve the channel
    modified = heightmap.copy()
    
    # Create a smooth channel profile.
    channel_radius = max(float(channel_width), 0.0)
    falloff_softness = max(float(channel_falloff), 0.05)
    slope_softness = max(float(slope_falloff), 0.05)
    channel_radius_cells = max(int(np.ceil(channel_radius)), 0)

    for y, x in path_indices:
        # Carve the channel point and its neighbors
        for dy in range(-channel_radius_cells, channel_radius_cells + 1):
            for dx in range(-channel_radius_cells, channel_radius_cells + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if channel_radius <= 1e-6:
                        if dy == 0 and dx == 0:
                            modified[ny, nx] = max(0.0, modified[ny, nx] - carve_depth)
                        continue
                    dist = np.hypot(dy, dx)
                    if dist <= channel_radius:
                        carve_amount = float(carve_depth) * _channel_carve_profile(
                            dist / channel_radius,
                            falloff_softness,
                            slope_softness,
                        )
                        modified[ny, nx] = max(0.0, modified[ny, nx] - carve_amount)
    
    # Smooth the carved area
    from scipy.ndimage import gaussian_filter
    # Create mask of carved area
    carved_mask = np.zeros_like(land_mask, dtype=bool)
    for y, x in path_indices:
        for dy in range(-channel_radius_cells - 1, channel_radius_cells + 2):
            for dx in range(-channel_radius_cells - 1, channel_radius_cells + 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    carved_mask[ny, nx] = True
    
    # Apply smoothing only to carved area
    if np.any(carved_mask):
        smoothed = gaussian_filter(modified, sigma=1.0)
        modified[carved_mask] = smoothed[carved_mask]
    
    return modified


def connect_inland_seas(heightmap: np.ndarray, land_mask: np.ndarray,
                       min_sea_size: int = 20,
                       carve_depth: float = 0.1,
                       channel_width: float = 5.0,
                       channel_falloff: float = 1.2,
                       slope_falloff: float = 1.2,
                       fill_height: float = 0.01,
                       water_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Connect all inland seas to the ocean or fill them if too small.
    
    Args:
        heightmap: Terrain heightmap  
        land_mask: Boolean mask where True = land
        min_sea_size: Minimum size for inland seas (smaller ones are filled)
        carve_depth: Channel depth relative to water level
        channel_width: Channel radius in pixels/cells
        channel_falloff: Edge softness for the carved channel profile
        slope_falloff: Bank-specific softness used near the channel edges
        fill_height: Height assigned to filled lakes above water level
        water_level: Absolute sea level used by the raster
        
    Returns:
        Modified heightmap and land mask
    """
    ocean_mask, inland_seas = identify_inland_seas(land_mask)
    
    if not inland_seas:
        # No inland seas found
        return heightmap, land_mask
    
    dtype = np.asarray(heightmap).dtype
    relative_heightmap = np.asarray(heightmap, dtype=np.float32) - float(water_level)
    modified_heightmap = relative_heightmap.copy()
    
    print(f"Found {len(inland_seas)} inland water bodies")
    
    for i, sea_mask in enumerate(inland_seas):
        sea_size = np.sum(sea_mask)
        
        if sea_size < min_sea_size:
            # Fill small lakes
            print(f"  Filling small lake {i+1} (size: {sea_size})")
            # Set to slightly above water level
            modified_heightmap[sea_mask] = float(fill_height)
        else:
            # Carve channel for larger seas
            print(f"  Carving channel for inland sea {i+1} (size: {sea_size})")
            modified_heightmap = carve_channel_to_ocean(
                modified_heightmap, land_mask, sea_mask, ocean_mask,
                carve_depth=float(carve_depth),
                channel_width=float(channel_width),
                channel_falloff=float(channel_falloff),
                slope_falloff=float(slope_falloff),
            )
    
    # Recompute land mask after modifications
    # Water is anything at or below 0
    new_land_mask = modified_heightmap > 0.001
    
    # Ensure carved channels are properly marked as water
    modified_heightmap = np.where(new_land_mask, modified_heightmap, 0.0)
    modified_heightmap = np.where(
        new_land_mask,
        modified_heightmap + float(water_level),
        float(water_level),
    ).astype(dtype, copy=False)
    
    return modified_heightmap, new_land_mask

@njit(cache=True)
def _finite_min_max(a: np.ndarray) -> tuple:
    """Return (min, max) over finite values of a 2D float array."""
    h, w = a.shape
    mn = np.inf
    mx = -np.inf
    for y in range(h):
        for x in range(w):
            v = a[y, x]
            if np.isfinite(v):
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
    if not np.isfinite(mn) or not np.isfinite(mx):
        mn = 0.0
        mx = 0.0
    return mn, mx

@njit(cache=True, parallel=True)
def _gray_to_rgba_norm(src: np.ndarray, land: np.ndarray, out_rgba: np.ndarray):
    """
    Normalize float32 src to [0,255] and write as grayscale RGBA.
    Alpha=255 on land, 0 off land.
    """
    h, w = src.shape
    mn, mx = _finite_min_max(src)
    scale = 0.0
    if mx > mn:
        scale = 255.0 / (mx - mn)

    for y in prange(h):
        for x in range(w):
            v = src[y, x]
            if np.isfinite(v) and mx > mn:
                g = int((v - mn) * scale + 0.5)  # round
            else:
                g = 0
            if g < 0: g = 0
            if g > 255: g = 255
            a = 255 if land[y, x] else 0
            i = (y * w + x) * 4
            out_rgba.flat[i + 0] = g
            out_rgba.flat[i + 1] = g
            out_rgba.flat[i + 2] = g
            out_rgba.flat[i + 3] = a
            
@njit(cache=True, parallel=True)
def _deposition_to_rgba(src: np.ndarray, land: np.ndarray, out_rgba: np.ndarray):
    """
    Map deposition/erosion to grayscale:
      negative (erosion) -> darker than 128
      positive (deposition) -> brighter than 128
      neutral -> 128
    Alpha=255 on land, 0 off.
    """
    h, w = src.shape
    # symmetric normalization around 0
    mn, mx = _finite_min_max(src)
    rng = mx
    if -mn > rng:
        rng = -mn
    scale = 0.0
    if rng > 0.0:
        scale = 127.0 / rng

    for y in prange(h):
        for x in range(w):
            v = src[y, x]
            if np.isfinite(v) and rng > 0.0:
                g = int(128.0 + v * scale + (0.5 if v >= 0 else -0.5))
            else:
                g = 128
            if g < 0: g = 0
            if g > 255: g = 255
            a = 255 if land[y, x] else 0
            i = (y * w + x) * 4
            out_rgba.flat[i + 0] = g
            out_rgba.flat[i + 1] = g
            out_rgba.flat[i + 2] = g
            out_rgba.flat[i + 3] = a

@njit(cache=True)
def _hsv_to_rgb_u8(h: float, s: float, v: float) -> tuple:
    # h in [0,1), s,v in [0,1]
    h = h - np.floor(h)
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
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
    return int(r * 255.0 + 0.5), int(g * 255.0 + 0.5), int(b * 255.0 + 0.5)

@njit(cache=True)
def _index_to_rgb_u8(idx: int) -> tuple:
    # Deterministic, well-distributed palette similar in spirit
    # to the exporter’s HSV-based mapping.
    hue = (idx * 0.61803398875) % 1.0
    sat = 0.55 + 0.35 * (((idx * 0.37) % 1.0))
    val = 0.70 + 0.25 * (((idx * 0.23) % 1.0))
    return _hsv_to_rgb_u8(hue, sat, val)

@njit(cache=True)
def _build_palette_u8(n: int) -> np.ndarray:
    if n <= 0:
        n = 1
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        r, g, b = _index_to_rgb_u8(i)
        pal[i, 0] = r
        pal[i, 1] = g
        pal[i, 2] = b
    return pal

@njit(cache=True, parallel=True)
def _labels_to_rgba(labels: np.ndarray, land: np.ndarray, palette: np.ndarray, out_rgba: np.ndarray):
    h, w = labels.shape
    for y in prange(h):
        for x in range(w):
            idx = labels[y, x]
            if idx < 0:
                idx = 0
            if idx >= palette.shape[0]:
                idx = (idx % palette.shape[0]) if palette.shape[0] > 0 else 0
            r = palette[idx, 0]
            g = palette[idx, 1]
            b = palette[idx, 2]
            a = 255 if land[y, x] else 0
            i = (y * w + x) * 4
            out_rgba.flat[i + 0] = r
            out_rgba.flat[i + 1] = g
            out_rgba.flat[i + 2] = b
            out_rgba.flat[i + 3] = a
