"""River network generation and computation."""

import numpy as np
import heapq
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass

from .utils import lerp

# --- numba import with a tiny safety fallback ---
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

# ---------- Numba helpers (CSR graph, topo, volume, watersheds) ----------

@njit(cache=True)
def _downstream_to_upstream_csr(downstream: np.ndarray):
    """
    Build CSR arrays (indptr, indices) for the upstream adjacency.
    downstream: int32 array of size n, -1 means no downstream.
    """
    n = downstream.shape[0]
    counts = np.zeros(n, np.int32)
    for i in range(n):
        j = downstream[i]
        if j != -1:
            counts[j] += 1

    indptr = np.empty(n + 1, np.int32)
    indptr[0] = 0
    for i in range(n):
        indptr[i + 1] = indptr[i] + counts[i]

    indices = np.empty(indptr[-1], np.int32)
    fill = indptr.copy()
    for i in range(n):
        j = downstream[i]
        if j != -1:
            k = fill[j]
            indices[k] = i
            fill[j] = k + 1

    return indptr, indices  # CSR: upstream neighbors of node i in indices[indptr[i]:indptr[i+1]]

@njit(cache=True)
def _topo_order_kahn(downstream: np.ndarray, indptr: np.ndarray, indices: np.ndarray):
    """
    Produce a topological order using in-degrees from upstream CSR.
    Raises on cycles.
    """
    n = downstream.shape[0]
    indeg = np.empty(n, np.int32)
    for i in range(n):
        indeg[i] = indptr[i + 1] - indptr[i]

    order = np.empty(n, np.int32)
    q = np.empty(n, np.int32)  # simple queue
    head = 0
    tail = 0

    for i in range(n):
        if indeg[i] == 0:
            q[tail] = i
            tail += 1

    k = 0
    while head < tail:
        i = q[head]
        head += 1
        order[k] = i
        k += 1

        d = downstream[i]
        if d != -1:
            indeg[d] -= 1
            if indeg[d] == 0:
                q[tail] = d
                tail += 1

    if k != n:
        raise ValueError("Cycle detected in river upstream graph.")
    return order

@njit(cache=True, fastmath=True)
def _compute_water_volume_kahn(downstream: np.ndarray,
                               indptr: np.ndarray,
                               indices: np.ndarray,
                               default_water_level: float,
                               evaporation_rate: float):
    """
    Compute volume in topological order (sources -> sinks).
    """
    n = downstream.shape[0]
    order = _topo_order_kahn(downstream, indptr, indices)
    volume = np.empty(n, np.float64)
    keep = 1.0 - evaporation_rate

    for t in range(n):
        i = order[t]
        v = default_water_level
        # sum upstream volumes
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            u = indices[p]
            v += volume[u]
        volume[i] = v * keep

    return volume, order

@njit(cache=True)
def _compute_watersheds_from_order(downstream: np.ndarray,
                                   land_mask_u8: np.ndarray,
                                   order: np.ndarray):
    """
    Assign watershed ids using reverse topological order.
    Ocean (land_mask==0) cells each get a unique id.
    Land points inherit id from downstream sink; new id if no downstream.
    """
    n = downstream.shape[0]
    watershed = np.empty(n, np.int32)
    for i in range(n):
        watershed[i] = -1

    next_id = 1

    # First give ocean cells stable unique ids
    for i in range(n):
        if land_mask_u8[i] == 0:
            watershed[i] = next_id
            next_id += 1

    # Now go from sinks upstream
    for t in range(n - 1, -1, -1):
        i = order[t]
        if land_mask_u8[i] == 0:
            # already assigned unique id
            continue

        d = downstream[i]
        if d == -1:
            if watershed[i] == -1:
                watershed[i] = next_id
                next_id += 1
        else:
            # inherit downstream basin (ocean or land)
            if watershed[i] == -1:
                watershed[i] = watershed[d] if watershed[d] != -1 else -1
            if watershed[i] == -1:
                # If downstream still -1 (shouldn't happen with correct order), make new
                watershed[i] = next_id
                next_id += 1

    return watershed


@dataclass
class RiverNetwork:
    """Container for river network data."""
    upstream: List[Set[int]]
    downstream: List[Optional[int]]
    volume: np.ndarray
    watershed: np.ndarray

class RiverGenerator:
    """Generates river networks on terrain."""
    
    def __init__(self, directional_inertia: float = 0.2,
                 default_water_level: float = 1.0,
                 evaporation_rate: float = 0.2):
        self.directional_inertia = directional_inertia
        self.default_water_level = default_water_level
        self.evaporation_rate = evaporation_rate
    
    def _build_upstream_connections_sets(self, downstream_list: List[Optional[int]]) -> List[Set[int]]:
        """(Python) Build upstream connections as List[Set[int]] for compatibility."""
        n = len(downstream_list)
        upstream_sets: List[Set[int]] = [set() for _ in range(n)]
        for i, j in enumerate(downstream_list):
            if j is not None:
                upstream_sets[j].add(i)
        return upstream_sets

    def compute_network(self, points: np.ndarray,
                        neighbors: List[List[int]],
                        heights: np.ndarray,
                        land_mask: np.ndarray) -> RiverNetwork:
        """Compute complete river network."""
        # 1) Flow directions (Python / heapq)
        downstream_list = self._compute_flow_directions(points, neighbors, heights, land_mask)

        # Convert downstream to a numba-friendly int32 array (-1 for None)
        downstream = np.array([-1 if j is None else j for j in downstream_list], dtype=np.int32)

        # 2) Upstream CSR (Numba)
        indptr, indices = _downstream_to_upstream_csr(downstream)

        # 3) Water volume (Numba, iterative + topo)
        volume, topo_order = _compute_water_volume_kahn(
            downstream, indptr, indices, float(self.default_water_level), float(self.evaporation_rate)
        )

        # 4) Watersheds (Numba, reverse topo)
        land_mask_u8 = land_mask.astype(np.uint8, copy=False)
        watershed = _compute_watersheds_from_order(downstream, land_mask_u8, topo_order)

        # 5) Upstream as sets for your public API
        upstream_sets = self._build_upstream_connections_sets(downstream_list)

        return RiverNetwork(upstream_sets, downstream_list, volume, watershed)
    
    def _compute_flow_directions(self, points: np.ndarray,
                                neighbors: List[List[int]], 
                                heights: np.ndarray,
                                land_mask: np.ndarray) -> List[Optional[int]]:
        """Compute downstream flow direction for each point."""
        num_points = len(points)
        
        def unit_delta(i, j):
            delta = points[j] - points[i]
            norm = np.linalg.norm(delta)
            return delta / norm if norm > 0 else delta
        
        # Initialize priority queue with coastal points
        q = []
        roots = set()
        
        for i in range(num_points):
            if land_mask[i]:
                continue
            
            is_root = True
            for j in neighbors[i]:
                if not land_mask[j]:
                    continue
                is_root = True
                heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
            
            if is_root:
                roots.add(i)
        
        # Compute flow directions
        downstream = [None] * num_points
        
        while len(q) > 0:
            (_, (i, j, direction)) = heapq.heappop(q)
            
            if downstream[j] is not None:
                continue
            
            downstream[j] = i
            
            # Process neighbors
            for k in neighbors[j]:
                if (heights[k] < heights[j] or 
                    downstream[k] is not None or 
                    not land_mask[k]):
                    continue
                
                neighbor_direction = unit_delta(j, k)
                priority = -np.dot(direction, neighbor_direction)
                
                weighted_direction = lerp(
                    neighbor_direction, direction,
                    self.directional_inertia
                )
                
                heapq.heappush(q, (priority, (j, k, weighted_direction)))
        
        return downstream
    