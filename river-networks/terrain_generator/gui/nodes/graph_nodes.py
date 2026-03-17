"""Graph-based terrain nodes that work with triangulated point data."""

import numpy as np
import scipy.spatial
import traceback
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.tri as mtri

from .base_nodes import TerrainBaseNode
from terrain_generator.core.utils import (
    normalize,
    gaussian_gradient,
    poisson_disc_sampling,
    render_triangulation,
    connect_inland_seas,
)

try:
    from numba import njit, prange
    _NUMBA = True
except Exception:
    _NUMBA = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    def prange(*args):
        return range(*args)


@dataclass
class TerrainGraph:
    """
    Container for graph-based terrain representation.
    
    This represents terrain as a set of sampled points with triangulation,
    rather than a regular heightfield grid. Used for advanced terrain
    generation operations like river carving and geological simulation.
    """
    points: np.ndarray  # Nx2 array of (row, col) point coordinates
    points_height: np.ndarray  # N-length array of heights at each point
    neighbors: List[np.ndarray]  # List of neighbor indices for each point
    edge_weights: List[np.ndarray]  # List of edge distances for each point
    triangulation: Any  # scipy.spatial.Delaunay or matplotlib.tri.Triangulation
    
    # Optional metadata sampled from original heightfield
    points_land: Optional[np.ndarray] = None  # Boolean land mask at points
    points_deltas: Optional[np.ndarray] = None  # Gradient deltas at points
    
    # Reference to original dimension for rasterization
    dimension: int = 256
    
    def rasterize(self, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert graph representation back to a heightfield grid.
        
        Args:
            target_shape: Optional (height, width) for output. 
                         Defaults to (dimension, dimension).
        
        Returns:
            Heightfield as 2D numpy array
        """
        if target_shape is None:
            target_shape = (self.dimension, self.dimension)
        
        print(f"TerrainGraph.rasterize: target_shape={target_shape}, num_points={len(self.points)}")
        
        # Ensure we have a matplotlib Triangulation
        if isinstance(self.triangulation, mtri.Triangulation):
            print("TerrainGraph.rasterize: Using existing matplotlib Triangulation")
            tri = self.triangulation
        elif hasattr(self.triangulation, 'points') and hasattr(self.triangulation, 'simplices'):
            # scipy.spatial.Delaunay object
            print("TerrainGraph.rasterize: Converting scipy Delaunay to matplotlib Triangulation")
            tri = mtri.Triangulation(
                self.triangulation.points[:, 0],
                self.triangulation.points[:, 1],
                self.triangulation.simplices
            )
        else:
            raise ValueError(f"Invalid triangulation object: {type(self.triangulation)}")
        
        print(f"TerrainGraph.rasterize: Triangulation has {len(tri.x)} points, {len(tri.triangles)} triangles")
        print(f"TerrainGraph.rasterize: points_height shape: {self.points_height.shape}, range: [{self.points_height.min():.3f}, {self.points_height.max():.3f}]")
        
        # Render using linear interpolation
        try:
            heightfield = render_triangulation(
                target_shape, 
                tri, 
                self.points_height, 
                triangulation=tri,
                fill_value=0.0
            )
            print(f"TerrainGraph.rasterize: Successfully rasterized to {heightfield.shape}")
            return heightfield
        except Exception as e:
            print(f"TerrainGraph.rasterize: ERROR during render_triangulation: {e}")
            import traceback
            traceback.print_exc()
            raise


@njit(parallel=True, fastmath=True)
def _compute_edge_weights_from_csr_numba(points, indptr, indices, distance_normalizer):
    """Numba-accelerated edge weight computation."""
    n_edges = indices.size
    out = np.empty(n_edges, dtype=np.float64)
    for src in prange(indptr.size - 1):
        start = indptr[src]
        end = indptr[src + 1]
        px0 = points[src, 0]
        py0 = points[src, 1]
        for e in range(start, end):
            dst = indices[e]
            dx = points[dst, 0] - px0
            dy = points[dst, 1] - py0
            dist = (np.sqrt(dx*dx + dy*dy)) * distance_normalizer
            out[e] = dist
    return out


class BuildTerrainNode(TerrainBaseNode):
    """
    Build Terrain node: Convert heightfield to graph-based terrain representation.
    
    This node performs the core terrain graph construction:
    1. Samples points using Poisson disc sampling
    2. Creates Delaunay triangulation
    3. Computes graph connectivity and edge weights
    4. Samples heightfield values at points
    5. Computes initial height using Dijkstra's algorithm
    
    The output is a TerrainGraph structure suitable for advanced operations
    like river carving, erosion simulation, and geological layering.
    """
    
    NODE_NAME = 'Build Terrain'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(120, 80, 150)  # Purple for graph operations
        
        # Input ports
        self.add_input('heightfield', color=(150, 200, 150))
        self.add_input('land_mask', color=(120, 180, 120))
        
        # Output port for terrain graph
        self.add_output('terrain_graph', color=(180, 120, 200))
        
        # Parameters
        self.add_text_input('disc_radius', 'Point Spacing', text='1.0')
        self.add_text_input('max_delta', 'Max Steepness', text='0.05')
        self.add_text_input('seed', 'Seed', text='42')
    
    def execute(self) -> Optional[TerrainGraph]:
        """Execute: build terrain graph from heightfield."""
        try:
            print(f"{self.name()}: Starting terrain graph construction")
            
            # Get input heightfield
            heightfield = self._get_input_array('heightfield', required=True)

            # Get land mask (required)
            land_mask = self._get_input_array('land_mask', required=True)

            if land_mask.shape != heightfield.shape:
                raise ValueError(
                    f"land_mask shape {land_mask.shape} does not match heightfield {heightfield.shape}"
                )

            # Ensure boolean mask
            if land_mask.dtype != np.bool_:
                land_mask = land_mask.astype(bool, copy=False)

            # Get dimension
            dim = self.context.get_resolution()
            target_shape = (dim, dim)

            if heightfield.shape != target_shape:
                raise ValueError(
                    f"heightfield shape {heightfield.shape} does not match target {target_shape}"
                )

            # Match generator behaviour: keep oceans at zero height
            heightfield = np.where(land_mask, heightfield, 0.0)

            # Parse parameters
            disc_radius = float(self.get_property('disc_radius'))
            max_delta = float(self.get_property('max_delta'))
            seed = int(self.get_property('seed'))
            
            print(f"{self.name()}: Using dimension={dim}, disc_radius={disc_radius}, "
                  f"max_delta={max_delta}, seed={seed}")
            
            # Step 1: Compute deltas (gradient magnitude)
            self.emit_progress(0.1)
            print(f"{self.name()}: Computing gradients...")
            deltas = normalize(np.abs(gaussian_gradient(heightfield)))
            
            # Step 2: Sample points using Poisson disc sampling
            self.emit_progress(0.2)
            print(f"{self.name()}: Sampling points with disc_radius={disc_radius}...")
            np.random.seed(seed)
            points = poisson_disc_sampling(target_shape, disc_radius, seed=seed)
            num_points = len(points)
            print(f"{self.name()}: Sampled {num_points} points")
            
            # Step 3: Create Delaunay triangulation
            self.emit_progress(0.3)
            print(f"{self.name()}: Creating Delaunay triangulation...")
            tri = scipy.spatial.Delaunay(points)
            
            # Step 4: Extract neighbor connectivity
            self.emit_progress(0.4)
            print(f"{self.name()}: Building neighbor graph...")
            indptr, indices = tri.vertex_neighbor_vertices
            neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(len(points))]
            
            # Step 5: Compute edge weights
            self.emit_progress(0.5)
            print(f"{self.name()}: Computing edge weights...")
            dim_scale = dim / 256.0
            distance_normalizer = 1.0 / dim_scale
            
            if _NUMBA:
                weights_flat = _compute_edge_weights_from_csr_numba(
                    points.astype(np.float64), indptr, indices, distance_normalizer
                )
            else:
                # Fallback without numba
                src = np.repeat(np.arange(len(points)), np.diff(indptr))
                dst = indices
                diffs = points[dst] - points[src]
                weights_flat = np.linalg.norm(diffs, axis=1) * distance_normalizer
            
            edge_weights = [weights_flat[indptr[k]:indptr[k + 1]].copy() 
                           for k in range(len(points))]
            
            # Step 6: Sample values at points
            self.emit_progress(0.6)
            print(f"{self.name()}: Sampling heightfield at points...")
            coords = self._points_to_indices(points, target_shape)
            points_deltas = deltas[coords[:, 0], coords[:, 1]]
            points_land = land_mask[coords[:, 0], coords[:, 1]]
            
            # Step 7: Compute initial height at points using Dijkstra
            self.emit_progress(0.7)
            print(f"{self.name()}: Computing initial heights...")
            points_height = self._compute_height(
                points, neighbors, edge_weights, points_deltas, max_delta, dim_scale
            )
            
            print(f"{self.name()}: Height range: [{points_height.min():.3f}, "
                  f"{points_height.max():.3f}]")
            
            # Step 8: Create matplotlib triangulation for rendering
            self.emit_progress(0.9)
            mtri_obj = mtri.Triangulation(
                tri.points[:, 0], 
                tri.points[:, 1], 
                tri.simplices
            )
            
            # Create TerrainGraph output
            terrain_graph = TerrainGraph(
                points=points,
                points_height=points_height,
                neighbors=neighbors,
                edge_weights=edge_weights,
                triangulation=mtri_obj,
                points_land=points_land,
                points_deltas=points_deltas,
                dimension=dim
            )
            
            self.set_output_data(terrain_graph)
            self.signals.execution_finished.emit(self)
            self.emit_progress(1.0)
            
            print(f"{self.name()}: Successfully created terrain graph with {num_points} points")
            return terrain_graph
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise
    
    def _get_input_array(self, port_name: str, required: bool) -> Optional[np.ndarray]:
        """Fetch input data from a port, executing upstream nodes if needed."""
        port = self.inputs().get(port_name)
        if port is None:
            raise ValueError(f"Port '{port_name}' not found")
        
        connected_ports = port.connected_ports()
        if not connected_ports:
            if required:
                raise ValueError(f"Input '{port_name}' is not connected")
            return None
        
        source_port = connected_ports[0]
        source_node = source_port.node()
        
        if isinstance(source_node, TerrainBaseNode):
            if source_node._is_dirty:
                source_node.execute()
            data = source_node.get_output_data()
        else:
            raise ValueError(f"Connected node for '{port_name}' is not a terrain node")
        
        if data is None:
            raise ValueError(f"No data received from '{port_name}'")
        
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Input '{port_name}' must be a numpy array")
        
        return data
    
    def _points_to_indices(self, points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Convert float sample coordinates to safe integer grid indices."""
        h, w = shape
        coords = np.floor(points).astype(np.int64)
        np.clip(coords[:, 0], 0, h - 1, out=coords[:, 0])  # row / y
        np.clip(coords[:, 1], 0, w - 1, out=coords[:, 1])  # col / x
        return coords
    
    def _prepare_graph(self, neighbors: List[np.ndarray],
                       edge_weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
        """Flatten neighbor/weight lists into CSR arrays for graph traversal."""
        dim = len(neighbors)
        lengths = np.fromiter((len(n) for n in neighbors), dtype=np.int64, count=dim)
        indptr = np.empty(dim + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        total_edges = int(indptr[-1])
        
        if total_edges == 0:
            indices = np.empty(0, dtype=np.int64)
            weights = np.empty(0, dtype=np.float64)
            row_indices = np.empty(0, dtype=np.int64)
        else:
            indices = np.concatenate(neighbors).astype(np.int64, copy=False)
            weights = np.concatenate(edge_weights).astype(np.float64, copy=False)
            row_indices = np.repeat(np.arange(dim, dtype=np.int64), lengths)
        
        return indptr, indices, row_indices, weights
    
    def _compute_height(self, points: np.ndarray, neighbors: List[np.ndarray],
                       edge_weights: List[np.ndarray], deltas: np.ndarray,
                       max_delta: float, dim_scale: float) -> np.ndarray:
        """Compute heights for each point using Dijkstra's algorithm."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        
        indptr, indices, row_indices, weights = self._prepare_graph(neighbors, edge_weights)
        dim = len(points)
        seed_idx = int(np.argmin(points.sum(axis=1)))
        
        if indices.size == 0:
            return np.zeros(dim, dtype=np.float64)
        
        # Compute edge costs: delta * max_delta * distance
        edge_costs = deltas[indices] * max_delta * weights
        
        # Build CSR graph
        graph = csr_matrix((edge_costs, indices, indptr), shape=(dim, dim))
        
        # Run Dijkstra from seed point
        distances = dijkstra(graph, indices=seed_idx, directed=True,
                           return_predecessors=False)
        distances[np.isinf(distances)] = 0.0
        
        # Scale heights by dimension ratio
        result = distances * dim_scale
        
        # Ensure minimum is 0
        result = result - result.min()
        
        return result
    
    def get_output_for_visualization(self) -> Optional[np.ndarray]:
        """
        Override to provide rasterized heightfield for visualization.
        
        This is called when the node is pinned, converting the graph
        representation back to a heightfield for display.
        """
        try:
            print(f"{self.name()}: get_output_for_visualization called")
            print(f"{self.name()}: _cached_output type: {type(self._cached_output)}")
            
            if self._cached_output is None:
                print(f"{self.name()}: No cached output")
                return None
            
            if not isinstance(self._cached_output, TerrainGraph):
                print(f"{self.name()}: Cached output is not a TerrainGraph")
                return None
            
            print(f"{self.name()}: Rasterizing terrain graph for visualization...")
            
            # Rasterize the graph to a heightfield
            heightfield = self._cached_output.rasterize()
            
            print(f"{self.name()}: Rasterized to {heightfield.shape}, "
                  f"range=[{heightfield.min():.3f}, {heightfield.max():.3f}]")
            
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR in get_output_for_visualization: {e}")
            import traceback
            traceback.print_exc()
            return None


class GenerateLandMaskNode(TerrainBaseNode):
    """Node that derives a land mask from an input heightfield."""

    NODE_NAME = 'Generate Land Mask'

    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(90, 140, 90)

        self.add_input('heightfield', color=(150, 200, 150))
        self.add_output('land_mask', color=(120, 180, 120))

        self.add_text_input('sea_level', 'Sea Level', text='0.0')

    def execute(self) -> Optional[np.ndarray]:
        """Generate a boolean land mask from the supplied heightfield."""
        try:
            print(f"{self.name()}: Generating land mask")

            heightfield = self._get_input_array('heightfield', required=True)
            sea_level = float(self.get_property('sea_level'))

            dim = self.context.get_resolution()
            target_shape = (dim, dim)

            if heightfield.shape != target_shape:
                raise ValueError(
                    f"heightfield shape {heightfield.shape} does not match target {target_shape}"
                )

            self.emit_progress(0.2)

            flooded = np.where(heightfield > sea_level, heightfield - sea_level, 0.0)
            land_mask = flooded > 0.001

            self.emit_progress(0.5)

            flooded, land_mask = connect_inland_seas(
                flooded,
                land_mask,
                min_sea_size=30,
            )

            flooded = flooded * land_mask
            land_mask = (flooded > 0.001)

            self.emit_progress(0.9)

            land_mask = np.ascontiguousarray(land_mask, dtype=bool)
            self.set_output_data(land_mask)
            self.signals.execution_finished.emit(self)
            self.emit_progress(1.0)

            print(f"{self.name()}: Land mask generated (land pixels={land_mask.sum()})")

            return land_mask

        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise

    def _get_input_array(self, port_name: str, required: bool) -> Optional[np.ndarray]:
        """Fetch input data from a port, executing upstream nodes if needed."""
        port = self.inputs().get(port_name)
        if port is None:
            raise ValueError(f"Port '{port_name}' not found")

        connected_ports = port.connected_ports()
        if not connected_ports:
            if required:
                raise ValueError(f"Input '{port_name}' is not connected")
            return None

        source_port = connected_ports[0]
        source_node = source_port.node()

        if isinstance(source_node, TerrainBaseNode):
            if source_node._is_dirty:
                source_node.execute()
            data = source_node.get_output_data()
        else:
            raise ValueError(f"Connected node for '{port_name}' is not a terrain node")

        if data is None:
            raise ValueError(f"No data received from '{port_name}'")

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Input '{port_name}' must be a numpy array")

        return data
