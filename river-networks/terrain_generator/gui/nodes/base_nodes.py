"""Base node classes for terrain generation."""

import numpy as np
import traceback
from typing import Optional, Dict, Any
from NodeGraphQt import BaseNode
from PyQt5.QtCore import pyqtSignal, QObject

from .context import get_global_context
from .custom_node_view import CustomNodeItem


class NodeSignals(QObject):
    """Signals for node execution."""
    execution_finished = pyqtSignal(object)  # Emits the node that finished
    progress_updated = pyqtSignal(object, float)  # Emits (node, progress 0-1)


class TerrainBaseNode(BaseNode):
    """Base class for all terrain generation nodes."""
    
    # Node identifier
    __identifier__ = 'terrain'
    
    def __init__(self):
        # Use custom node item with dynamic border support
        super().__init__(qgraphics_item=CustomNodeItem)
        self.signals = NodeSignals()
        self._cached_output = None
        self._is_dirty = True
        self.context = get_global_context()
        
        # Store the clean base name
        self._base_name = self.NODE_NAME if hasattr(self, 'NODE_NAME') else 'Node'
        
        # Style the node
        self.set_color(80, 80, 120)
    
    def emit_progress(self, progress):
        """Emit progress update (0.0 to 1.0)."""
        self.signals.progress_updated.emit(self, progress)
    
    def set_property(self, name: str, value: Any, **kwargs):
        """Override to mark node as dirty when properties change."""
        # Get old value
        try:
            old_value = self.get_property(name)
        except:
            old_value = None
        
        # Set the new value using parent's implementation with all kwargs
        super().set_property(name, value, **kwargs)
        
        # Mark dirty if value actually changed
        # IMPORTANT: Ignore UI/internal properties that don't affect computation
        ui_properties = {'name', 'selected', 'pos', 'disabled', 'visible', 'color'}
        if old_value != value and not name.startswith('_') and name not in ui_properties:
            self.mark_dirty()
    
    def mark_dirty(self):
        """Mark this node and downstream nodes as needing recomputation."""
        self._is_dirty = True
        self._cached_output = None
        
        # Mark all downstream nodes as dirty
        for output_port in self.output_ports():
            for connected_port in output_port.connected_ports():
                connected_node = connected_port.node()
                if isinstance(connected_node, TerrainBaseNode):
                    connected_node.mark_dirty()
    
    def execute(self) -> Optional[Any]:
        """
        Execute this node's computation.
        Override this in subclasses.
        Returns the output data.
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_output_data(self) -> Optional[Any]:
        """Get the cached output data."""
        return self._cached_output
    
    def set_output_data(self, data: Any):
        """Set the cached output data."""
        self._cached_output = data
        self._is_dirty = False


class MapPropertiesNode(TerrainBaseNode):
    """Node that defines global map properties (resolution, etc)."""
    
    # Node metadata
    NODE_NAME = 'Map Properties'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(60, 100, 60)
        
        # This node has no ports - it's purely for global settings
        
        # Add property for dimension
        self.add_combo_menu('dimension', 'Dimension', items=[
            '512', '1024', '2048', '4096'
        ])
        self.set_property('dimension', '1024')
        
        # Add a hidden property to mark this as a global node
        self.create_property('_is_global', True)
        
        # Register this node with the global context
        self.context.set_map_properties_node(self)
    
    def mark_dirty(self):
        """When map properties change, mark ALL nodes as dirty."""
        super().mark_dirty()
        
        # Get all nodes in the graph and mark them dirty
        try:
            graph = self.graph()
            if graph is not None:
                for node in graph.all_nodes():
                    if isinstance(node, TerrainBaseNode) and node != self:
                        node.mark_dirty()
        except Exception as e:
            # Node not yet added to graph, or graph not available
            pass
    
    def execute(self) -> Dict[str, int]:
        """Execute: update global context."""
        try:
            print(f"{self.name()}: Updating global context")
            dim_str = self.get_property('dimension')
            dim = int(dim_str)
            print(f"{self.name()}: Set global dimension to {dim}")
            
            # The context automatically queries this node, so we just mark as clean
            self._is_dirty = False
            self.signals.execution_finished.emit(self)
            return {'dim': dim}
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class FBMNode(TerrainBaseNode):
    """Node that generates FBM (Fractal Brownian Motion) noise."""
    
    # Node metadata
    NODE_NAME = 'FBM Noise'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(80, 120, 150)
        
        # No resolution input needed - uses global context
        
        # Add output port for heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add FBM parameters
        self.add_text_input('scale', 'Scale', text='-6.0')
        self.add_text_input('octaves', 'Octaves', text='6')
        self.add_text_input('persistence', 'Persistence', text='0.5')
        self.add_text_input('lacunarity', 'Lacunarity', text='2.0')
        self.add_text_input('lower', 'Lower Bound', text='2.0')
        self.add_text_input('upper', 'Upper Bound', text='inf')
        self.add_text_input('seed', 'Seed', text='42')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: generate FBM noise."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Parse FBM parameters
            scale = float(self.get_property('scale'))
            octaves = int(self.get_property('octaves'))
            persistence = float(self.get_property('persistence'))
            lacunarity = float(self.get_property('lacunarity'))
            
            lower_str = self.get_property('lower')
            lower = float('inf') if lower_str.lower() == 'inf' else float(lower_str)
            
            upper_str = self.get_property('upper')
            upper = float('inf') if upper_str.lower() == 'inf' else float(upper_str)
            
            seed = int(self.get_property('seed'))
            
            print(f"{self.name()}: Parameters - scale={scale}, octaves={octaves}, "
                  f"persistence={persistence}, lacunarity={lacunarity}, "
                  f"lower={lower}, upper={upper}, seed={seed}")
            
            # Import the FBM noise generator
            from terrain_generator.core import ConsistentFBMNoise
            
            # Create FBM noise generator
            fbm = ConsistentFBMNoise(
                scale=scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                lower=lower,
                upper=upper,
                seed_offset=0,
                base_seed=seed
            )
            
            # Generate noise
            print(f"{self.name()}: Generating FBM noise...")
            heightfield = fbm.generate((dim, dim))
            
            print(f"{self.name()}: Generated {dim}x{dim} heightfield, "
                  f"range=[{heightfield.min():.3f}, {heightfield.max():.3f}]")
            
            self.set_output_data(heightfield)
            self.signals.execution_finished.emit(self)
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class ConstantNode(TerrainBaseNode):
    """Node that creates a constant heightfield."""

    # Node metadata
    NODE_NAME = 'Constant'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(100, 80, 120)
        
        # No resolution input needed - uses global context
        
        # Add output port for heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add property for constant value
        self.add_text_input('value', 'Value', text='0.5')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: create constant heightfield."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Get constant value
            value_str = self.get_property('value')
            try:
                value = float(value_str)
            except ValueError:
                print(f"{self.name()}: Invalid value '{value_str}', using 0.5")
                value = 0.5
            
            # Create constant heightfield
            print(f"{self.name()}: Creating heightfield...")
            heightfield = np.full((dim, dim), value, dtype=np.float32)
            
            print(f"{self.name()}: Created {dim}x{dim} heightfield with value {value}")
            
            self.set_output_data(heightfield)
            self.signals.execution_finished.emit(self)
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class ShapeNode(TerrainBaseNode):
    """Node that generates geometric shapes for use as masks."""
    
    # Node metadata
    NODE_NAME = 'Shape'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(180, 120, 90)  # Orange/tan color for mask nodes
        
        # Add output port for heightfield/mask
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Shape type selector
        self.add_combo_menu('shape_type', 'Shape', items=[
            'Circle', 'Square', 'Triangle', 'Rounded Square'
        ])
        self.set_property('shape_type', 'Circle')
        
        # Position and scale parameters
        self.add_text_input('scale', 'Scale', text='1.0')
        self.add_text_input('offset_x', 'Offset X', text='0.0')
        self.add_text_input('offset_y', 'Offset Y', text='0.0')
        self.add_text_input('falloff', 'Falloff', text='0.1')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: generate shape mask."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Parse parameters
            shape_type = (self.get_property('shape_type') or 'Circle').strip().lower()
            scale = float(self.get_property('scale'))
            offset_x = float(self.get_property('offset_x'))
            offset_y = float(self.get_property('offset_y'))
            falloff = max(float(self.get_property('falloff')), 0.001)  # Avoid divide by zero
            
            print(f"{self.name()}: Parameters - shape={shape_type}, scale={scale}, "
                  f"offset=({offset_x}, {offset_y}), falloff={falloff}")
            
            # Generate shape
            print(f"{self.name()}: Generating shape...")
            heightfield = self._generate_shape(
                dim, shape_type, scale, offset_x, offset_y, falloff
            )
            
            print(f"{self.name()}: Generated {dim}x{dim} shape mask, "
                  f"range=[{heightfield.min():.3f}, {heightfield.max():.3f}]")
            
            self.set_output_data(heightfield)
            self.signals.execution_finished.emit(self)
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise
    
    def _generate_shape(self, dim: int, shape_type: str, scale: float, 
                       offset_x: float, offset_y: float, falloff: float) -> np.ndarray:
        """Generate the specified shape with falloff."""
        # Create coordinate grid centered at (0, 0)
        # Scale coordinates so that the map spans [-1, 1] in both dimensions
        y, x = np.meshgrid(
            np.linspace(-1, 1, dim),
            np.linspace(-1, 1, dim),
            indexing='ij'
        )
        
        # Apply offset (normalized coordinates, so offset of 1.0 moves by map width)
        x = x - offset_x
        y = y - offset_y
        
        # Generate distance field based on shape type
        if shape_type == 'circle':
            # Euclidean distance from center
            dist = np.sqrt(x**2 + y**2)
            # Radius in normalized coordinates (scale=1.0 means radius=1.0, touching edges)
            radius = scale
            
        elif shape_type == 'square':
            # Chebyshev distance (max of abs values)
            dist = np.maximum(np.abs(x), np.abs(y))
            radius = scale
            
        elif shape_type == 'triangle':
            # Equilateral triangle pointing up
            # Define three edges and take intersection
            # Edge 1: bottom edge at y = -scale
            # Edge 2: right diagonal
            # Edge 3: left diagonal
            
            # Height of equilateral triangle with side 2*scale
            height = scale * np.sqrt(3)
            
            # Distance from bottom edge
            d1 = y + scale
            
            # Distance from right edge (normal: (-sqrt(3)/2, -1/2), passes through (scale, -scale))
            # Distance = (normal . (point - point_on_line)) / |normal|
            d2 = (-np.sqrt(3) * (x - scale) - (y + scale)) / 2.0
            
            # Distance from left edge (normal: (sqrt(3)/2, -1/2), passes through (-scale, -scale))
            d3 = (np.sqrt(3) * (x + scale) - (y + scale)) / 2.0
            
            # Inside triangle when all distances are positive
            # Distance to triangle is negative of minimum distance when inside
            dist = -np.minimum(np.minimum(d1, d2), d3)
            radius = 0  # Triangle has exact edges
            
        elif shape_type == 'rounded square':
            # Square with rounded corners (superellipse)
            # Use smooth approximation: (|x|^p + |y|^p)^(1/p) with p=4
            p = 4.0
            dist = (np.abs(x)**p + np.abs(y)**p)**(1.0/p)
            radius = scale
            
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        # Convert distance field to mask with falloff
        # Inside shape (dist < radius): value = 1.0
        # At boundary (dist = radius): value = 0.5
        # Outside shape (dist > radius + falloff): value = 0.0
        
        # Smooth transition using smoothstep function
        # smoothstep(x) = 3x^2 - 2x^3 for x in [0, 1]
        edge_dist = (dist - radius) / falloff
        
        # Clamp to [0, 1] range
        edge_dist = np.clip(edge_dist, 0.0, 1.0)
        
        # Apply smoothstep for smooth falloff
        # We want 1.0 inside and 0.0 outside, so invert the distance
        mask = 1.0 - (3.0 * edge_dist**2 - 2.0 * edge_dist**3)
        
        return mask.astype(np.float32)


class CombineNode(TerrainBaseNode):
    """Node that combines two heightfields using blend operations."""

    NODE_NAME = 'Combine'

    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(140, 90, 130)

        # Inputs: two mandatory heightfields and optional mask
        self.add_input('heightfield_a', color=(150, 200, 150))
        self.add_input('heightfield_b', color=(150, 200, 150))
        self.add_input('mask', color=(200, 200, 200))

        # Output heightfield
        self.add_output('heightfield', color=(150, 200, 150))

        # Operation selector and related parameters
        self.add_combo_menu('operation', 'Operation', items=[
            'Fade', 'Add', 'Subtract', 'Multiply', 'Divide',
            'Smooth Max', 'Smooth Min', 'Pow'
        ])
        self.set_property('operation', 'Fade')

        self.add_text_input('fade_amount', 'Fade Amount', text='0.5')
        self.add_text_input('smoothness', 'Smoothness', text='5.0')
        self.add_text_input('divide_epsilon', 'Divide Epsilon', text='1e-5')

    def execute(self) -> Optional[np.ndarray]:
        """Execute: combine two heightfields."""
        try:
            print(f"{self.name()}: Starting execution")

            height_a = self._get_input_array('heightfield_a', required=True)
            height_b = self._get_input_array('heightfield_b', required=True)
            mask = self._get_input_array('mask', required=False)

            if height_a.shape != height_b.shape:
                raise ValueError("Input heightfields must have the same shape")

            height_a = height_a.astype(np.float32, copy=False)
            height_b = height_b.astype(np.float32, copy=False)

            mask_array = self._prepare_mask(mask, height_a.shape)

            operation = (self.get_property('operation') or 'Fade').strip().lower()
            combined = self._apply_operation(operation, height_a, height_b)

            # Blend result based on mask intensity
            result = height_a + mask_array * (combined - height_a)

            self.set_output_data(result)
            self.signals.execution_finished.emit(self)
            return result

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

    def _prepare_mask(self, mask: Optional[np.ndarray], target_shape) -> np.ndarray:
        """Prepare mask array ensuring correct shape and range."""
        if mask is None:
            return np.ones(target_shape, dtype=np.float32)

        if mask.shape != target_shape:
            raise ValueError("Mask shape must match heightfield shape")

        original_dtype = mask.dtype
        mask_array = mask.astype(np.float32, copy=False)

        if np.issubdtype(original_dtype, np.integer):
            mask_array = mask_array / 255.0

        mask_array = np.clip(mask_array, 0.0, 1.0)
        return mask_array

    def _apply_operation(self, operation: str, height_a: np.ndarray, height_b: np.ndarray) -> np.ndarray:
        """Apply combination operation between two heightfields."""
        if operation == 'fade':
            fade_amount = float(self.get_property('fade_amount'))
            fade_amount = float(np.clip(fade_amount, 0.0, 1.0))
            return (1.0 - fade_amount) * height_a + fade_amount * height_b

        if operation == 'add':
            return height_a + height_b

        if operation == 'subtract':
            return height_a - height_b

        if operation == 'multiply':
            return height_a * height_b

        if operation == 'divide':
            epsilon = abs(float(self.get_property('divide_epsilon')))
            epsilon = epsilon if epsilon > 0 else 1e-6
            safe_sign = np.where(height_b >= 0.0, 1.0, -1.0)
            safe_denominator = np.where(np.abs(height_b) < epsilon, safe_sign * epsilon, height_b)
            return height_a / safe_denominator

        if operation == 'smooth max':
            smoothness = max(abs(float(self.get_property('smoothness'))), 1e-6)
            return np.logaddexp(smoothness * height_a, smoothness * height_b) / smoothness

        if operation == 'smooth min':
            smoothness = max(abs(float(self.get_property('smoothness'))), 1e-6)
            return -np.logaddexp(-smoothness * height_a, -smoothness * height_b) / smoothness

        if operation == 'pow':
            safe_base = np.clip(height_a, 1e-6, None)
            return np.power(safe_base, height_b)

        raise ValueError(f"Unsupported operation '{operation}'")


class InvertNode(TerrainBaseNode):
    """Node that inverts a heightfield (1.0 - value)."""
    
    # Node metadata
    NODE_NAME = 'Invert'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(130, 100, 140)  # Purple tone
        
        # Add input port for heightfield
        self.add_input('heightfield', color=(150, 200, 150))
        
        # Add output port for inverted heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add inversion mode option
        self.add_combo_menu('mode', 'Mode', items=[
            'Normalized (1 - x)',
            'Range Flip (max - x + min)'
        ])
        self.set_property('mode', 'Normalized (1 - x)')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: invert heightfield."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get heightfield from connected node
            heightfield_port = self.inputs().get('heightfield')
            if heightfield_port is None:
                raise ValueError("Heightfield port not found")
            
            connected_ports = heightfield_port.connected_ports()
            if not connected_ports:
                raise ValueError("No heightfield input connected")
            
            # Get the connected node and execute if needed
            source_port = connected_ports[0]
            source_node = source_port.node()
            
            if isinstance(source_node, TerrainBaseNode):
                if source_node._is_dirty:
                    source_node.execute()
                heightfield = source_node.get_output_data()
            else:
                raise ValueError("Invalid heightfield source")
            
            if heightfield is None:
                raise ValueError("No heightfield data available")
            
            # Get mode
            mode = self.get_property('mode') or 'Normalized (1 - x)'
            
            print(f"{self.name()}: Inverting with mode: {mode}")
            
            # Invert based on mode
            if 'Normalized' in mode:
                # Simple inversion: 1 - x (assumes values in [0, 1])
                inverted = 1.0 - heightfield
            else:
                # Range flip: inverts within the actual data range
                # This preserves the range but flips high/low values
                h_min = heightfield.min()
                h_max = heightfield.max()
                if h_max > h_min:
                    inverted = h_max - heightfield + h_min
                else:
                    # If all values are the same, just return as-is
                    inverted = heightfield.copy()
            
            print(f"{self.name()}: Inverted heightfield, "
                  f"input range=[{heightfield.min():.3f}, {heightfield.max():.3f}], "
                  f"output range=[{inverted.min():.3f}, {inverted.max():.3f}]")
            
            self.set_output_data(inverted.astype(np.float32))
            self.signals.execution_finished.emit(self)
            return inverted
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise
        

class DomainWarpNode(TerrainBaseNode):
    """Node that applies domain warping to a heightfield."""
    
    # Node metadata
    NODE_NAME = 'Domain Warp'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(150, 100, 80)
        
        # Add input port for heightfield only
        self.add_input('heightfield', color=(150, 200, 150))
        
        # Add output port for warped heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add domain warp parameters
        self.add_text_input('offset_scale', 'Offset Scale', text='-5.0')
        self.add_text_input('offset_lower', 'Offset Lower', text='1.5')
        self.add_text_input('offset_upper', 'Offset Upper', text='inf')
        self.add_text_input('offset_amplitude', 'Warp Strength', text='150.0')
        self.add_text_input('seed', 'Seed', text='42')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: apply domain warp to heightfield."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get heightfield from connected node
            heightfield_port = self.inputs().get('heightfield')
            if heightfield_port is None:
                raise ValueError("Heightfield port not found")
            
            connected_ports = heightfield_port.connected_ports()
            if not connected_ports:
                raise ValueError("No heightfield input connected")
            
            # Get the connected node and execute if needed
            source_port = connected_ports[0]
            source_node = source_port.node()
            
            if isinstance(source_node, TerrainBaseNode):
                if source_node._is_dirty:
                    source_node.execute()
                heightfield = source_node.get_output_data()
            else:
                raise ValueError("Invalid heightfield source")
            
            if heightfield is None:
                raise ValueError("No heightfield data available")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Parse parameters
            offset_scale = float(self.get_property('offset_scale'))
            offset_lower_str = self.get_property('offset_lower')
            offset_lower = float('inf') if offset_lower_str.lower() == 'inf' else float(offset_lower_str)
            offset_upper_str = self.get_property('offset_upper')
            offset_upper = float('inf') if offset_upper_str.lower() == 'inf' else float(offset_upper_str)
            offset_amplitude = float(self.get_property('offset_amplitude'))
            seed = int(self.get_property('seed'))
            
            print(f"{self.name()}: Parameters - offset_scale={offset_scale}, "
                  f"offset_lower={offset_lower}, offset_upper={offset_upper}, "
                  f"offset_amplitude={offset_amplitude}, seed={seed}")
            
            # Import the FBM noise generator
            from terrain_generator.core import ConsistentFBMNoise
            
            # Generate offset noise fields
            print(f"{self.name()}: Generating offset noise fields...")
            
            # Use different seed offsets for X and Y to ensure they're different
            fbm_x = ConsistentFBMNoise(
                scale=offset_scale,
                octaves=6,
                persistence=0.5,
                lacunarity=2.0,
                lower=offset_lower,
                upper=offset_upper,
                seed_offset=1000,
                base_seed=seed
            )
            
            fbm_y = ConsistentFBMNoise(
                scale=offset_scale,
                octaves=6,
                persistence=0.5,
                lacunarity=2.0,
                lower=offset_lower,
                upper=offset_upper,
                seed_offset=2000,
                base_seed=seed
            )
            
            offset_x = fbm_x.generate((dim, dim))
            offset_y = fbm_y.generate((dim, dim))
            
            # Create complex offset field
            offsets = offset_amplitude * (offset_x + 1j * offset_y)
            
            # Apply domain warp using bilinear sampling
            print(f"{self.name()}: Applying domain warp...")
            warped_heightfield = self._sample(heightfield, offsets)
            
            print(f"{self.name()}: Domain warp complete, "
                  f"range=[{warped_heightfield.min():.3f}, {warped_heightfield.max():.3f}]")
            
            self.set_output_data(warped_heightfield)
            self.signals.execution_finished.emit(self)
            return warped_heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise
    
    @staticmethod
    def _sample(a: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Sample array with domain warping using bilinear interpolation."""
        shape = np.array(a.shape)
        delta = np.array((offset.real, offset.imag))
        
        # Create coordinate grid
        coords = np.array(np.meshgrid(*map(range, shape), indexing='ij')) - delta
        
        # Get lower and upper coordinates
        lower_coords = np.floor(coords).astype(int)
        upper_coords = lower_coords + 1
        coord_offsets = coords - lower_coords
        
        # Wrap coordinates (periodic boundary conditions)
        lower_coords[0] = lower_coords[0] % shape[0]
        lower_coords[1] = lower_coords[1] % shape[1]
        upper_coords[0] = upper_coords[0] % shape[0]
        upper_coords[1] = upper_coords[1] % shape[1]
        
        # Bilinear interpolation
        def lerp(a, b, t):
            return a * (1 - t) + b * t
        
        return lerp(
            lerp(a[lower_coords[0], lower_coords[1]],
                 a[lower_coords[0], upper_coords[1]],
                 coord_offsets[1]),
            lerp(a[upper_coords[0], lower_coords[1]],
                 a[upper_coords[0], upper_coords[1]],
                 coord_offsets[1]),
            coord_offsets[0]
        )
