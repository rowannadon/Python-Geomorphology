"""Node editor widget with pinning, auto-update, and cache state indicators."""

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QMessageBox, QSplitter,
                            QUndoStack,
                            QCheckBox)
from PyQt5.QtCore import pyqtSignal, QEvent, Qt, QTimer
from PyQt5.QtGui import QContextMenuEvent
from NodeGraphQt import NodeGraph
from NodeGraphQt.constants import PipeLayoutEnum
from NodeGraphQt.widgets.viewer import NodeViewer
import numpy as np
import time

from ..visualization import TerrainViewport
from .nodes import (
    MapPropertiesNode,
    ConstantNode,
    FBMNode,
    CombineNode,
    DomainWarpNode,
    ShapeNode,
    InvertNode,
    GenerateLandMaskNode,
    BuildTerrainNode,
    TerrainGraph,
)
from .nodes.execution_widgets import NodeProgressBar, NodeExecutionLabel


class MacFriendlyNodeViewer(NodeViewer):
    """Node viewer tweaks for right-drag panning and macOS delete keys."""

    def __init__(self, parent=None, undo_stack=None, delete_callback=None):
        super().__init__(parent=parent, undo_stack=undo_stack)
        self._delete_callback = delete_callback
        self._right_mouse_press_pos = None
        self._right_mouse_dragged = False
        self._suppress_context_menu = False

    def contextMenuEvent(self, event):
        if self._suppress_context_menu:
            self._suppress_context_menu = False
            event.accept()
            return
        super().contextMenuEvent(event)

    def _clear_context_menu_suppression(self):
        self._suppress_context_menu = False

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if callable(self._delete_callback):
                self._delete_callback()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._right_mouse_press_pos = event.pos()
            self._right_mouse_dragged = False
            self._suppress_context_menu = False
            self.RMB_state = True
            self._origin_pos = event.pos()
            self._previous_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.RightButton) and not self.ALT_state:
            if self._right_mouse_press_pos is None:
                self._right_mouse_press_pos = event.pos()
                self._previous_pos = event.pos()

            drag_distance = (event.pos() - self._right_mouse_press_pos).manhattanLength()
            if drag_distance < QApplication.startDragDistance() and not self._right_mouse_dragged:
                event.accept()
                return

            self.RMB_state = True
            self._right_mouse_dragged = True
            self.viewport().setCursor(Qt.ClosedHandCursor)

            previous_pos = self.mapToScene(self._previous_pos)
            current_pos = self.mapToScene(event.pos())
            delta = previous_pos - current_pos
            self._set_viewer_pan(delta.x(), delta.y())
            self._previous_pos = event.pos()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.viewport().unsetCursor()
            press_pos = self._right_mouse_press_pos
            was_dragged = self._right_mouse_dragged
            self._right_mouse_press_pos = None
            self._right_mouse_dragged = False
            self.RMB_state = False
            self._previous_pos = event.pos()

            if was_dragged:
                self._suppress_context_menu = True
                QTimer.singleShot(150, self._clear_context_menu_suppression)
                self._node_positions = {}
                event.accept()
                return

            if press_pos is not None:
                context_event = QContextMenuEvent(
                    QContextMenuEvent.Mouse,
                    event.pos(),
                    event.globalPos(),
                    event.modifiers(),
                )
                self.contextMenuEvent(context_event)
                event.accept()
                return

        super().mouseReleaseEvent(event)


class NodeEditorWidget(QWidget):
    """Widget containing the node graph editor with integrated visualization."""
    
    node_visualized = pyqtSignal(object)
    
    # Border colors for different node states (R, G, B)
    BORDER_COLORS = {
        'clean': (80, 80, 80),      # Default gray - never executed
        'cached': (0, 200, 0),       # Green - has cached result
        'dirty': (255, 200, 0),      # Yellow/amber - needs recompute
        'executing': (100, 150, 255), # Light blue - currently executing
        'error': (255, 50, 50),      # Red - error state
        'pinned': (0, 150, 255),     # Blue - pinned for display (takes precedence)
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.node_graph = None
        self.node_viewport = None
        self.main_terrain_viewport = None
        self.map_properties_node = None
        self.main_window = None
        
        # Pin and auto-update system
        self.pinned_node = None
        self.auto_update_enabled = True
        self.is_generating = False
        self.pending_update = False
        
        # Progress tracking
        self.active_progress_bars = {}  # node -> progress bar widget
        self.active_execution_labels = {}  # node -> execution label widget
        
        # Debounce timer for parameter changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._execute_pinned_node)
        self.update_cooldown_ms = 500
        
        self.setup_ui()
        self.setup_node_graph()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add node buttons
        self.add_fbm_btn = QPushButton("Add FBM")
        self.add_fbm_btn.clicked.connect(self.add_fbm_node)
        toolbar_layout.addWidget(self.add_fbm_btn)
        
        self.add_constant_btn = QPushButton("Add Constant")
        self.add_constant_btn.clicked.connect(self.add_constant_node)
        toolbar_layout.addWidget(self.add_constant_btn)

        self.add_combine_btn = QPushButton("Add Combine")
        self.add_combine_btn.clicked.connect(self.add_combine_node)
        toolbar_layout.addWidget(self.add_combine_btn)

        self.add_domain_warp_btn = QPushButton("Add Domain Warp")
        self.add_domain_warp_btn.clicked.connect(self.add_domain_warp_node)
        toolbar_layout.addWidget(self.add_domain_warp_btn)

        self.add_shape_btn = QPushButton("Add Shape")
        self.add_shape_btn.clicked.connect(self.add_shape_node)
        toolbar_layout.addWidget(self.add_shape_btn)

        self.add_invert_btn = QPushButton("Add Invert")
        self.add_invert_btn.clicked.connect(self.add_invert_node)
        toolbar_layout.addWidget(self.add_invert_btn)

        self.add_land_mask_btn = QPushButton("Add Land Mask")
        self.add_land_mask_btn.clicked.connect(self.add_land_mask_node)
        toolbar_layout.addWidget(self.add_land_mask_btn)

        self.add_build_terrain_btn = QPushButton("Add Build Terrain")
        self.add_build_terrain_btn.clicked.connect(self.add_build_terrain_node)
        toolbar_layout.addWidget(self.add_build_terrain_btn)

        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self._delete_selected_nodes)
        toolbar_layout.addWidget(self.remove_selected_btn)
        
        toolbar_layout.addStretch()
        
        # Auto-update checkbox
        self.auto_update_checkbox = QCheckBox("Auto-Update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_toggled)
        toolbar_layout.addWidget(self.auto_update_checkbox)
        
        # Unpin button
        self.unpin_btn = QPushButton("Unpin Display")
        self.unpin_btn.clicked.connect(self._unpin_node)
        self.unpin_btn.setEnabled(False)
        toolbar_layout.addWidget(self.unpin_btn)
        
        # Clear cache button
        self.clear_btn = QPushButton("Clear All Caches")
        self.clear_btn.clicked.connect(self._clear_all_caches)
        toolbar_layout.addWidget(self.clear_btn)
        
        # Clear graph button
        self.clear_graph_btn = QPushButton("Clear Graph")
        self.clear_graph_btn.clicked.connect(self.clear_graph)
        toolbar_layout.addWidget(self.clear_graph_btn)
        
        layout.addWidget(toolbar)
        
        # Status bar showing current pin state
        self.status_bar = QLabel("Display: None - Double-click a node to pin")
        self.status_bar.setStyleSheet("background: #2a2a2a; padding: 4px; color: #aaa;")
        layout.addWidget(self.status_bar)
        
        # Create horizontal splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Node graph
        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.graph_container)
        
        # Right side: Terrain viewport
        self.node_viewport = TerrainViewport()
        self.splitter.addWidget(self.node_viewport)
        
        # Set initial sizes (50/50 split)
        self.splitter.setSizes([500, 500])
        
        layout.addWidget(self.splitter, stretch=1)
    
    def setup_node_graph(self):
        """Initialize the NodeGraphQt graph."""
        undo_stack = QUndoStack(self)
        viewer = MacFriendlyNodeViewer(
            undo_stack=undo_stack,
            delete_callback=self._delete_selected_nodes,
        )
        self.node_graph = NodeGraph(viewer=viewer, undo_stack=undo_stack)
        
        graph_widget = self.node_graph.widget
        graph_widget.installEventFilter(self)
        self.node_graph.viewer().installEventFilter(self)
        self.node_graph.viewer().viewport().installEventFilter(self)
        self.graph_layout.addWidget(graph_widget)
        
        self.node_graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        self.node_graph.set_zoom(1.0)
        
        # Connect signals
        self.node_graph.node_double_clicked.connect(self._on_node_double_clicked)
        self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        
        # Register and create nodes
        self.register_nodes()
        self.create_map_properties_node()
    
    def register_nodes(self):
        """Register custom node types."""
        self.node_graph.register_node(MapPropertiesNode)
        self.node_graph.register_node(ConstantNode)
        self.node_graph.register_node(FBMNode)
        self.node_graph.register_node(CombineNode)
        self.node_graph.register_node(DomainWarpNode)
        self.node_graph.register_node(ShapeNode)
        self.node_graph.register_node(InvertNode)
        self.node_graph.register_node(GenerateLandMaskNode)
        self.node_graph.register_node(BuildTerrainNode)
    
    def create_map_properties_node(self):
        """Create the global Map Properties node."""
        self.map_properties_node = self.node_graph.create_node(
            'terrain.MapPropertiesNode',
            name='Map Properties',
            pos=[-300, 0]
        )
        self._setup_node_execution(self.map_properties_node)
        self._update_node_visual_state(self.map_properties_node)
    
    def set_main_terrain_viewport(self, viewport):
        """Set reference to the main terrain viewport for syncing."""
        self.main_terrain_viewport = viewport
    
    def set_main_window(self, main_window):
        """Set the main window reference (kept for compatibility)."""
        self.main_window = main_window
    
    def add_fbm_node(self):
        """Add an FBM node to the graph."""
        node = self.node_graph.create_node(
            'terrain.FBMNode',
            name='FBM Noise',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)
    
    def add_constant_node(self):
        """Add a Constant node to the graph."""
        node = self.node_graph.create_node(
            'terrain.ConstantNode',
            name='Constant',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_combine_node(self):
        """Add a Combine node to the graph."""
        node = self.node_graph.create_node(
            'terrain.CombineNode',
            name='Combine',
            pos=[100, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_domain_warp_node(self):
        """Add a Domain Warp node to the graph."""
        node = self.node_graph.create_node(
            'terrain.DomainWarpNode',
            name='Domain Warp',
            pos=[200, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_shape_node(self):
        """Add a Shape node to the graph."""
        node = self.node_graph.create_node(
            'terrain.ShapeNode',
            name='Shape',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_invert_node(self):
        """Add an Invert node to the graph."""
        node = self.node_graph.create_node(
            'terrain.InvertNode',
            name='Invert',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_land_mask_node(self):
        """Add a Generate Land Mask node to the graph."""
        node = self.node_graph.create_node(
            'terrain.GenerateLandMaskNode',
            name='Generate Land Mask',
            pos=[250, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)

    def add_build_terrain_node(self):
        """Add a Build Terrain node to the graph."""
        node = self.node_graph.create_node(
            'terrain.BuildTerrainNode',
            name='Build Terrain',
            pos=[300, 0]
        )
        self._setup_node_execution(node)
        self._update_node_visual_state(node)
    
    def _setup_node_execution(self, node):
        """Setup execution for a node."""
        # Connect signals
        if hasattr(node, 'signals'):
            node.signals.execution_finished.connect(self._on_node_executed)
            node.signals.progress_updated.connect(self._on_node_progress_updated)
        
        # Store the original name without any indicators
        if not hasattr(node, '_base_name'):
            node._base_name = node.name()
        
        # Monitor property changes for auto-update
        if hasattr(node, 'set_property'):
            original_set_property = node.set_property
            
            def wrapped_set_property(name, value, **kwargs):
                result = original_set_property(name, value, **kwargs)
                if not name.startswith('_') and name not in ('name', 'selected', 'pos'):
                    self._on_node_property_changed(node)
                return result
            
            node.set_property = wrapped_set_property
    
    def _on_auto_update_toggled(self, state):
        """Handle auto-update checkbox toggle."""
        self.auto_update_enabled = (state == Qt.Checked)
        status = "enabled" if self.auto_update_enabled else "disabled"
        print(f"Auto-update {status}")
        self._update_status_bar()
    
    def _on_node_double_clicked(self, node):
        """Pin node on double-click and execute."""
        # Set this node as the pinned display node
        self._pin_node(node)
        
        # Execute immediately (bypass auto-update checks for initial pin)
        try:
            self.is_generating = True
            self._execute_node_with_deps(node)
            
            # Get output for visualization
            # For graph nodes, this might need rasterization
            output_data = None
            if hasattr(node, 'get_output_for_visualization'):
                # Node provides custom visualization (e.g., graph -> heightfield)
                print(f"Node has get_output_for_visualization, calling it...")
                output_data = node.get_output_for_visualization()
            else:
                # Standard output
                output_data = node.get_output_data()
            
            # Visualize result
            if output_data is not None and isinstance(output_data, np.ndarray):
                self._visualize_heightfield(output_data, node._base_name)
                QMessageBox.information(
                    self,
                    "Node Pinned",
                    f"Node '{node._base_name}' pinned and visualized!\n"
                    f"Output shape: {output_data.shape}\n"
                    f"Range: [{output_data.min():.3f}, {output_data.max():.3f}]"
                )
            else:
                # Not a heightfield node or visualization failed
                output_type = type(node.get_output_data()).__name__ if node.get_output_data() else "None"
                QMessageBox.information(
                    self,
                    "Node Pinned",
                    f"Node '{node._base_name}' pinned!\n"
                    f"Output type: {output_type}\n"
                    f"(No heightfield output to visualize)"
                )
                
        except Exception as e:
            self._update_node_visual_state(node, 'error')
            QMessageBox.critical(
                self,
                "Execution Error",
                f"Error executing node '{node._base_name}':\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        finally:
            self.is_generating = False
    
    def _pin_node(self, node):
        """Pin a node for display."""
        # Unpin previous node
        if self.pinned_node is not None:
            self._update_node_visual_state(self.pinned_node)
        
        # Pin new node
        self.pinned_node = node
        self._update_node_visual_state(node)
        
        # Update UI
        self.unpin_btn.setEnabled(True)
        self._update_status_bar()
        
        print(f"Pinned node: {node._base_name}")

    def _unpin_node(self):
        """Unpin the current node."""
        if self.pinned_node is not None:
            self._update_node_visual_state(self.pinned_node)
            print(f"Unpinned node: {self.pinned_node._base_name}")
            self.pinned_node = None
        
        self.unpin_btn.setEnabled(False)
        self._update_status_bar()

    def _update_node_visual_state(self, node, cache_state=None):
        """
        Update the visual state of a node (border color and width).
        Pinned state takes precedence over cache state.
        
        Args:
            node: The node to update
            cache_state: Optional explicit cache state ('cached', 'dirty', 'error', etc.)
                        If None, infer from node properties
        """
        # Check if this is the pinned node - pinned state has highest priority
        is_pinned = (node == self.pinned_node)
        
        if is_pinned:
            color = self.BORDER_COLORS['pinned']
            border_width = 4.0  # Extra thick for pinned
        else:
            # Determine cache state if not provided
            if cache_state is None:
                if hasattr(node, '_cached_output'):
                    if node._cached_output is not None and not node._is_dirty:
                        cache_state = 'cached'
                    elif node._is_dirty:
                        cache_state = 'dirty'
                    else:
                        cache_state = 'clean'
                else:
                    cache_state = 'clean'
            
            # Get border color based on cache state
            color = self.BORDER_COLORS.get(cache_state, self.BORDER_COLORS['clean'])
            
            # Set border width based on state
            if cache_state == 'error':
                border_width = 3.0  # Thick for errors
            elif cache_state == 'executing':
                border_width = 2.5  # Medium-thick for executing
            elif cache_state in ['cached', 'dirty']:
                border_width = 2.0  # Visible for active states
            else:
                border_width = 1.2  # Default for clean
        
        # Set the border color and width on the view
        r, g, b = color
        if hasattr(node, 'view'):
            # Set border color on view with full alpha
            node.view.border_color = (r, g, b, 255)
            # Set custom border width
            if hasattr(node.view, 'set_border_width'):
                node.view.set_border_width(border_width)
            # Also set on model to keep them in sync
            node.model.border_color = (r, g, b, 255)
            # Force the view to redraw
            node.view.update()
    
    def _update_status_bar(self):
        """Update the status bar text."""
        if self.pinned_node is None:
            self.status_bar.setText("Display: None - Double-click a node to pin")
        else:
            auto_status = "Auto-Update ON" if self.auto_update_enabled else "Auto-Update OFF"
            self.status_bar.setText(
                f"Display: {self.pinned_node._base_name} (Pinned) - {auto_status}"
            )
    
    def _on_node_property_changed(self, node):
        """Handle property change on any node."""
        # Mark node as dirty
        if hasattr(node, 'mark_dirty'):
            node.mark_dirty()
            self._update_node_visual_state(node)
        
        # Check if we need to auto-update
        if self.auto_update_enabled and self.pinned_node is not None:
            # Check if changed node is upstream of pinned node
            if self._is_upstream_of(node, self.pinned_node) or node == self.pinned_node:
                # Trigger debounced update
                self._schedule_auto_update()
    
    def _schedule_auto_update(self):
        """Schedule an auto-update with cooldown."""
        if self.is_generating:
            # Spam protection: don't queue if already generating
            print("Update skipped: already generating")
            self.pending_update = True
            return
        
        # Restart the debounce timer
        self.update_timer.stop()
        self.update_timer.start(self.update_cooldown_ms)
        print(f"Auto-update scheduled in {self.update_cooldown_ms}ms")
    
    def _execute_pinned_node(self):
        """Execute the pinned node (called after cooldown)."""
        if self.pinned_node is None or self.is_generating:
            return
        
        try:
            self.is_generating = True
            self.pending_update = False
            
            # Execute the pinned node
            self._execute_node_with_deps(self.pinned_node)
            
            # Get output for visualization
            output_data = None
            if hasattr(self.pinned_node, 'get_output_for_visualization'):
                # Node provides custom visualization (e.g., graph -> heightfield)
                output_data = self.pinned_node.get_output_for_visualization()
            else:
                # Standard output
                output_data = self.pinned_node.get_output_data()
            
            # Update visualization
            if output_data is not None and isinstance(output_data, np.ndarray):
                self._visualize_heightfield(output_data, self.pinned_node._base_name)
            
        except Exception as e:
            print(f"Auto-update error: {e}")
            import traceback
            traceback.print_exc()
            self._update_node_visual_state(self.pinned_node, 'error')
        finally:
            self.is_generating = False
            
            # Check if another update was queued during execution
            if self.pending_update:
                self._schedule_auto_update()
    
    def _is_upstream_of(self, node, target_node):
        """Check if node is upstream of target_node in the graph."""
        # BFS to check if there's a path from node to target_node
        visited = set()
        queue = [node]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Check all output connections
            for output_port in current.output_ports():
                for connected_port in output_port.connected_ports():
                    downstream_node = connected_port.node()
                    if downstream_node == target_node:
                        return True
                    queue.append(downstream_node)
        
        return False
    
    def _on_node_progress_updated(self, node, progress):
        """Handle progress update from a node."""
        if node in self.active_progress_bars:
            self.active_progress_bars[node].set_progress(progress)
    
    def _show_progress_bar(self, node):
        """Show progress bar for a node."""
        if node not in self.active_progress_bars:
            # Progress bar is created as child of node.view
            progress_bar = NodeProgressBar(node.view)
            self.active_progress_bars[node] = progress_bar
            # Start as indeterminate (animated)
            progress_bar.set_indeterminate(True)
    
    def _hide_progress_bar(self, node, execution_time=None):
        """Hide progress bar and optionally show execution time."""
        if node in self.active_progress_bars:
            progress_bar = self.active_progress_bars[node]
            # No need to manually remove from scene - it's a child of node_view
            # Just clean up our reference and stop timers
            if hasattr(progress_bar, 'animation_timer'):
                progress_bar.animation_timer.stop()
            del self.active_progress_bars[node]
        
        # Show execution time label if provided
        if execution_time is not None and execution_time > 0.01:  # Only show if >10ms
            # Remove old label if exists for this specific node
            if node in self.active_execution_labels:
                old_label = self.active_execution_labels[node]
                # No need to manually remove - it's a child of node_view
                del self.active_execution_labels[node]
            
            # Create new label as child of node_view
            label = NodeExecutionLabel(node.view, execution_time)
            self.active_execution_labels[node] = label

    def _execute_node_with_deps(self, node):
        """Execute a node and all its dependencies recursively."""
        if not hasattr(node, '_is_dirty'):
            # Not a TerrainBaseNode, execute directly
            print(f"Executing non-terrain node: {node._base_name}")
            node.execute()
            self._update_node_visual_state(node, 'cached')
            return
        
        # If cached and not dirty, no need to execute
        if not node._is_dirty and node._cached_output is not None:
            print(f"Using cached result for: {node._base_name}")
            self._update_node_visual_state(node, 'cached')
            return
        
        # Execute upstream dependencies first (only if dirty)
        for input_port in node.input_ports():
            for connected_port in input_port.connected_ports():
                upstream_node = connected_port.node()
                if hasattr(upstream_node, '_is_dirty'):
                    self._execute_node_with_deps(upstream_node)
        
        # Now execute this node
        print(f"Executing: {node._base_name}")
        self._update_node_visual_state(node, 'executing')
        
        # Show progress bar
        self._show_progress_bar(node)
        
        try:
            # Time the execution
            start_time = time.time()
            
            node.execute()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            output = node.get_output_data()
            print(f"Executed {node._base_name} in {execution_time:.3f}s, output type: {type(output)}, "
                f"output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
            
            # Hide progress bar and show execution time
            self._hide_progress_bar(node, execution_time)
            
            self._update_node_visual_state(node, 'cached')
            
        except Exception as e:
            print(f"Error executing {node._base_name}: {e}")
            self._hide_progress_bar(node)
            self._update_node_visual_state(node, 'error')
            raise
    
    def _clear_all_caches(self):
        """Clear all node caches."""
        if not self.node_graph:
            return
        
        count = 0
        for node in self.node_graph.all_nodes():
            if hasattr(node, '_cached_output'):
                node._cached_output = None
                node._is_dirty = True
                self._update_node_visual_state(node, 'dirty')
                count += 1
        
        print(f"Cleared {count} node caches")
        QMessageBox.information(self, "Cache Cleared", 
                              f"Cleared cache for {count} nodes")
    
    def _on_node_executed(self, node):
        """Handle node execution completion."""
        self._update_node_visual_state(node, 'cached')
    
    def _visualize_heightfield(self, heightfield: np.ndarray, node_name: str):
        """Visualize a heightfield in both viewports."""
        from terrain_generator.core import TerrainData
        
        dim = heightfield.shape[0]
        
        # Create minimal TerrainData
        land_mask = np.ones((dim, dim), dtype=bool)
        river_volume = np.zeros((dim, dim), dtype=np.float32)
        watershed_mask = np.zeros((dim, dim), dtype=np.int32)
        deposition_map = np.zeros((dim, dim), dtype=np.float32)
        
        terrain_data = TerrainData(
            heightmap=heightfield,
            land_mask=land_mask,
            river_volume=river_volume,
            watershed_mask=watershed_mask,
            deposition_map=deposition_map,
            rock_map=None,
            triangulation=None,
            rock_types=None,
            rock_albedo=None,
            points=None,
            neighbors=None
        )
        
        terrain_data.dim = dim
        
        # Normalize for display
        h_min, h_max = heightfield.min(), heightfield.max()
        if h_max > h_min:
            terrain_data.heightmap_normalized = (heightfield - h_min) / (h_max - h_min)
        else:
            terrain_data.heightmap_normalized = np.zeros_like(heightfield)
        
        terrain_data.z_min = float(h_min)
        terrain_data.z_max = float(h_max)
        
        # Update viewports
        self.node_viewport.set_terrain(terrain_data)
        if self.main_terrain_viewport is not None:
            self.main_terrain_viewport.set_terrain(terrain_data)
        
        print(f"Visualized {node_name}: {dim}x{dim}, range=[{h_min:.3f}, {h_max:.3f}]")
        self.node_visualized.emit(node_name)
    
    def _on_nodes_deleted(self, nodes):
        """Handle node deletion."""
        for node in nodes:
            # Clean up our tracking dictionaries
            # (The widgets themselves are children of node_view and will be auto-deleted)
            if node in self.active_execution_labels:
                del self.active_execution_labels[node]
            
            if node in self.active_progress_bars:
                progress_bar = self.active_progress_bars[node]
                if hasattr(progress_bar, 'animation_timer'):
                    progress_bar.animation_timer.stop()
                del self.active_progress_bars[node]
            
            # Handle special nodes
            if node == self.pinned_node:
                self.pinned_node = None
                self.unpin_btn.setEnabled(False)
                self._update_status_bar()
            
            if node == self.map_properties_node:
                print("Warning: Map Properties node was deleted!")
                self.map_properties_node = None
    
    def eventFilter(self, obj, event):
        """Handle key events from the node graph widget."""
        if self.node_graph and obj in (
            self.node_graph.widget,
            self.node_graph.viewer(),
            self.node_graph.viewer().viewport(),
        ):
            if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._delete_selected_nodes()
                return True
        return super().eventFilter(obj, event)
    
    def _delete_selected_nodes(self):
        """Delete currently selected nodes, skipping protected ones."""
        if not self.node_graph:
            return
        
        selected = self.node_graph.selected_nodes()
        protected_nodes = [self.map_properties_node]
        
        for node in selected:
            if node in protected_nodes:
                print(f"Cannot delete protected node: {node._base_name if hasattr(node, '_base_name') else node.name()}")
                continue
            
            # Check if it's the pinned node
            if node == self.pinned_node:
                self.pinned_node = None
                self.unpin_btn.setEnabled(False)
                self._update_status_bar()
            
            # Delete the node
            self.node_graph.delete_node(node)
    
    def clear_graph(self):
        """Clear the entire graph except the Map Properties node."""
        if not self.node_graph:
            return
        
        reply = QMessageBox.question(
            self,
            "Clear Graph",
            "Are you sure you want to clear the entire node graph?\n"
            "(Map Properties node will be preserved)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Get all nodes except Map Properties
            all_nodes = self.node_graph.all_nodes()
            for node in all_nodes:
                if node != self.map_properties_node:
                    # Unpin if necessary
                    if node == self.pinned_node:
                        self.pinned_node = None
                        self.unpin_btn.setEnabled(False)
                    
                    self.node_graph.delete_node(node)
            
            self._update_status_bar()
            print("Graph cleared")
