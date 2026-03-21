"""Node editor widget with typed payloads, async execution, and graph persistence."""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from NodeGraphQt import NodeGraph
from NodeGraphQt.constants import PipeLayoutEnum
from NodeGraphQt.widgets.viewer import NodeViewer
from PIL import Image
from PyQt5.QtCore import QEvent, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QContextMenuEvent
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolButton,
    QUndoStack,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from ..io import TerrainExporter
from ..visualization import TerrainViewport
from .nodes.context import ExecutionCancellationToken, NodeExecutionCancelled
from .nodes import (
    AETHeuristicNode,
    AlbedoHeuristicNode,
    ApplyRiverDowncuttingNode,
    AridityHeuristicNode,
    AspectHeuristicNode,
    AssignRockLayersNode,
    BiomeHeuristicBundleNode,
    BiomeHeuristicNode,
    BuildErosionParameterMapsNode,
    BundleTerrainOutputsNode,
    ClimateHeuristicBundleNode,
    CombineNode,
    ConnectInlandSeasNode,
    ComputeRiverNetworkNode,
    ConstantNode,
    ContinuousAlbedoHeuristicNode,
    CurvatureHeuristicNode,
    CurveRemapNode,
    DomainWarpNode,
    FBMNode,
    FlowAccumulationHeuristicNode,
    FoliageColorHeuristicNode,
    ForestDensityHeuristicNode,
    GaussianBlurNode,
    GroundcoverDensityHeuristicNode,
    GRAPH_SCHEMA_VERSION,
    HeightfieldData,
    ImportHeightmapNode,
    InvertNode,
    LandMaskNode,
    MapOverlayData,
    NodeProgressBar,
    NodeExecutionLabel,
    NormalizeClampNode,
    NormalHeuristicNode,
    ParticleErosionNode,
    PETHeuristicNode,
    PrecipitationHeuristicNode,
    ProjectSettingsNode,
    RasterizeGraphFieldNode,
    RockStackWarpNode,
    SampleTerrainGraphNode,
    save_graph_payload,
    load_graph_payload,
    build_graph_payload,
    ShapeNode,
    SlopeHeuristicNode,
    SolveBaseGraphElevationNode,
    SVFHeuristicNode,
    TopographicHeuristicBundleNode,
    TPIHeuristicNode,
    TWIHeuristicNode,
    TemperatureHeuristicNode,
    TerrainBaseNode,
    TerrainBundleData,
    TerraceMaxDeltaNode,
    ThresholdFloodNode,
    terrain_data_from_bundle,
    terrain_data_from_heightfield,
    rgba_from_mask,
    UnbundleTerrainBundleNode,
    ViewerNode,
    WatershedMaskOverlayNode,
    WorldSettingsNode,
    port_type_for_payload,
    SettingsData,
    MaskData,
)

NODE_MENU_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, Type[TerrainBaseNode]], ...]], ...] = (
    (
        "View",
        (
            ("Viewer", ViewerNode),
        ),
    ),
    (
        "Settings",
        (
            ("Project Settings", ProjectSettingsNode),
            ("World Settings", WorldSettingsNode),
        ),
    ),
    (
        "Terrain",
        (
            ("FBM Noise", FBMNode),
            ("Constant", ConstantNode),
            ("Import Heightmap", ImportHeightmapNode),
            ("Shape Mask", ShapeNode),
            ("Combine", CombineNode),
            ("Domain Warp", DomainWarpNode),
            ("Curve Remap", CurveRemapNode),
            ("Threshold/Flood", ThresholdFloodNode),
            ("Gaussian Blur", GaussianBlurNode),
            ("Connect Inland Lakes", ConnectInlandSeasNode),
            ("Invert", InvertNode),
            ("Normalize/Clamp", NormalizeClampNode),
            ("Land Mask", LandMaskNode),
        ),
    ),
    (
        "Graph",
        (
            ("Sample Terrain Graph", SampleTerrainGraphNode),
            ("Solve Base Graph Elevation", SolveBaseGraphElevationNode),
            ("Terrace/Max Delta", TerraceMaxDeltaNode),
            ("Rock Stack Warp", RockStackWarpNode),
            ("Assign Rock Layers", AssignRockLayersNode),
            ("Compute River Network", ComputeRiverNetworkNode),
            ("Watershed Mask Overlay", WatershedMaskOverlayNode),
            ("Apply River Downcutting", ApplyRiverDowncuttingNode),
            ("Rasterize Graph Field", RasterizeGraphFieldNode),
            ("Bundle Terrain Outputs", BundleTerrainOutputsNode),
            ("Unbundle Terrain Bundle", UnbundleTerrainBundleNode),
            ("Build Erosion Maps", BuildErosionParameterMapsNode),
            ("Particle Erosion", ParticleErosionNode),
        ),
    ),
    (
        "Heuristics",
        (
            ("Climate Maps", ClimateHeuristicBundleNode),
            ("Topographic Maps", TopographicHeuristicBundleNode),
            ("Biome & Cover Maps", BiomeHeuristicBundleNode),
        ),
    ),
)

LEGACY_REGISTER_ONLY_NODES: Tuple[Type[TerrainBaseNode], ...] = (
    SlopeHeuristicNode,
    AspectHeuristicNode,
    NormalHeuristicNode,
    CurvatureHeuristicNode,
    TPIHeuristicNode,
    FlowAccumulationHeuristicNode,
    TWIHeuristicNode,
    SVFHeuristicNode,
    TemperatureHeuristicNode,
    PrecipitationHeuristicNode,
    PETHeuristicNode,
    AETHeuristicNode,
    AridityHeuristicNode,
    BiomeHeuristicNode,
    AlbedoHeuristicNode,
    ContinuousAlbedoHeuristicNode,
    FoliageColorHeuristicNode,
    ForestDensityHeuristicNode,
    GroundcoverDensityHeuristicNode,
)


class MacFriendlyNodeViewer(NodeViewer):
    """Node viewer tweaks for right-drag panning and graph delete shortcuts."""

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
        if event.key() in (Qt.Key_Delete, Qt.Key_X):
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


class GraphExecutionThread(QThread):
    """Execute a node graph in the background and parallelize ready branches."""

    started_node = pyqtSignal(object)
    finished_node = pyqtSignal(object, float)
    failed_node = pyqtSignal(object, str)
    completed = pyqtSignal(object, object)
    cancelled = pyqtSignal(object)

    def __init__(self, root_node, max_workers: Optional[int] = None):
        super().__init__()
        self.root_node = root_node
        self.max_workers = max_workers or max(1, os.cpu_count() or 1)
        self._cancel_token = ExecutionCancellationToken()

    def request_cancel(self):
        self._cancel_token.cancel()

    def run(self):
        executor: Optional[ThreadPoolExecutor] = None
        try:
            dependency_map = self._build_dependency_map(self.root_node, set(), {})
            executable_nodes = {
                node for node in dependency_map
                if isinstance(node, TerrainBaseNode)
                and (node._is_dirty or node.get_output_data() is None)
            }
            if not executable_nodes:
                self._cancel_token.raise_if_cancelled()
                payload = self.root_node.get_visualization_payload()
                self.completed.emit(self.root_node, payload)
                return

            dependent_map = defaultdict(list)
            pending_counts = {}
            for node, dependencies in dependency_map.items():
                if node not in executable_nodes:
                    continue
                pending_count = 0
                for dependency in dependencies:
                    if dependency in executable_nodes:
                        pending_count += 1
                        dependent_map[dependency].append(node)
                pending_counts[node] = pending_count

            ready = deque(node for node, pending in pending_counts.items() if pending == 0)
            future_to_node = {}
            max_workers = min(max(1, self.max_workers), max(1, len(executable_nodes)))
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="terrain-node")

            while ready or future_to_node:
                self._cancel_token.raise_if_cancelled()
                while ready and len(future_to_node) < max_workers and not self._cancel_token.is_cancelled():
                    node = ready.popleft()
                    future_to_node[executor.submit(self._execute_single_node, node)] = node
                if not future_to_node:
                    break
                done, _pending = wait(tuple(future_to_node.keys()), timeout=0.1, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    node = future_to_node.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        if self._is_cancellation_exception(exc):
                            self._cancel_token.cancel()
                            raise NodeExecutionCancelled("Execution cancelled.") from exc
                        self._cancel_token.cancel()
                        self.failed_node.emit(node, str(exc))
                        return
                    for downstream_node in dependent_map.get(node, ()):
                        pending_counts[downstream_node] -= 1
                        if pending_counts[downstream_node] == 0:
                            ready.append(downstream_node)

            self._cancel_token.raise_if_cancelled()
            payload = self.root_node.get_visualization_payload()
            self.completed.emit(self.root_node, payload)
        except NodeExecutionCancelled:
            self.cancelled.emit(self.root_node)
        except Exception as exc:
            if self._cancel_token.is_cancelled():
                self.cancelled.emit(self.root_node)
            else:
                self.failed_node.emit(self.root_node, str(exc))
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def _build_dependency_map(self, node, visiting, collected):
        if node in visiting:
            raise RuntimeError(f"Cycle detected while executing '{node.name()}'.")
        if node in collected:
            return collected
        visiting.add(node)
        dependencies = []
        if hasattr(node, "input_ports"):
            for input_port in node.input_ports():
                for connected_port in input_port.connected_ports():
                    upstream_node = connected_port.node()
                    if isinstance(upstream_node, TerrainBaseNode):
                        self._build_dependency_map(upstream_node, visiting, collected)
                        dependencies.append(upstream_node)
        visiting.remove(node)
        collected[node] = tuple(dict.fromkeys(dependencies))
        return collected

    def _execute_single_node(self, node):
        self._cancel_token.raise_if_cancelled()
        if not isinstance(node, TerrainBaseNode):
            return
        if not node._is_dirty and node.get_output_data() is not None:
            return
        self.started_node.emit(node)
        start = time.time()
        node.bind_execution_token(self._cancel_token)
        try:
            node.check_cancelled()
            node.execute()
            node.check_cancelled()
        finally:
            node.clear_execution_token(self._cancel_token)
        elapsed = time.time() - start
        self.finished_node.emit(node, elapsed)

    def _is_cancellation_exception(self, exc: Exception) -> bool:
        if isinstance(exc, NodeExecutionCancelled):
            return True
        return self._cancel_token.is_cancelled() and str(exc).strip().lower() == "execution cancelled."


class NodeEditorWidget(QWidget):
    """Widget containing the node graph editor with integrated visualization."""

    BORDER_COLORS = {
        "clean": (80, 80, 80),
        "cached": (0, 200, 0),
        "dirty": (255, 200, 0),
        "executing": (100, 150, 255),
        "error": (255, 50, 50),
        "pinned": (0, 150, 255),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.node_graph = None
        self.node_viewport = None
        self.project_settings_node = None
        self.world_settings_node = None
        self.pinned_node = None
        self.auto_update_enabled = True
        self.is_generating = False
        self.pending_update = False
        self.executing_nodes: set[object] = set()
        self.active_progress_bars: Dict[object, NodeProgressBar] = {}
        self.active_execution_labels: Dict[object, NodeExecutionLabel] = {}
        self.execution_thread = None
        self.node_type_registry: Dict[str, Type[TerrainBaseNode]] = {}
        self.create_type_lookup: Dict[str, str] = {}
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._execute_pinned_node)
        self.update_cooldown_ms = 400
        self.setup_ui()
        self.setup_node_graph()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        for label, entries in NODE_MENU_GROUPS:
            self._add_node_menu(toolbar_layout, label, entries)

        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self._delete_selected_nodes)
        toolbar_layout.addWidget(self.remove_selected_btn)

        self.save_graph_btn = QPushButton("Save Graph")
        self.save_graph_btn.clicked.connect(self.save_graph_to_file)
        toolbar_layout.addWidget(self.save_graph_btn)

        self.load_graph_btn = QPushButton("Load Graph")
        self.load_graph_btn.clicked.connect(self.load_graph_from_file)
        toolbar_layout.addWidget(self.load_graph_btn)

        self.export_btn = QPushButton("Export Output")
        self.export_btn.clicked.connect(self.export_pinned_output)
        toolbar_layout.addWidget(self.export_btn)

        self.stop_btn = QPushButton("Stop Compute")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_current_execution)
        toolbar_layout.addWidget(self.stop_btn)

        toolbar_layout.addStretch()

        self.auto_update_checkbox = QCheckBox("Auto-Update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_toggled)
        toolbar_layout.addWidget(self.auto_update_checkbox)

        self.unpin_btn = QPushButton("Unpin Display")
        self.unpin_btn.setEnabled(False)
        self.unpin_btn.clicked.connect(self._unpin_node)
        toolbar_layout.addWidget(self.unpin_btn)

        self.clear_btn = QPushButton("Clear Caches")
        self.clear_btn.clicked.connect(self._clear_all_caches)
        toolbar_layout.addWidget(self.clear_btn)

        self.clear_graph_btn = QPushButton("Clear Graph")
        self.clear_graph_btn.clicked.connect(self.clear_graph)
        toolbar_layout.addWidget(self.clear_graph_btn)

        layout.addWidget(toolbar)

        self.status_bar = QLabel("Display: None - Double-click a viewer node to pin")
        self.status_bar.setStyleSheet("background: #2a2a2a; padding: 4px; color: #aaa;")
        layout.addWidget(self.status_bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.graph_container)
        self.node_viewport = TerrainViewport()
        self.splitter.addWidget(self.node_viewport)
        self.splitter.setSizes([640, 640])
        layout.addWidget(self.splitter, stretch=1)

    def _add_node_menu(self, layout, label, entries):
        button = QToolButton()
        button.setText(label)
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        for entry_label, node_cls in entries:
            action = menu.addAction(entry_label)
            action.triggered.connect(lambda _checked=False, cls=node_cls: self.add_node(cls))
        button.setMenu(menu)
        layout.addWidget(button)

    def setup_node_graph(self):
        undo_stack = QUndoStack(self)
        viewer = MacFriendlyNodeViewer(undo_stack=undo_stack, delete_callback=self._delete_selected_nodes)
        viewer.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.node_graph = NodeGraph(viewer=viewer, undo_stack=undo_stack)
        graph_widget = self.node_graph.widget
        graph_widget.installEventFilter(self)
        self.node_graph.viewer().installEventFilter(self)
        self.node_graph.viewer().viewport().installEventFilter(self)
        self.graph_layout.addWidget(graph_widget)
        self.node_graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        self.node_graph.set_zoom(1.0)
        self.node_graph.node_double_clicked.connect(self._on_node_double_clicked)
        self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.node_graph.port_connected.connect(self._on_port_connected)
        self.node_graph.port_disconnected.connect(self._on_port_disconnected)
        self.node_graph.property_changed.connect(self._on_graph_property_changed)
        self.register_nodes()
        self.create_global_nodes()

    def register_nodes(self):
        registered: set[Type[TerrainBaseNode]] = set()
        for node_cls in LEGACY_REGISTER_ONLY_NODES:
            if node_cls in registered:
                continue
            registered.add(node_cls)
            self.node_graph.register_node(node_cls)
            node_type = f"{node_cls.__identifier__}.{node_cls.__name__}"
            self.node_type_registry[node_type] = node_cls
            self.create_type_lookup[node_cls.__name__] = node_type
        for _group_name, entries in NODE_MENU_GROUPS:
            for _entry_label, node_cls in entries:
                if node_cls in registered:
                    continue
                registered.add(node_cls)
                self.node_graph.register_node(node_cls)
                node_type = f"{node_cls.__identifier__}.{node_cls.__name__}"
                self.node_type_registry[node_type] = node_cls
                self.create_type_lookup[node_cls.__name__] = node_type

    def create_global_nodes(self):
        self.project_settings_node = self._create_node_instance(ProjectSettingsNode, pos=(-520, -140))
        self.world_settings_node = self._create_node_instance(WorldSettingsNode, pos=(-520, 160))

    def _create_node_instance(
        self,
        node_cls: Type[TerrainBaseNode],
        *,
        name: Optional[str] = None,
        pos: Tuple[int, int] = (0, 0),
        properties: Optional[Dict[str, object]] = None,
        base_path: Optional[Path] = None,
    ):
        node_type = self.create_type_lookup[node_cls.__name__]
        node = self.node_graph.create_node(node_type, name=name or getattr(node_cls, "NODE_NAME", node_cls.__name__), pos=[int(pos[0]), int(pos[1])])
        node._node_type_name = node_type
        if not hasattr(node, "_persist_id"):
            node._persist_id = uuid.uuid4().hex
        self._setup_node_execution(node)
        if properties:
            allowed_properties = None
            if isinstance(node, TerrainBaseNode):
                properties = node.restore_serialized_properties(properties, base_path=base_path)
                allowed_properties = set(node.serializable_properties().keys())
            for key, value in properties.items():
                if allowed_properties is not None and key not in allowed_properties:
                    continue
                try:
                    node.set_property(key, value)
                except Exception:
                    continue
        self._update_node_visual_state(node)
        return node

    def _current_view_center(self) -> Tuple[int, int]:
        if not self.node_graph:
            return (0, 0)
        viewer = self.node_graph.viewer()
        viewport_rect = viewer.viewport().rect()
        scene_center = viewer.mapToScene(viewport_rect.center())
        return (int(scene_center.x()), int(scene_center.y()))

    def add_node(self, node_cls: Type[TerrainBaseNode]):
        self._create_node_instance(node_cls, pos=self._current_view_center())

    def _setup_node_execution(self, node):
        if not isinstance(node, TerrainBaseNode):
            return
        node.signals.progress_updated.connect(self._on_node_progress_updated)
        node.signals.state_changed.connect(self._on_node_state_changed)
        node.signals.error_emitted.connect(self._on_node_error)
        if not hasattr(node, "_base_name"):
            node._base_name = node.name()
        original_set_property = node.set_property

        def wrapped_set_property(name, value, **kwargs):
            result = original_set_property(name, value, **kwargs)
            if not name.startswith("_") and name not in ("name", "selected", "pos"):
                self._on_node_property_changed(node)
            return result

        node.set_property = wrapped_set_property

    def _on_auto_update_toggled(self, state):
        self.auto_update_enabled = (state == Qt.Checked)
        self._update_status_bar()

    def _on_graph_property_changed(self, node, name, _value):
        if not isinstance(node, TerrainBaseNode):
            return
        if name.startswith("_") or name in ("name", "selected", "pos"):
            return
        self._on_node_property_changed(node)

    def _on_port_connected(self, output_port, input_port):
        self._on_graph_connection_changed(output_port, input_port)

    def _on_port_disconnected(self, output_port, input_port):
        self._on_graph_connection_changed(output_port, input_port)

    def _on_graph_connection_changed(self, output_port, input_port):
        target_node = getattr(input_port, "node", lambda: None)()
        source_node = getattr(output_port, "node", lambda: None)()
        if isinstance(target_node, TerrainBaseNode):
            target_node.mark_dirty()
            self._update_node_visual_state(target_node)
        if isinstance(source_node, TerrainBaseNode):
            self._update_node_visual_state(source_node)
        if self.auto_update_enabled and self.pinned_node is not None:
            if (
                target_node == self.pinned_node
                or source_node == self.pinned_node
                or self._is_upstream_of(target_node, self.pinned_node)
                or self._is_upstream_of(source_node, self.pinned_node)
            ):
                self._schedule_auto_update()

    def _on_node_property_changed(self, node):
        if hasattr(node, "mark_dirty"):
            node.mark_dirty()
            self._update_node_visual_state(node)
        if self.auto_update_enabled and self.pinned_node is not None:
            if (
                node == self.project_settings_node
                or node == self.world_settings_node
                or self._is_upstream_of(node, self.pinned_node)
                or node == self.pinned_node
            ):
                self._schedule_auto_update()

    def _schedule_auto_update(self):
        self.update_timer.stop()
        if self.is_generating:
            self.pending_update = True
            self._cancel_active_execution(reschedule=True, status_text="Cancelling stale execution...")
            return
        self.update_timer.start(self.update_cooldown_ms)

    def _execute_pinned_node(self):
        if self.pinned_node is None or self.is_generating:
            return
        self.execute_node_async(self.pinned_node)

    def execute_node_async(self, node):
        if self.execution_thread and self.execution_thread.isRunning():
            self.pending_update = True
            self._cancel_active_execution(reschedule=True, status_text=f"Cancelling current run for {node._base_name}...")
            return
        self.is_generating = True
        self.pending_update = False
        self.executing_nodes.clear()
        self.execution_thread = GraphExecutionThread(node)
        self.execution_thread.started_node.connect(self._on_thread_node_started)
        self.execution_thread.finished_node.connect(self._on_thread_node_finished)
        self.execution_thread.failed_node.connect(self._on_thread_node_failed)
        self.execution_thread.completed.connect(self._on_thread_execution_completed)
        self.execution_thread.cancelled.connect(self._on_thread_execution_cancelled)
        self.execution_thread.finished.connect(self._on_thread_finished)
        self._refresh_execution_controls()
        self.status_bar.setText(f"Executing: {node._base_name}")
        self.execution_thread.start()

    def _on_thread_node_started(self, node):
        if self.sender() is not self.execution_thread:
            return
        self.executing_nodes.add(node)
        self._show_progress_bar(node)
        self._update_node_visual_state(node, "executing")
        self.status_bar.setText(f"Executing: {node._base_name}")

    def _on_thread_node_finished(self, node, execution_time):
        if self.sender() is not self.execution_thread:
            return
        self.executing_nodes.discard(node)
        self._hide_progress_bar(node, execution_time)
        self._update_node_visual_state(node, "cached")

    def _on_thread_node_failed(self, node, message):
        if self.sender() is not self.execution_thread:
            return
        self.executing_nodes.discard(node)
        self._hide_progress_bar(node)
        self._update_node_visual_state(node, "error")
        QMessageBox.critical(self, "Node Execution Error", f"{node._base_name}: {message}")

    def _on_thread_execution_completed(self, node, payload):
        if self.sender() is not self.execution_thread:
            return
        self._pin_node(node)
        try:
            visualized = self._visualize_payload(payload, node._base_name)
        except Exception as exc:
            self._update_node_visual_state(node, "error")
            self.status_bar.setText(f"{node._base_name}: visualization failed")
            QMessageBox.warning(
                self,
                "Visualization Error",
                f"Could not visualize '{node._base_name}':\n{exc}",
            )
            return
        if visualized:
            self.status_bar.setText(f"Display: {node._base_name} (Pinned)")

    def _on_thread_execution_cancelled(self, _node):
        if self.sender() is not self.execution_thread:
            return
        if self.pending_update:
            self.status_bar.setText("Cancelled stale execution. Recomputing...")
        else:
            self.status_bar.setText("Computation stopped.")

    def _on_thread_finished(self):
        if self.sender() is not self.execution_thread:
            return
        self._clear_running_execution_artifacts()
        self.is_generating = False
        self.execution_thread = None
        self._refresh_execution_controls()
        if self.pending_update and self.pinned_node is not None:
            self._schedule_auto_update()
        else:
            self._update_status_bar()

    def _cancel_active_execution(self, *, reschedule: bool, status_text: Optional[str] = None):
        thread = self.execution_thread
        if thread is None or not thread.isRunning():
            return False
        self.pending_update = reschedule
        thread.request_cancel()
        if status_text:
            self.status_bar.setText(status_text)
        self._refresh_execution_controls()
        return True

    def _stop_current_execution(self):
        self.update_timer.stop()
        self._cancel_active_execution(reschedule=False, status_text="Stopping computation...")

    def _refresh_execution_controls(self):
        self.stop_btn.setEnabled(self.is_generating)

    def _clear_running_execution_artifacts(self):
        for node in list(self.executing_nodes):
            self._hide_progress_bar(node)
            if isinstance(node, TerrainBaseNode):
                state = "cached" if not node._is_dirty and node.get_output_data() is not None else "dirty"
                self._update_node_visual_state(node, state)
        self.executing_nodes.clear()

    def _validate_node_input_types(self, node):
        if not isinstance(node, TerrainBaseNode):
            return
        for port_name, allowed_types in node.INPUT_TYPES.items():
            port = node.inputs().get(port_name)
            if port is None:
                continue
            for source_port in port.connected_ports():
                source_node = source_port.node()
                if not isinstance(source_node, TerrainBaseNode):
                    continue
                data = source_node.get_output_data()
                source_name_attr = getattr(source_port, "name", None)
                if isinstance(data, dict):
                    if callable(source_name_attr):
                        source_name = source_name_attr()
                    elif source_name_attr is not None:
                        source_name = source_name_attr
                    else:
                        source_name = source_port.model.name
                    data = data.get(source_name)
                if data is None:
                    continue
                actual_type = port_type_for_payload(data)
                if allowed_types and actual_type not in allowed_types:
                    raise TypeError(
                        f"Invalid connection into {node._base_name}:{port_name}. "
                        f"Expected {allowed_types}, received '{actual_type}'."
                    )

    def _on_node_double_clicked(self, node):
        if not isinstance(node, ViewerNode):
            self.status_bar.setText("Display: Double-click a viewer node to pin and preview")
            return
        self._validate_node_input_types(node)
        self._pin_node(node)
        self.execute_node_async(node)

    def _pin_node(self, node):
        if self.pinned_node is not None and self.pinned_node is not node:
            self._update_node_visual_state(self.pinned_node)
        self.pinned_node = node
        self.unpin_btn.setEnabled(True)
        self._update_node_visual_state(node)
        self._update_status_bar()

    def _unpin_node(self):
        if self.pinned_node is not None:
            self._update_node_visual_state(self.pinned_node)
        self.pinned_node = None
        self.unpin_btn.setEnabled(False)
        self._update_status_bar()

    def _update_node_visual_state(self, node, cache_state=None):
        is_pinned = (node == self.pinned_node)
        if is_pinned:
            color = self.BORDER_COLORS["pinned"]
            border_width = 4.0
        else:
            if cache_state is None:
                if getattr(node, "_cached_output", None) is not None and not getattr(node, "_is_dirty", True):
                    cache_state = "cached"
                elif getattr(node, "_is_dirty", False):
                    cache_state = "dirty"
                else:
                    cache_state = "clean"
            color = self.BORDER_COLORS.get(cache_state, self.BORDER_COLORS["clean"])
            border_width = 3.0 if cache_state == "error" else 2.5 if cache_state == "executing" else 2.0 if cache_state in {"cached", "dirty"} else 1.2
        if hasattr(node, "view"):
            r, g, b = color
            node.view.border_color = (r, g, b, 255)
            if hasattr(node.view, "set_border_width"):
                node.view.set_border_width(border_width)
            node.model.border_color = (r, g, b, 255)
            node.view.update()

    def _update_status_bar(self):
        if self.pinned_node is None:
            self.status_bar.setText("Display: None - Double-click a viewer node to pin")
        else:
            auto_status = "Auto-Update ON" if self.auto_update_enabled else "Auto-Update OFF"
            self.status_bar.setText(f"Display: {self.pinned_node._base_name} (Pinned) - {auto_status}")

    def _on_node_progress_updated(self, node, progress, message):
        if node not in self.active_progress_bars:
            self._show_progress_bar(node)
        bar = self.active_progress_bars[node]
        if progress <= 0.0:
            bar.set_indeterminate(True)
        else:
            bar.set_progress(progress, message)
            bar.set_message(message)

    def _on_node_state_changed(self, node, state):
        self._update_node_visual_state(node, state)

    def _on_node_error(self, node, message):
        self._update_node_visual_state(node, "error")
        self.status_bar.setText(f"{node._base_name}: {message}")

    def _show_progress_bar(self, node):
        if node in self.active_progress_bars:
            return
        progress_bar = NodeProgressBar(node.view)
        progress_bar.set_indeterminate(True)
        self.active_progress_bars[node] = progress_bar

    def _hide_progress_bar(self, node, execution_time=None):
        if node in self.active_progress_bars:
            progress_bar = self.active_progress_bars[node]
            if hasattr(progress_bar, "animation_timer"):
                progress_bar.animation_timer.stop()
            progress_bar.setParentItem(None)
            del self.active_progress_bars[node]
        if execution_time is not None and execution_time > 0.01:
            if node in self.active_execution_labels:
                old_label = self.active_execution_labels[node]
                old_label.setParentItem(None)
                del self.active_execution_labels[node]
            self.active_execution_labels[node] = NodeExecutionLabel(node.view, execution_time)

    def _is_upstream_of(self, node, target_node):
        if not isinstance(node, TerrainBaseNode) or target_node is None:
            return False
        visited = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for output_port in current.output_ports():
                for connected_port in output_port.connected_ports():
                    downstream_node = connected_port.node()
                    if downstream_node == target_node:
                        return True
                    queue.append(downstream_node)
        return False

    def _visualize_payload(self, payload, node_name):
        if payload is None:
            QMessageBox.warning(
                self,
                "Visualization Unavailable",
                f"'{node_name}' did not produce anything that can be displayed.",
            )
            self.status_bar.setText(f"{node_name}: nothing to visualize")
            return False
        if isinstance(payload, HeightfieldData):
            self.node_viewport.clear_image()
            terrain = terrain_data_from_heightfield(payload)
            self.node_viewport.clear_overlay_image()
            self.node_viewport.set_overlay_visible(False)
            self.node_viewport.set_terrain(terrain)
        elif isinstance(payload, TerrainBundleData):
            self.node_viewport.clear_image()
            terrain = terrain_data_from_bundle(payload)
            self.node_viewport.clear_overlay_image()
            self.node_viewport.set_overlay_visible(False)
            self.node_viewport.set_terrain(terrain)
        elif isinstance(payload, MapOverlayData):
            if payload.metadata.get("viewer_mode") == "2d":
                self.node_viewport.clear_overlay_image()
                self.node_viewport.set_overlay_visible(False)
                self.node_viewport.set_image(payload.rgba)
                return True
            preview_bundle = payload.metadata.get("preview_bundle")
            if isinstance(preview_bundle, TerrainBundleData):
                terrain = terrain_data_from_bundle(preview_bundle)
            else:
                terrain = terrain_data_from_heightfield(payload.base_heightfield)
            overlay_opacity = payload.metadata.get("overlay_opacity", ViewerNode.DEFAULT_OVERLAY_OPACITY)
            try:
                overlay_opacity = float(overlay_opacity)
            except (TypeError, ValueError):
                overlay_opacity = ViewerNode.DEFAULT_OVERLAY_OPACITY
            overlay_opacity = max(0.0, min(1.0, overlay_opacity))
            self.node_viewport.clear_image()
            self.node_viewport.set_terrain(terrain)
            self.node_viewport.set_overlay_opacity(overlay_opacity)
            self.node_viewport.set_overlay_image(payload.rgba)
            self.node_viewport.set_overlay_visible(True)
        elif isinstance(payload, MaskData):
            self.node_viewport.clear_overlay_image()
            self.node_viewport.set_overlay_visible(False)
            self.node_viewport.set_image(rgba_from_mask(payload))
        elif isinstance(payload, SettingsData):
            self.status_bar.setText(f"{node_name}: settings node executed")
            QMessageBox.information(
                self,
                "Settings Node",
                f"'{node_name}' provides shared settings and does not have a terrain preview.",
            )
            return False
        else:
            QMessageBox.warning(
                self,
                "Visualization Unsupported",
                f"'{node_name}' produced '{type(payload).__name__}', which does not have a viewer yet.",
            )
            self.status_bar.setText(f"{node_name}: unsupported preview type")
            return False
        return True

    def _clear_all_caches(self):
        if not self.node_graph:
            return
        if self.is_generating:
            self._stop_current_execution()
            QMessageBox.information(self, "Cache Clear Deferred", "Stopped the active computation. Clear caches again once the run has finished.")
            return
        for node in self.node_graph.all_nodes():
            if hasattr(node, "_cached_output"):
                node._cached_output = None
                node._is_dirty = True
                self._update_node_visual_state(node, "dirty")
        if self.project_settings_node is not None:
            self.project_settings_node.context.clear_runtime_caches()
        QMessageBox.information(self, "Cache Cleared", "Cleared all node caches.")

    def _serialize_graph(self, *, base_path=None) -> Dict[str, object]:
        nodes_payload = []
        connections = []
        node_ids: Dict[object, str] = {}
        for node in self.node_graph.all_nodes():
            persist_id = getattr(node, "_persist_id", uuid.uuid4().hex)
            node._persist_id = persist_id
            node_ids[node] = persist_id
            pos_value = node.get_property("pos") if hasattr(node, "get_property") else [0, 0]
            nodes_payload.append(
                {
                    "id": persist_id,
                    "node_type": getattr(node, "_node_type_name", self.create_type_lookup.get(node.__class__.__name__, "")),
                    "name": node.name(),
                    "pos": list(pos_value) if pos_value is not None else [0, 0],
                    "properties": node.serializable_properties(base_path=base_path) if isinstance(node, TerrainBaseNode) else {},
                }
            )
        for node in self.node_graph.all_nodes():
            source_node_id = node_ids[node]
            outputs = getattr(node, "outputs", lambda: {})()
            for port_name, output_port in outputs.items():
                for connected_port in output_port.connected_ports():
                    target_node = connected_port.node()
                    if target_node not in node_ids:
                        continue
                    target_name = getattr(connected_port, "name", None)
                    if callable(target_name):
                        target_port_name = target_name()
                    elif target_name is not None:
                        target_port_name = target_name
                    else:
                        target_port_name = connected_port.model.name
                    connections.append(
                        {
                            "source_node_id": source_node_id,
                            "source_port": port_name,
                            "target_node_id": node_ids[target_node],
                            "target_port": target_port_name,
                        }
                    )
        pinned_id = node_ids.get(self.pinned_node) if self.pinned_node is not None else None
        return build_graph_payload(nodes=nodes_payload, connections=connections, pinned_node_id=pinned_id)

    @staticmethod
    def _migrated_distance_value(value: Any, scale: float) -> Any:
        try:
            scaled = float(value) * scale
        except (TypeError, ValueError):
            return value
        if isinstance(value, str):
            return f"{scaled:.12g}"
        return scaled

    def _migrate_legacy_graph_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        try:
            version = int(payload.get("version", 1))
        except (TypeError, ValueError):
            version = 1
        if version >= GRAPH_SCHEMA_VERSION:
            return payload

        migrated = dict(payload)
        raw_nodes = payload.get("nodes", []) or []
        nodes_data = [dict(entry) for entry in raw_nodes if isinstance(entry, dict)]

        dimension = 1024.0
        terrain_size_km = 0.0
        fallback_cellsize = 1500.0
        for entry in nodes_data:
            node_type = entry.get("node_type")
            properties = entry.get("properties", {})
            if not isinstance(properties, dict):
                continue
            if node_type == "terrain.ProjectSettingsNode":
                try:
                    dimension = float(properties.get("dimension", 1024) or 1024)
                except (TypeError, ValueError):
                    dimension = 1024.0
            elif node_type == "terrain.WorldSettingsNode":
                try:
                    terrain_size_km = float(properties.get("terrain_size_km", 0) or 0)
                except (TypeError, ValueError):
                    terrain_size_km = 0.0
                try:
                    fallback_cellsize = float(properties.get("cellsize", 1500) or 1500)
                except (TypeError, ValueError):
                    fallback_cellsize = 1500.0
        if terrain_size_km <= 0.0:
            terrain_size_km = max(fallback_cellsize, 1e-6) * max(dimension, 1.0) / 1000.0
        scale = terrain_size_km / 1024.0

        for entry in nodes_data:
            properties = entry.get("properties", {})
            if not isinstance(properties, dict):
                continue
            migrated_properties = dict(properties)
            node_type = entry.get("node_type")
            if node_type == "terrain.WorldSettingsNode":
                try:
                    saved_terrain_size = float(migrated_properties.get("terrain_size_km", 0) or 0)
                except (TypeError, ValueError):
                    saved_terrain_size = 0.0
                if saved_terrain_size <= 0.0:
                    terrain_value = f"{terrain_size_km:.12g}" if isinstance(migrated_properties.get("terrain_size_km"), str) else terrain_size_km
                    migrated_properties["terrain_size_km"] = terrain_value
                if "biome_mixing" in migrated_properties:
                    migrated_properties["biome_mixing"] = self._migrated_distance_value(migrated_properties["biome_mixing"], scale)
            elif node_type == "terrain.SampleTerrainGraphNode" and "disc_radius" in migrated_properties:
                migrated_properties["disc_radius"] = self._migrated_distance_value(migrated_properties["disc_radius"], scale)
            elif node_type == "terrain.DomainWarpNode" and "offset_amplitude" in migrated_properties:
                migrated_properties["offset_amplitude"] = self._migrated_distance_value(migrated_properties["offset_amplitude"], scale)
            elif node_type == "terrain.GaussianBlurNode" and "sigma" in migrated_properties:
                migrated_properties["sigma"] = self._migrated_distance_value(migrated_properties["sigma"], scale)
            elif node_type == "terrain.ConnectInlandSeasNode" and "channel_width" in migrated_properties:
                migrated_properties["channel_width"] = self._migrated_distance_value(migrated_properties["channel_width"], scale)
            elif node_type == "terrain.ParticleErosionNode" and "erosion_step_size" in migrated_properties:
                migrated_properties["erosion_step_size"] = self._migrated_distance_value(migrated_properties["erosion_step_size"], scale)
            entry["properties"] = migrated_properties

        migrated["version"] = GRAPH_SCHEMA_VERSION
        migrated["nodes"] = nodes_data
        return migrated

    def _apply_graph_payload(self, payload: Dict[str, object], *, base_path=None):
        payload = self._migrate_legacy_graph_payload(payload)
        nodes_data = payload.get("nodes", []) or []
        connections = payload.get("connections", []) or []
        pinned_id = payload.get("pinned_node_id")
        self.clear_graph(skip_confirmation=True, recreate_globals=False)
        created = {}
        for entry in nodes_data:
            node_type = entry.get("node_type")
            node_cls = self.node_type_registry.get(node_type)
            if node_cls is None:
                continue
            properties = entry.get("properties", {})
            node = self._create_node_instance(
                node_cls,
                name=entry.get("name"),
                pos=tuple(entry.get("pos", [0, 0])),
                properties=properties if isinstance(properties, dict) else None,
                base_path=base_path,
            )
            node._persist_id = entry.get("id", uuid.uuid4().hex)
            created[node._persist_id] = node
            if isinstance(node, ProjectSettingsNode):
                self.project_settings_node = node
            elif isinstance(node, WorldSettingsNode):
                self.world_settings_node = node
        for link in connections:
            src_node = created.get(link.get("source_node_id"))
            dst_node = created.get(link.get("target_node_id"))
            if src_node is None or dst_node is None:
                continue
            source_port = src_node.outputs().get(link.get("source_port"))
            target_port = dst_node.inputs().get(link.get("target_port"))
            if source_port is None or target_port is None:
                continue
            try:
                source_port.connect_to(target_port)
            except Exception:
                continue
        if self.project_settings_node is None:
            self.project_settings_node = self._create_node_instance(ProjectSettingsNode, pos=(-520, -140))
        if self.world_settings_node is None:
            self.world_settings_node = self._create_node_instance(WorldSettingsNode, pos=(-520, 160))
        if pinned_id and pinned_id in created:
            self._pin_node(created[pinned_id])

    def save_graph_to_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Node Graph", "", "Node Graph (*.terrain_graph.json *.json);;All Files (*)")
        if not filename:
            return
        payload = self._serialize_graph(base_path=Path(filename).resolve().parent)
        saved = save_graph_payload(filename, payload)
        self.status_bar.setText(f"Saved graph: {saved.name}")

    def load_graph_from_file(self):
        if self.is_generating:
            self._stop_current_execution()
            QMessageBox.information(self, "Load Deferred", "Stopped the active computation. Load the graph again once the run has finished.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Load Node Graph", "", "Node Graph (*.terrain_graph.json *.json);;All Files (*)")
        if not filename:
            return
        payload = load_graph_payload(filename)
        self._apply_graph_payload(payload, base_path=Path(filename).resolve().parent)
        self.status_bar.setText(f"Loaded graph: {os.path.basename(filename)}")

    def _on_nodes_deleted(self, nodes):
        for node in nodes:
            self.active_progress_bars.pop(node, None)
            self.active_execution_labels.pop(node, None)
            if node == self.pinned_node:
                self.pinned_node = None
                self.unpin_btn.setEnabled(False)
            if node == self.project_settings_node:
                self.project_settings_node = None
            if node == self.world_settings_node:
                self.world_settings_node = None
        self._update_status_bar()

    def _delete_selected_nodes(self):
        if not self.node_graph:
            return
        if self.is_generating:
            self._stop_current_execution()
            QMessageBox.information(self, "Delete Deferred", "Stopped the active computation. Delete the selected nodes again once the run has finished.")
            return
        protected_nodes = {self.project_settings_node, self.world_settings_node}
        for node in list(self.node_graph.selected_nodes()):
            if node in protected_nodes:
                continue
            self.node_graph.delete_node(node)

    def clear_graph(self, skip_confirmation: bool = False, recreate_globals: bool = True):
        if not self.node_graph:
            return
        if self.is_generating:
            self._stop_current_execution()
            if not skip_confirmation:
                QMessageBox.information(self, "Clear Deferred", "Stopped the active computation. Clear the graph again once the run has finished.")
            return
        if not skip_confirmation:
            reply = QMessageBox.question(
                self,
                "Clear Graph",
                "Are you sure you want to clear the node graph?\n(Project and World settings will be recreated.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        for node in list(self.node_graph.all_nodes()):
            self.node_graph.delete_node(node)
        self.project_settings_node = None
        self.world_settings_node = None
        self.pinned_node = None
        if recreate_globals:
            self.create_global_nodes()
        self._update_status_bar()

    def _current_visual_payload(self):
        if self.pinned_node is None or not isinstance(self.pinned_node, TerrainBaseNode):
            return None
        return self.pinned_node.get_visualization_payload()

    @staticmethod
    def _save_overlay_to_path(overlay_array: np.ndarray, filename: str):
        array = np.asarray(overlay_array)
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] == 3:
            mode = "RGB"
        elif array.ndim == 3 and array.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError("Unsupported overlay format for export.")
        Image.fromarray(array.astype(np.uint8), mode=mode).save(filename)

    def export_pinned_output(self):
        payload = self._current_visual_payload()
        if payload is None:
            QMessageBox.information(self, "Export Output", "Pin and execute a node before exporting.")
            return

        node_name = self.pinned_node._base_name if self.pinned_node is not None else "output"
        safe_name = node_name.lower().replace(" ", "_")

        if isinstance(payload, HeightfieldData):
            self._export_heightfield(payload.array, default_stem=safe_name)
            return

        if isinstance(payload, TerrainBundleData):
            self._export_heightfield(payload.heightfield.array, default_stem=safe_name)
            return

        if isinstance(payload, MapOverlayData):
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Overlay",
                f"{safe_name}.png",
                "PNG Files (*.png);;TIFF Files (*.tiff *.tif)",
            )
            if not filename:
                return
            try:
                self._save_overlay_to_path(payload.rgba, filename)
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))
                return
            QMessageBox.information(self, "Export Successful", f"Saved overlay to {filename}")
            return

        QMessageBox.information(
            self,
            "Export Unavailable",
            f"'{type(payload).__name__}' does not have an export path yet.",
        )

    def _export_heightfield(self, array: np.ndarray, *, default_stem: str):
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Heightmap",
            f"{default_stem}.png",
            "PNG (8-bit) (*.png);;PNG (16-bit) (*.png);;TIFF (32-bit float) (*.tiff *.tif)",
        )
        if not filename:
            return
        format_map = {
            "PNG (8-bit) (*.png)": "PNG_8",
            "PNG (16-bit) (*.png)": "PNG_16",
            "TIFF (32-bit float) (*.tiff *.tif)": "TIFF_32",
        }
        export_format = format_map.get(selected_filter, "PNG_8")
        try:
            TerrainExporter.export_heightmap(np.asarray(array, dtype=np.float32), filename, export_format)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(self, "Export Successful", f"Saved heightmap to {filename}")

    def eventFilter(self, obj, event):
        if self.node_graph and obj in (self.node_graph.widget, self.node_graph.viewer(), self.node_graph.viewer().viewport()):
            if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Delete, Qt.Key_X):
                self._delete_selected_nodes()
                return True
        return super().eventFilter(obj, event)
