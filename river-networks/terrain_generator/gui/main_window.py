"""Main application window."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from numba import njit, prange

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QProgressBar, QLabel, QPushButton,
                            QMessageBox, QFileDialog, QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat

from ..config import PresetError
from ..core import (
    TerrainGenerator, TerrainParameters, TerrainData, 
    normalize, _labels_to_rgba, _build_palette_u8, 
    _deposition_to_rgba, _gray_to_rgba_norm)
from ..visualization import TerrainViewport
from ..io import TerrainExporter
from ..heuristics import HeuristicEngine, HeuristicSettings, qimage_to_rgba
from .controls import ControlPanel, AnalysisPanel
from .node_editor import NodeEditorWidget

try:
    import qdarktheme
    DARK_THEME_AVAILABLE = True
except ImportError:
    DARK_THEME_AVAILABLE = False

class TerrainGeneratorThread(QThread):
    """Thread for terrain generation."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # TerrainData
    error = pyqtSignal(str)
    
    def __init__(self, params: TerrainParameters):
        super().__init__()
        self.params = params
    
    def run(self):
        """Run terrain generation."""
        try:
            generator = TerrainGenerator(self.params)
            terrain_data = generator.generate(
                progress_callback=lambda v, m: self.progress.emit(v, m)
            )
            self.finished.emit(terrain_data)
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class TerrainPreviewThread(QThread):
    """Thread for terrain preview generation."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # TerrainData
    error = pyqtSignal(str)
    
    def __init__(self, params: TerrainParameters):
        super().__init__()
        self.params = params
    
    def run(self):
        """Run terrain preview generation."""
        try:
            generator = TerrainGenerator(self.params)
            terrain_data = generator.generate_preview(
                progress_callback=lambda v, m: self.progress.emit(v, m)
            )
            self.finished.emit(terrain_data)
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)



class HeuristicComputationThread(QThread):
    """Thread that wraps the heuristic engine computations."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, dict)  # overlays, raw arrays
    error = pyqtSignal(str)

    def __init__(self, heightmap: np.ndarray, request: dict, 
                 flow_override: Optional[np.ndarray] = None,
                 deposition_map: Optional[np.ndarray] = None,
                 rock_map: Optional[np.ndarray] = None,
                 rock_types: Optional[List[str]] = None,
                 rock_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None):
        super().__init__()
        self.heightmap = np.asarray(heightmap, dtype=np.float32)
        self.request = request
        self.flow_override = None if flow_override is None else np.asarray(flow_override, dtype=np.float32)
        self.deposition_map = None if deposition_map is None else np.asarray(deposition_map, dtype=np.float32)
        self.rock_map = None if rock_map is None else np.ascontiguousarray(rock_map, dtype=np.int32)
        if rock_types is not None:
            self.rock_types: Optional[Tuple[str, ...]] = tuple(str(name) for name in rock_types)
        else:
            self.rock_types = None
        if rock_colors is not None:
            normalized: list[Optional[Tuple[int, int, int]]] = []
            for entry in rock_colors:
                if entry is None:
                    normalized.append(None)
                    continue
                try:
                    components = tuple(int(max(0, min(255, float(c)))) for c in entry[:3])
                except (TypeError, ValueError):
                    normalized.append(None)
                    continue
                normalized.append(components)
            self.rock_colors: Optional[Tuple[Optional[Tuple[int, int, int]], ...]] = tuple(normalized)
        else:
            self.rock_colors = None

    def run(self):
        try:
            settings_kwargs = dict(self.request.get('settings', {}))
            if 'tpi_radii' in settings_kwargs:
                settings_kwargs['tpi_radii'] = tuple(settings_kwargs['tpi_radii'])
            settings_kwargs['albedo_mode'] = 'physical'
            settings = HeuristicSettings(**settings_kwargs)

            overrides_input = self.request.get('cellsize_overrides', {}) or {}
            cellsize_overrides: Dict[str, float] = {}
            for key, value in overrides_input.items():
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if value_f > 0.0:
                    cellsize_overrides[str(key)] = value_f

            selections = list(self.request.get('selections', []))
            dependency_overrides = {
                'albedo': ['slope', 'twi', 'foliage', 'forest_density', 'groundcover_density'],
                'albedo_continuous': ['slope', 'twi', 'foliage', 'forest_density', 'groundcover_density'],
            }
            for source, deps in dependency_overrides.items():
                if source in cellsize_overrides:
                    for dep in deps:
                        cellsize_overrides.setdefault(dep, cellsize_overrides[source])

            if 'albedo' in selections:
                for dep in ['slope', 'twi', 'foliage', 'forest_density', 'groundcover_density']:
                    if dep not in selections:
                        selections.append(dep)

            overlays: Dict[str, np.ndarray] = {}
            arrays: Dict[str, np.ndarray] = {}

            def forward_progress(message: str, percent: int):
                self.progress.emit(percent, message)

            def apply_flow_override(engine_instance: HeuristicEngine):
                if self.flow_override is None:
                    return
                qt = engine_instance.qt_engine
                override = np.asarray(self.flow_override, dtype=np.float32)
                qt.cache['acc'] = override.copy()
                qt.params['flowacc_texture'] = None
                for key in ['twi', 'foliage_rgb', 'forest_density', 'groundcover_density', 'albedo_rgb']:
                    qt.cache.pop(key, None)

            groups: Dict[float, List[str]] = {}
            default_cellsize = settings.cellsize
            for sel in selections:
                base_key = sel.split('@', 1)[0]
                group_value = cellsize_overrides.get(base_key, default_cellsize)
                groups.setdefault(group_value, []).append(sel)

            for group_cellsize, group_selections in groups.items():
                if not group_selections:
                    continue
                group_kwargs = dict(settings_kwargs)
                group_kwargs['cellsize'] = group_cellsize
                group_settings = HeuristicSettings(**group_kwargs)

                engine_instance = HeuristicEngine()
                qt = engine_instance.qt_engine
                qt.progress.connect(forward_progress)
                try:
                    engine_instance.prepare(self.heightmap, group_settings)
                    if self.rock_map is not None:
                        engine_instance.inject_rock_map(self.rock_map, self.rock_types, self.rock_colors)
                    if self.deposition_map is not None:
                        engine_instance.inject_deposition_map(self.deposition_map)
                    apply_flow_override(engine_instance)
                    images, arrs = engine_instance.compute(group_selections)
                    overlays.update({name: qimage_to_rgba(image) for name, image in images.items()})
                    arrays.update(arrs)
                finally:
                    try:
                        qt.progress.disconnect(forward_progress)
                    except TypeError:
                        pass

            self.finished.emit(overlays, arrays)
        except Exception as exc:
            import traceback
            error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class TerrainGeneratorWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.generator_thread = None
        self.preview_thread = None
        self.heuristic_thread = None
        self.current_terrain_data = None
        self.current_overlay_source = None  # ('file', path) or ('computed', key)
        self.computed_overlays = {}
        self.heuristic_arrays = {}

        self.setup_ui()
        self.preset_manager = self.control_panel.preset_manager
        self.last_preset_directory = str(self.preset_manager.default_directory())
        self.setup_connections()
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("3D Terrain Generator with River Networks")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create the traditional terrain view tab
        self.terrain_tab = self.create_terrain_tab()
        self.tab_widget.addTab(self.terrain_tab, "Terrain View")
        
        # Create the node editor tab
        self.node_editor_tab = NodeEditorWidget()
        self.tab_widget.addTab(self.node_editor_tab, "Node Editor")
        
        # Apply theme if available
        if DARK_THEME_AVAILABLE:
            self.setStyleSheet(qdarktheme.load_stylesheet())

    def create_terrain_tab(self) -> QWidget:
        """Create the traditional terrain generation view."""
        terrain_widget = QWidget()
        terrain_layout = QHBoxLayout(terrain_widget)
        terrain_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left control panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(650)
        self.control_panel.setMinimumWidth(650)
        self.control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        terrain_layout.addWidget(self.control_panel)
        
        # Center visualization stack
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        
        self.terrain_viewport = TerrainViewport()
        self.terrain_viewport.setMinimumHeight(800)
        center_layout.addWidget(self.terrain_viewport)
        
        status_container = QWidget()
        status_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_container.setMaximumHeight(28)
        status_row = QHBoxLayout(status_container)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setMinimumHeight(18)
        status_row.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setMinimumWidth(260)
        self.progress_bar.setMaximumWidth(400)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_row.addWidget(self.progress_bar)
        
        center_layout.addWidget(status_container)
        
        terrain_layout.addLayout(center_layout, stretch=1)
        
        # Right analysis panel
        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.setMaximumWidth(450)
        self.analysis_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        terrain_layout.addWidget(self.analysis_panel)
        
        # Initialize visualization settings from analysis panel defaults
        height_control = self.analysis_panel.visual_controls['height_scale']
        sun_control = self.analysis_panel.visual_controls['sun_altitude']
        self.terrain_viewport.set_height_scale(height_control.value())
        self.terrain_viewport.set_sun_altitude(sun_control.value())
        
        return terrain_widget
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Generation buttons
        self.control_panel.generate_button.clicked.connect(self.generate_terrain)
        self.control_panel.preview_button.clicked.connect(self.generate_preview)  # Connect preview button
        
        # Export buttons
        self.control_panel.export_button.clicked.connect(self.export_terrain)
        self.control_panel.export_flow_button.clicked.connect(self.export_flow_mask)
        self.control_panel.export_watershed_button.clicked.connect(self.export_watershed_mask)
        self.control_panel.export_deposition_button.clicked.connect(self.export_deposition_mask)
        self.control_panel.export_rock_button.clicked.connect(self.export_rock_map)
        self.control_panel.load_preset_button.clicked.connect(self.load_preset_from_file)
        self.control_panel.save_preset_button.clicked.connect(self.save_preset_to_file)

        # Visualization controls
        self.analysis_panel.visualization_changed.connect(self.update_visualization)
        self.analysis_panel.overlay_selected.connect(self.load_overlay_texture)
        self.analysis_panel.overlay_cleared.connect(self.clear_overlay_texture)
        self.analysis_panel.heuristics_requested.connect(self.compute_heuristics)
        self.analysis_panel.computed_overlay_requested.connect(self.apply_computed_overlay)
        self.analysis_panel.export_computed_overlay_requested.connect(self.export_computed_overlay)
        self.analysis_panel.export_all_computed_requested.connect(self.export_all_computed_overlays)

        # Node editor connections
        self.node_editor_tab.set_main_terrain_viewport(self.terrain_viewport)
        self.node_editor_tab.set_main_window(self)
        self.node_editor_tab.node_visualized.connect(self.on_node_visualized)

    def on_node_executed(self, node):
        """Handle node execution."""
        # Switch to terrain view tab to see results if it's a visualization node
        from .nodes import VisualizationNode
        if isinstance(node, VisualizationNode):
            self.tab_widget.setCurrentIndex(0)  # Switch to Terrain View tab
    
    def on_node_visualized(self, node_name):
        """Handle node visualization."""
        # Update status
        self.status_label.setText(f"Visualized: {node_name}")

    def _normalize_to_uint8(self, arr: np.ndarray, land_mask: np.ndarray = None) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if land_mask is not None:
            arr = np.where(land_mask, arr, 0.0)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax > vmin:
            norm = (arr - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr, dtype=np.float32)
        return np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)

    def _gray_to_rgba(self, gray_u8: np.ndarray) -> np.ndarray:
        h, w = gray_u8.shape
        a = np.full((h, w), 255, dtype=np.uint8)
        return np.stack([gray_u8, gray_u8, gray_u8, a], axis=-1)

    def _build_generated_texture_overlays(self, td: TerrainData) -> dict:
        """Return a dict[name]->RGBA uint8 for the core simulation textures."""
        hmap = np.ascontiguousarray(td.heightmap, dtype=np.float32)
        land = np.ascontiguousarray(td.land_mask, dtype=np.bool_)
        flow = np.ascontiguousarray(td.river_volume, dtype=np.float32) if getattr(td, 'river_volume', None) is not None else None
        wshd = np.ascontiguousarray(td.watershed_mask, dtype=np.int32) if getattr(td, 'watershed_mask', None) is not None else None
        depo = np.ascontiguousarray(td.deposition_map, dtype=np.float32) if getattr(td, 'deposition_map', None) is not None else None
        rock = np.ascontiguousarray(td.rock_map, dtype=np.int32) if getattr(td, 'rock_map', None) is not None else None

        H, W = hmap.shape
        overlays: dict[str, np.ndarray] = {}

        # 1) Heightmap (grayscale)
        out = np.zeros((H, W, 4), dtype=np.uint8)
        _gray_to_rgba_norm(hmap, land, out)
        overlays['heightmap'] = out

        # 2) Flow mask (river_volume) – grayscale, land only
        # Flow mask (grayscale)
        if flow is not None and flow.shape == hmap.shape:
            out = np.zeros((H, W, 4), dtype=np.uint8)
            _gray_to_rgba_norm(flow, land, out)
            overlays['flow_mask'] = out

        # 3) Deposition map – grayscale with 0.5 neutral
        if depo is not None and depo.shape == hmap.shape:
            out = np.zeros((H, W, 4), dtype=np.uint8)
            _deposition_to_rgba(depo, land, out)
            overlays['deposition_mask'] = out

        # 4) Watershed mask – categorical (stable palette), land only
        if wshd is not None and wshd.shape == hmap.shape:
            max_lbl = int(np.max(wshd)) if wshd.size else 0
            palette = _build_palette_u8(max_lbl + 1)
            out = np.zeros((H, W, 4), dtype=np.uint8)
            _labels_to_rgba(wshd, land, palette, out)
            overlays['watershed_mask'] = out

        # 5) Rock map – categorical color (honors configured albedo if available), land only
        if rock is not None and rock.shape == hmap.shape:
            max_idx = int(np.max(rock)) if rock.size else 0
            # If user supplied rock colors exist, you can pass them here; for speed we build palette.
            palette = _build_palette_u8(max_idx + 1)
            out = np.zeros((H, W, 4), dtype=np.uint8)
            _labels_to_rgba(rock, land, palette, out)
            overlays['rock_map'] = out

        return overlays

    def _ingest_generated_overlays(self):
        """Merge the generated texture overlays into the computed-overlay model & UI."""
        if not self.current_terrain_data:
            return
        generated = self._build_generated_texture_overlays(self.current_terrain_data)
        if not generated:
            return
        # Merge into in-memory store
        self.computed_overlays.update(generated)
        # Refresh the combo without clobbering selection
        selected = None
        if self.current_overlay_source and self.current_overlay_source[0] == 'computed':
            selected = self.current_overlay_source[1]
        names = sorted(self.computed_overlays.keys())
        self.analysis_panel.set_computed_overlays(names, selected)

    def load_preset_from_file(self):
        """Load terrain and heuristic settings from a preset file."""
        start_dir = self.last_preset_directory or str(self.preset_manager.default_directory())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Preset",
            start_dir,
            "Preset Files (*.json);;All Files (*)"
        )
        if not filename:
            return

        try:
            terrain_state, heuristics_state, _ = self.preset_manager.load_preset(filename)
        except PresetError as exc:
            QMessageBox.critical(self, "Preset Error", str(exc))
            return

        self.control_panel.apply_state(terrain_state)
        self.analysis_panel.apply_state(heuristics_state)
        self.last_preset_directory = os.path.dirname(filename) or self.last_preset_directory
        self.reset_computed_overlays()
        self.status_label.setText(f"Preset loaded: {os.path.basename(filename)}")

    def save_preset_to_file(self):
        """Export the current GUI settings to a preset file."""
        start_dir = self.last_preset_directory or str(self.preset_manager.default_directory())
        suggestion = os.path.join(start_dir, "terrain_preset.json")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            suggestion,
            "Preset Files (*.json);;All Files (*)"
        )
        if not filename:
            return

        terrain_state = self.control_panel.get_state()
        heuristics_state = self.analysis_panel.get_state()

        metadata: Dict[str, Any] = {}
        numeric_controls = terrain_state.get('numeric_controls', {})
        if 'dimension' in numeric_controls:
            try:
                metadata['terrain_dimension'] = int(float(numeric_controls['dimension']))
            except (TypeError, ValueError):
                pass
        if 'seed' in numeric_controls:
            try:
                metadata['seed'] = int(float(numeric_controls['seed']))
            except (TypeError, ValueError):
                pass

        try:
            target_path = self.preset_manager.save_preset(
                filename,
                terrain_state=terrain_state,
                heuristics_state=heuristics_state,
                metadata=metadata,
            )
        except PresetError as exc:
            QMessageBox.critical(self, "Preset Error", str(exc))
            return

        self.last_preset_directory = os.path.dirname(os.fspath(target_path)) or self.last_preset_directory
        self.status_label.setText(f"Preset saved: {os.path.basename(os.fspath(target_path))}")

    def generate_terrain(self):
        """Start terrain generation."""
        if self.generator_thread and self.generator_thread.isRunning():
            return

        # Get parameters from control panel
        params = self.control_panel.get_parameters()
        
        # Update UI
        self.control_panel.set_generation_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start generation thread
        self.generator_thread = TerrainGeneratorThread(params)
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.finished.connect(self.terrain_generated)
        self.generator_thread.error.connect(self.generation_error)
        self.generator_thread.start()
    
    def generate_preview(self):
        """Generate terrain preview."""
        if self.generator_thread and self.generator_thread.isRunning():
            return
        
        # Get parameters from control panel
        params = self.control_panel.get_parameters()
        
        # Update UI
        self.control_panel.set_generation_enabled(False)
        self.control_panel.preview_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating preview...")
        
        # Start preview generation thread
        self.preview_thread = TerrainPreviewThread(params)
        self.preview_thread.progress.connect(self.update_progress)
        self.preview_thread.finished.connect(self.preview_generated)
        self.preview_thread.error.connect(self.preview_error)
        self.preview_thread.start()

    def preview_generated(self, terrain_data: TerrainData):
        """Handle completed preview generation."""
        self.current_terrain_data = terrain_data
        self.terrain_viewport.set_terrain(terrain_data)
        self.reset_computed_overlays()

        if self.current_overlay_source and self.terrain_viewport.renderer.overlay_image is None:
            self.current_overlay_source = None
            self.analysis_panel.reset_overlay_controls()

        if self.analysis_panel.show_overlay_checkbox.isEnabled() and self.analysis_panel.show_overlay_checkbox.isChecked():
            self.terrain_viewport.set_overlay_visible(True)

        # Re-enable ALL controls properly
        self.control_panel.set_generation_enabled(True)  # This re-enables the main generate button
        self.control_panel.set_export_enabled(False, heightmap_only=True)  # Allow heightmap export after preview
        self.progress_bar.setVisible(False)
        self.status_label.setText("Preview generated (no rivers). Generate full terrain for complete features.")

    def preview_error(self, error_msg: str):
        """Handle preview generation error."""
        self.control_panel.set_generation_enabled(True)  # Re-enable main generate button
        self.progress_bar.setVisible(False)
        self.status_label.setText("Preview generation failed!")
        
        QMessageBox.critical(self, "Preview Error", error_msg)

    def terrain_generated(self, terrain_data: TerrainData):
        """Handle completed terrain generation."""
        self.current_terrain_data = terrain_data
        self.terrain_viewport.set_terrain(terrain_data)
        self.reset_computed_overlays()
        self.status_label.setText("Generating overlays...")
        self._ingest_generated_overlays()

        if self.current_overlay_source and self.terrain_viewport.renderer.overlay_image is None:
            self.current_overlay_source = None
            self.analysis_panel.reset_overlay_controls()

        if self.analysis_panel.show_overlay_checkbox.isEnabled() and self.analysis_panel.show_overlay_checkbox.isChecked():
            self.terrain_viewport.set_overlay_visible(True)

        self.control_panel.set_generation_enabled(True)  # Re-enable controls
        self.control_panel.set_export_enabled(True)  # Enable export for full terrain
        self.progress_bar.setVisible(False)
        self.status_label.setText("Terrain generated successfully!")

    def generation_error(self, error_msg: str):
        """Handle generation error."""
        self.control_panel.set_generation_enabled(True)  # Re-enable controls
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation failed!")
        
        QMessageBox.critical(self, "Generation Error", error_msg)
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def update_visualization(self, viz_params: dict):
        """Update visualization settings."""
        if 'color_scheme' in viz_params:
            self.terrain_viewport.set_color_scheme(viz_params['color_scheme'])
        
        if 'height_scale' in viz_params:
            self.terrain_viewport.set_height_scale(viz_params['height_scale'])
        
        if 'sun_altitude' in viz_params:
            self.terrain_viewport.set_sun_altitude(viz_params['sun_altitude'])
        
        if 'show_rivers' in viz_params:
            self.terrain_viewport.set_show_rivers(viz_params['show_rivers'])

        if 'river_threshold' in viz_params:
            self.terrain_viewport.set_river_threshold(viz_params['river_threshold'])

        if 'overlay_visible' in viz_params:
            self.terrain_viewport.set_overlay_visible(viz_params['overlay_visible'])

    def _overlay_label(self):
        """Return a human-readable label for the active overlay."""
        if not self.current_overlay_source:
            return None
        source_type, value = self.current_overlay_source
        if source_type == 'file':
            return value
        if source_type == 'computed':
            return f"Computed: {value}"
        return str(value)

    def _prepare_simulated_flow_override(self) -> Optional[np.ndarray]:
        """Create a flow accumulation override from the simulated rivers."""
        if not self.current_terrain_data:
            return None

        flow = getattr(self.current_terrain_data, 'river_volume', None)
        if flow is None:
            return None

        flow_array = np.asarray(flow, dtype=np.float32)
        if flow_array.shape != self.current_terrain_data.heightmap.shape:
            return None

        land_mask = getattr(self.current_terrain_data, 'land_mask', None)
        if land_mask is not None:
            flow_array = np.where(land_mask, flow_array, 0.0)

        flow_array = np.nan_to_num(flow_array, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.any(flow_array):
            return None

        return flow_array

    def reset_computed_overlays(self):
        """Clear cached heuristic overlays when the heightmap changes."""
        had_computed_overlay = (
            self.current_overlay_source is not None and
            self.current_overlay_source[0] == 'computed'
        )
        self.computed_overlays.clear()
        self.heuristic_arrays.clear()
        self.analysis_panel.clear_computed_overlays()

        if had_computed_overlay:
            self.current_overlay_source = None
            self.terrain_viewport.clear_overlay_image()
            self.analysis_panel.reset_overlay_controls()
            self.status_label.setText("Computed overlays reset due to terrain update.")

    def compute_heuristics(self, request: dict):
        """Start heuristic computation for the current terrain."""
        request_data = dict(request)
        if self.generator_thread and self.generator_thread.isRunning():
            QMessageBox.information(
                self,
                "Terrain Busy",
                "Wait for terrain generation to finish before computing heuristics."
            )
            return

        if self.preview_thread and self.preview_thread.isRunning():
            QMessageBox.information(
                self,
                "Terrain Busy",
                "Wait for the preview to finish before computing heuristics."
            )
            return

        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before computing heuristics.")
            return

        if self.heuristic_thread and self.heuristic_thread.isRunning():
            return

        heightmap = np.asarray(self.current_terrain_data.heightmap, dtype=np.float32)
        heightmap = normalize(heightmap, bounds=(0.0, 1.0))

        deposition_map = None
        if hasattr(self.current_terrain_data, 'deposition_map'):
            deposition_map = self.current_terrain_data.deposition_map

        rock_map = getattr(self.current_terrain_data, 'rock_map', None)
        rock_types = getattr(self.current_terrain_data, 'rock_types', None)
        rock_colors = getattr(self.current_terrain_data, 'rock_albedo', None)

        sanitized_request = {
            'selections': list(request_data.get('selections', [])),
            'settings': dict(request_data.get('settings', {})),
            'cellsize_overrides': dict(request_data.get('cellsize_overrides', {})),
        }

        use_simulated_flow = bool(request_data.get('use_simulated_flow', False))
        flow_override = None
        if use_simulated_flow:
            flow_override = self._prepare_simulated_flow_override()
            if flow_override is None:
                self.status_label.setText("Simulated flow unavailable; using computed flow accumulation.")

        self.analysis_panel.set_heuristics_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Computing heuristics...")

        self.heuristic_thread = HeuristicComputationThread(
            heightmap, 
            sanitized_request, 
            flow_override=flow_override,
            deposition_map=deposition_map,
            rock_map=rock_map,
            rock_types=rock_types,
            rock_colors=rock_colors
        )
        self.heuristic_thread.progress.connect(self.update_progress)
        self.heuristic_thread.finished.connect(self.heuristics_finished)
        self.heuristic_thread.error.connect(self.heuristics_error)
        self.heuristic_thread.start()

    def heuristics_finished(self, overlays: dict, arrays: dict):
        """Handle completion of heuristic computation."""
        self.heuristic_thread = None

        # >>> CHANGED: merge (don’t clobber) so generated textures stay listed
        combined = dict(self.computed_overlays)
        combined.update(overlays)
        self.computed_overlays = combined

        self.heuristic_arrays = arrays
        overlay_names = sorted(self.computed_overlays.keys())

        active_overlay_key = None
        if self.current_overlay_source and self.current_overlay_source[0] == 'computed':
            active_overlay_key = self.current_overlay_source[1]

        self.analysis_panel.set_computed_overlays(overlay_names, active_overlay_key)
        self.analysis_panel.set_heuristics_busy(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        if overlay_names:
            self.status_label.setText("Heuristic maps ready. Choose a map to apply as an overlay.")
        else:
            self.status_label.setText("Heuristic computation completed.")

        if active_overlay_key and active_overlay_key in self.computed_overlays:
            self.apply_computed_overlay(active_overlay_key)

    def heuristics_error(self, error_msg: str):
        """Handle heuristic computation errors."""
        self.heuristic_thread = None
        self.analysis_panel.set_heuristics_busy(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Heuristics computation failed.")
        QMessageBox.critical(self, "Heuristics Error", error_msg)

    def apply_computed_overlay(self, overlay_key: str):
        """Apply a computed heuristic map as the active overlay."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before applying overlays.")
            return

        image = self.computed_overlays.get(overlay_key)
        if image is None:
            QMessageBox.warning(self, "Unavailable", "Compute heuristics to obtain this map.")
            return

        try:
            self.terrain_viewport.set_overlay_image(image)
        except ValueError as exc:
            QMessageBox.warning(self, "Overlay Mismatch", str(exc))
            return

        label = f"Computed: {overlay_key}"
        self.analysis_panel.set_overlay_controls(label, True)
        self.current_overlay_source = ('computed', overlay_key)
        self.terrain_viewport.set_overlay_visible(True)
        self.status_label.setText(f"Applied computed overlay: {overlay_key}")

    def _save_overlay_to_path(self, overlay_array: np.ndarray, filename: str):
        """Write an overlay array to disk using an appropriate pixel format."""
        array = np.asarray(overlay_array)
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] == 3:
            mode = "RGB"
        elif array.ndim == 3 and array.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError("Unsupported overlay format for export")
        Image.fromarray(array.astype(np.uint8), mode=mode).save(filename)

    def export_computed_overlay(self, overlay_key: str):
        """Export a computed heuristic overlay texture to disk."""
        overlay = self.computed_overlays.get(overlay_key)
        if overlay is None:
            QMessageBox.warning(
                self,
                "Unavailable",
                "Compute heuristics to obtain this map before exporting."
            )
            return

        default_name = f"{overlay_key}.png"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Heuristic Map",
            default_name,
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )

        if not filename:
            return

        try:
            self._save_overlay_to_path(overlay, filename)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Saved {overlay_key} to {filename}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def export_all_computed_overlays(self):
        """Export all computed overlays to a directory."""
        if not self.computed_overlays:
            QMessageBox.information(
                self,
                "No Maps",
                "Compute heuristics before exporting maps."
            )
            return

        directory = QFileDialog.getExistingDirectory(self, "Export Heuristic Maps")
        if not directory:
            return

        errors = []
        exported = 0
        for key, overlay in self.computed_overlays.items():
            target_path = os.path.join(directory, f"{key}.png")
            try:
                self._save_overlay_to_path(overlay, target_path)
                exported += 1
            except Exception as exc:
                errors.append((key, str(exc)))

        if errors and exported == 0:
            message = "\n".join(f"{name}: {err}" for name, err in errors)
            QMessageBox.critical(self, "Export Failed", f"Unable to export maps:\n{message}")
        elif errors:
            message = "\n".join(f"{name}: {err}" for name, err in errors)
            QMessageBox.warning(
                self,
                "Partial Export",
                f"Exported {exported} maps, but some failed:\n{message}"
            )
        else:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported {exported} maps to {directory}"
            )

    def load_overlay_texture(self, filepath: str):
        """Load an overlay texture from disk and apply it to the viewport."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before loading an overlay texture.")
            self.analysis_panel.reset_overlay_controls()
            return

        try:
            with Image.open(filepath) as img:
                # Preserve alpha if present, otherwise ensure RGB
                if img.mode not in ("RGBA", "RGB"):
                    img = img.convert("RGBA")
                overlay_array = np.array(img)
        except Exception as exc:
            QMessageBox.critical(self, "Overlay Load Failed", f"Could not load overlay image:\n{exc}")
            if self.current_overlay_source:
                visible = bool(
                    self.terrain_viewport.renderer.overlay_image is not None and
                    self.terrain_viewport.renderer.overlay_enabled
                )
                label = self._overlay_label()
                if label:
                    self.analysis_panel.set_overlay_controls(label, visible)
            else:
                self.analysis_panel.reset_overlay_controls()
            return

        try:
            self.terrain_viewport.set_overlay_image(overlay_array)
        except ValueError as exc:
            QMessageBox.warning(self, "Overlay Mismatch", str(exc))
            if self.current_overlay_source:
                visible = bool(
                    self.terrain_viewport.renderer.overlay_image is not None and
                    self.terrain_viewport.renderer.overlay_enabled
                )
                label = self._overlay_label()
                if label:
                    self.analysis_panel.set_overlay_controls(label, visible)
            else:
                self.analysis_panel.reset_overlay_controls()
            return

        self.current_overlay_source = ('file', filepath)
        self.terrain_viewport.set_overlay_visible(True)
        self.status_label.setText(f"Overlay applied: {filepath}")

    def clear_overlay_texture(self):
        """Remove the current overlay texture from the viewport."""
        self.current_overlay_source = None
        self.terrain_viewport.clear_overlay_image()
        self.analysis_panel.reset_overlay_controls()
        self.status_label.setText("Overlay texture cleared.")
    
    def export_terrain(self):
        """Export terrain heightmap."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data", 
                              "Please generate terrain first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Heightmap", "heightmap.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )
        
        if filename:
            try:
                export_format = self.control_panel.get_export_format()
                exporter = TerrainExporter()
                exporter.export_heightmap(
                    self.current_terrain_data.heightmap,
                    filename,
                    export_format
                )
                
                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))
    
    def export_flow_mask(self):
        """Export river flow mask."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data",
                              "Please generate terrain first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Flow Mask", "flow_mask.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )
        
        if filename:
            try:
                export_format = self.control_panel.get_flow_export_format()
                exporter = TerrainExporter()
                exporter.export_flow_mask(
                    self.current_terrain_data.river_volume,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )
                
                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def export_watershed_mask(self):
        """Export watershed mask texture."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data",
                              "Please generate terrain first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Watershed Mask", "watershed_mask.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )

        if filename:
            try:
                export_format = self.control_panel.get_watershed_export_format()
                exporter = TerrainExporter()
                exporter.export_watershed_mask(
                    self.current_terrain_data.watershed_mask,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )

                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))
                
    def export_deposition_mask(self):
        """Export deposition mask."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data",
                              "Please generate terrain first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Deposition Mask", "deposition_mask.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )

        if filename:
            try:
                export_format = self.control_panel.get_deposition_export_format()
                exporter = TerrainExporter()
                exporter.export_deposition_mask(
                    self.current_terrain_data.deposition_map,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )

                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def export_rock_map(self):
        """Export a colour-coded rock map texture."""
        if not self.current_terrain_data or self.current_terrain_data.rock_map is None:
            QMessageBox.warning(self, "No Data",
                              "Generate terrain with rock layers before exporting.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Rock Map", "rock_map.png",
            "PNG Files (*.png)"
        )

        if filename:
            try:
                export_format = self.control_panel.get_rock_export_format()
                exporter = TerrainExporter()
                exporter.export_rock_map(
                    self.current_terrain_data.rock_map,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )
                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))
