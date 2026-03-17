"""Control panel for terrain generation."""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                            QLineEdit, QMessageBox, QTabWidget, QDoubleSpinBox,
                            QScrollArea, QSizePolicy)
from PyQt5.QtCore import pyqtSignal, Qt
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core import TerrainParameters
from ..config import PresetManager, save_erosion_parameters, ErosionParameterSet
from .widgets import ParameterControl, RockLayerRow
from .curves_widget import HeightCurvesWidget


class DomainWarpedFBMWidget(QWidget):
    """Widget for controlling domain-warped FBM parameters."""
    
    parametersChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controls = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the FBM parameter controls."""
        layout = QVBoxLayout(self)
        
        # Main FBM parameters
        fbm_label = QLabel("<b>Main FBM Settings:</b>")
        layout.addWidget(fbm_label)
        
        self.controls['fbm_scale'] = ParameterControl(
            "Frequency Scale", -5.0, 0.0, -2.0, 0.1, 1
        )
        self.controls['fbm_scale'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_scale'])
        
        self.controls['fbm_lower'] = ParameterControl(
            "Lower Bound", 0.0, 10.0, 2.0, 0.1, 1
        )
        self.controls['fbm_lower'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_lower'])
        
        self.controls['fbm_upper'] = ParameterControl(
            "Upper Bound", 0.0, 10.0, 10.0, 0.5, 1
        )
        self.controls['fbm_upper'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_upper'])
        
        layout.addSpacing(10)
        
        # Offset FBM parameters (for domain warping)
        offset_label = QLabel("<b>Domain Warping Settings:</b>")
        layout.addWidget(offset_label)
        
        self.controls['offset_scale'] = ParameterControl(
            "Warp Frequency", -5.0, 0.0, -2.0, 0.1, 1
        )
        self.controls['offset_scale'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_scale'])
        
        self.controls['offset_lower'] = ParameterControl(
            "Warp Lower Bound", 0.0, 10.0, 1.5, 0.1, 1
        )
        self.controls['offset_lower'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_lower'])
        
        self.controls['offset_upper'] = ParameterControl(
            "Warp Upper Bound", 0.0, 10.0, 10.0, 0.5, 1
        )
        self.controls['offset_upper'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_upper'])
        
        self.controls['offset_amplitude'] = ParameterControl(
            "Warp Strength", 0.0, 500.0, 150.0, 10.0, 0
        )
        self.controls['offset_amplitude'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_amplitude'])
        
        layout.addSpacing(10)
        
        # Post-processing parameters
        post_label = QLabel("<b>Post-processing:</b>")
        layout.addWidget(post_label)
        
        self.controls['land_threshold'] = ParameterControl(
            "Land Threshold", 0.0, 1.0, 0.3, 0.01, 2
        )
        self.controls['land_threshold'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['land_threshold'])
        
        self.controls['blur_distance'] = ParameterControl(
            "Blur Distance", 0.0, 10.0, 2.5, 0.5, 1
        )
        self.controls['blur_distance'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['blur_distance'])
        
        edge_label = QLabel("<b>Edge Falloff:</b>")
        layout.addWidget(edge_label)

        self.controls['edge_falloff_distance'] = ParameterControl(
            "Falloff Distance (px)", 10.0, 600.0, 200.0, 5.0, 0
        )
        self.controls['edge_falloff_distance'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_falloff_distance'])

        self.controls['edge_falloff_rate'] = ParameterControl(
            "Falloff Rate", 1.0, 10.0, 1.5, 0.5, 1
        )
        self.controls['edge_falloff_rate'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_falloff_rate'])

        self.controls['edge_smoothness'] = ParameterControl(
            "Edge Smoothness", 0.01, 0.5, 0.2, 0.01, 2
        )
        self.controls['edge_smoothness'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_smoothness'])

        # Update the info label
        info_label = QLabel(
            "• Frequency Scale: Controls feature size\n"
            "• Bounds: Filter frequency range\n"
            "• Warp Strength: How much to distort\n"
            "• Land Threshold: Higher = less land\n"
            "• Blur Distance: Smooths terrain\n"
            "• Falloff Distance: How far from edge\n"
            "• Falloff Rate: Steepness of edge transition\n"
            "• Edge Smoothness: Blend sharpness"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def get_values(self) -> dict:
        """Get current values for all parameters."""
        values = {}
        for name, control in self.controls.items():
            value = control.value()
            # Convert upper bounds of 10.0 to infinity for frequency bounds
            if name in ['fbm_upper', 'offset_upper'] and value >= 10.0:
                value = np.inf
            values[name] = value
        return values

    def set_values(self, values: dict):
        """Apply parameter values from a mapping."""
        if not values:
            return
        for name, control in self.controls.items():
            if name not in values:
                continue
            raw_value = values[name]
            if isinstance(raw_value, str):
                lowered = raw_value.lower()
                if lowered in {"inf", "infinity", "+inf", "+infinity"}:
                    raw_value = control.spinbox.maximum()
                elif lowered in {"-inf", "-infinity"}:
                    raw_value = control.spinbox.minimum()
            elif raw_value is None:
                raw_value = control.spinbox.maximum()

            try:
                control.set_value(float(raw_value))
            except (TypeError, ValueError):
                continue

class ControlPanel(QWidget):
    """Main control panel for terrain generation."""

    parametersChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.preset_manager = PresetManager()
        self.controls = {}
        self.rock_layer_rows: List[RockLayerRow] = []
        self.setup_ui()

    @staticmethod
    def _resolve_control_value(control: ParameterControl, raw_value: Any) -> Optional[float]:
        """Convert persisted values into values acceptable by the control."""
        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in {"inf", "infinity"}:
                return float(control.spinbox.maximum())
            try:
                return float(raw_value)
            except ValueError:
                return None
        if raw_value is None:
            return float(control.spinbox.maximum())
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None
    
    def setup_ui(self):
        """Setup the control panel UI."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        # Title
        title = QLabel("Terrain Generation Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        scroll_layout.addWidget(title)

        # Preset management group
        self.create_preset_group(scroll_layout)

        # Heightmap import group
        self.create_import_group(scroll_layout)

        # Basic parameters group
        self.create_basic_group(scroll_layout)

        # Domain-warped FBM group
        self.create_fbm_group(scroll_layout)
        # Height curves adjustment group
        self.create_curves_group(scroll_layout)
        # Max delta curves group
        self.create_max_delta_curves_group(scroll_layout)

        # River parameters group
        self.create_river_group(scroll_layout)

        # Terrain parameters group
        self.create_terrain_group(scroll_layout)

        # Rock layers configuration
        self.create_rock_layers_group(scroll_layout)

        # Erosion parameters group
        self.create_erosion_group(scroll_layout)

        # Export group
        self.create_export_group(scroll_layout)

        # Instructions remain within the scrollable area
        self.add_instructions(scroll_layout)

        scroll_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll_area, stretch=1)

        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(12, 8, 12, 12)
        button_layout.setSpacing(8)

        # Primary action button stays visible outside the scroll area
        self.create_generation_buttons(button_layout)

        root_layout.addWidget(button_container)

    def create_preset_group(self, parent_layout):
        """Create import/export preset controls."""
        group = QGroupBox("Presets")
        layout = QVBoxLayout()

        description = QLabel(
            "Import or export all terrain and heuristic settings as JSON presets."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(description)

        button_row = QHBoxLayout()
        self.load_preset_button = QPushButton("Import Preset…")
        button_row.addWidget(self.load_preset_button)

        self.save_preset_button = QPushButton("Export Preset…")
        button_row.addWidget(self.save_preset_button)

        layout.addLayout(button_row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_import_group(self, parent_layout):
        """Create heightmap import group."""
        group = QGroupBox("Heightmap Import")
        layout = QVBoxLayout()
        
        # Enable import checkbox
        self.use_import_checkbox = QCheckBox("Use Imported Heightmap")
        self.use_import_checkbox.stateChanged.connect(self.toggle_import_mode)
        layout.addWidget(self.use_import_checkbox)
        
        # File selection
        file_layout = QHBoxLayout()
        self.heightmap_path_edit = QLineEdit()
        self.heightmap_path_edit.setPlaceholderText("No file selected")
        self.heightmap_path_edit.setEnabled(False)
        file_layout.addWidget(self.heightmap_path_edit)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self.browse_heightmap)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        
        # Blend factor control
        self.blend_control = ParameterControl(
            "Blend Factor", 0.0, 1.0, 1.0, step=0.01, decimals=2
        )
        self.blend_control.setEnabled(False)
        self.controls['heightmap_blend_factor'] = self.blend_control
        layout.addWidget(self.blend_control)
        
        # Instructions
        import_info = QLabel(
            "Import a grayscale image:\n"
            "• Black pixels = water\n"
            "• Gray/white pixels = land (lighter = higher)\n"
            "• Image will be resized to match dimension\n"
            "• Blend Factor: 0 = procedural, 1 = imported"
        )
        import_info.setWordWrap(True)
        import_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(import_info)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_basic_group(self, parent_layout):
        """Create basic parameters group."""
        group = QGroupBox("Basic Parameters")
        layout = QVBoxLayout()
        
        # Dimension control
        dim_control = ParameterControl(
            "Dimension", 64, 8192, 1024, step=64, decimals=0
        )
        self.controls['dimension'] = dim_control
        layout.addWidget(dim_control)
        
        # Seed control
        seed_control = ParameterControl(
            "Random Seed", 0, 9999, 42, step=1, decimals=0
        )
        self.controls['seed'] = seed_control
        layout.addWidget(seed_control)
        
        # Disc radius control
        disc_control = ParameterControl(
            "Point Spacing", 0.25, 8.0, 2.0, step=0.05, decimals=2
        )
        self.controls['disc_radius'] = disc_control
        layout.addWidget(disc_control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_fbm_group(self, parent_layout):
        """Create domain-warped FBM group."""
        group = QGroupBox("Terrain Generation")
        layout = QVBoxLayout()
        
        # Add the domain-warped FBM widget
        self.fbm_widget = DomainWarpedFBMWidget()
        self.fbm_widget.parametersChanged.connect(lambda: None)  # Connect if needed
        layout.addWidget(self.fbm_widget)
        
        # Preview checkbox (button lives near the main generate action)
        preview_layout = QHBoxLayout()
        self.preview_checkbox = QCheckBox("Quick Preview (no rivers)")
        self.preview_checkbox.stateChanged.connect(self.toggle_preview_mode)
        preview_layout.addWidget(self.preview_checkbox)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)
        
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_curves_group(self, parent_layout):
        """Create height curves adjustment group."""
        group = QGroupBox("Height Curves Adjustment")
        layout = QVBoxLayout()

        # Enable curves checkbox
        self.use_curves_checkbox = QCheckBox("Enable Height Curves")
        self.use_curves_checkbox.setChecked(False)
        self.use_curves_checkbox.stateChanged.connect(self.toggle_curves)
        layout.addWidget(self.use_curves_checkbox)
        
        # Curves widget
        self.curves_widget = HeightCurvesWidget()
        self.curves_widget.curvesChanged.connect(lambda: self.parametersChanged.emit() if hasattr(self, 'parametersChanged') else None)
        self.curves_widget.setEnabled(False)
        layout.addWidget(self.curves_widget)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_max_delta_curves_group(self, parent_layout):
        """Create max delta curves adjustment group."""
        group = QGroupBox("Max Height Delta Modulation")
        layout = QVBoxLayout()

        self.use_max_delta_curves_checkbox = QCheckBox("Enable Max Delta Curves")
        self.use_max_delta_curves_checkbox.stateChanged.connect(self.toggle_max_delta_curves)
        layout.addWidget(self.use_max_delta_curves_checkbox)

        self.max_delta_curves_widget = HeightCurvesWidget(
            title_text="Max Delta vs Elevation",
            default_curve=[(0.0, 1.0), (2.0, 2.0)]
        )
        self.max_delta_curves_widget.curvesChanged.connect(
            lambda: self.parametersChanged.emit() if hasattr(self, 'parametersChanged') else None
        )
        self.max_delta_curves_widget.setEnabled(False)
        layout.addWidget(self.max_delta_curves_widget)

        info = QLabel(
            "Define how erosion steepness varies with elevation."
        )
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def toggle_curves(self, state):
        """Toggle curves adjustment."""
        enabled = (state == 2)
        self.curves_widget.setEnabled(enabled)
        self.parametersChanged.emit()

    def toggle_max_delta_curves(self, state):
        """Toggle max delta curves adjustment."""
        enabled = (state == 2)
        self.max_delta_curves_widget.setEnabled(enabled)
        self.parametersChanged.emit()

    def create_river_group(self, parent_layout):
        """Create river parameters group."""
        group = QGroupBox("River Parameters")
        layout = QVBoxLayout()
        
        controls_data = [
            ('river_downcutting', "River Downcutting", 0.1, 3.0, 1.7, 0.1, 1),
            ('default_water_level', "Default Water Level", 0.1, 5.0, 0.8, 0.1, 1),
            ('evaporation_rate', "Evaporation Rate", 0.0, 0.5, 0.3, 0.01, 2),
            ('directional_inertia', "River Straightness", 0.0, 1.0, 0.2, 0.01, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in controls_data:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            self.controls[name] = control
            layout.addWidget(control)
        
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_terrain_group(self, parent_layout):
        """Create terrain parameters group."""
        group = QGroupBox("Terrain Parameters")
        layout = QVBoxLayout()
        
        # Max delta control
        max_delta_control = ParameterControl(
            "Base Max Height Delta", 0.01, 0.2, 0.05, 0.01, 2
        )
        self.controls['max_delta'] = max_delta_control
        layout.addWidget(max_delta_control)
        
        # Variable max delta checkbox
        self.variable_max_delta_checkbox = QCheckBox("Enable Terrace Effects")
        self.variable_max_delta_checkbox.stateChanged.connect(
            self.toggle_variable_max_delta
        )
        layout.addWidget(self.variable_max_delta_checkbox)
        
        # Terrace controls (initially hidden)
        self.variable_controls = []
        
        # Basic terrace parameters
        terrace_label = QLabel("<b>Terrace Configuration:</b>")
        terrace_label.setVisible(False)
        self.variable_controls.append(terrace_label)
        layout.addWidget(terrace_label)
        
        controls_data = [
            ('terrace_count', "Number of Terraces", 2, 10, 3, 1, 0),
            ('terrace_thickness', "Terrace Flatness", 0.1, 0.9, 0.5, 0.05, 2),
            ('terrace_flat_delta', "Flat Area Slope", 0.001, 0.02, 0.01, 0.001, 3),
            ('terrace_steep_delta', "Step Slope", 0.05, 0.3, 0.07, 0.01, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in controls_data:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.setVisible(False)
            self.controls[name] = control
            self.variable_controls.append(control)
            layout.addWidget(control)
        
        # Terrace strength modulation
        strength_label = QLabel("<b>Terrace Strength Modulation:</b>")
        strength_label.setVisible(False)
        self.variable_controls.append(strength_label)
        layout.addWidget(strength_label)
        
        strength_controls = [
            ('terrace_min_strength', "Minimum Strength", 0.0, 1.0, 0.1, 0.05, 2),
            ('terrace_max_strength', "Maximum Strength", 0.0, 1.0, 0.8, 0.05, 2),
            ('terrace_strength_scale', "Noise Scale", -4.0, 0.0, -1.7, 0.1, 1),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in strength_controls:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.setVisible(False)
            self.controls[name] = control
            self.variable_controls.append(control)
            layout.addWidget(control)
        
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_rock_layers_group(self, parent_layout):
        """Create rock layer configuration group."""
        group = QGroupBox("Rock Layers")
        layout = QVBoxLayout()

        description = QLabel(
            "Layers are processed from top to bottom. Thickness uses normalised heights."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(description)

        warp_label = QLabel("<b>Rock Stack Warp:</b>")
        layout.addWidget(warp_label)

        strength_control = ParameterControl("Warp Strength", 0.0, 1.0, 0.0, 0.01, 2)
        self.controls['rock_warp_strength'] = strength_control
        layout.addWidget(strength_control)

        scale_control = ParameterControl("FBM Scale", -5.0, 0.0, -2.0, 0.1, 1)
        self.controls['rock_warp_scale'] = scale_control
        layout.addWidget(scale_control)

        lower_control = ParameterControl("FBM Lower Bound", 0.0, 10.0, 1.0, 0.1, 1)
        self.controls['rock_warp_lower'] = lower_control
        layout.addWidget(lower_control)

        upper_control = ParameterControl("FBM Upper Bound", 0.0, 10.0, 10.0, 0.5, 1)
        self.controls['rock_warp_upper'] = upper_control
        layout.addWidget(upper_control)

        warp_info = QLabel(
            "Strength shifts the rock stack up/down; bounds clamp participating frequencies."
        )
        warp_info.setWordWrap(True)
        warp_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(warp_info)

        self.rock_layers_widget = QWidget()
        self.rock_layers_layout = QVBoxLayout(self.rock_layers_widget)
        self.rock_layers_layout.setContentsMargins(0, 0, 0, 0)
        self.rock_layers_layout.setSpacing(4)
        layout.addWidget(self.rock_layers_widget)

        controls_row = QHBoxLayout()
        self.add_layer_button = QPushButton("Add Layer (Top)")
        self.add_layer_button.clicked.connect(lambda: self.add_rock_layer(position=0))
        controls_row.addWidget(self.add_layer_button)

        self.append_layer_button = QPushButton("Add Layer (Bottom)")
        self.append_layer_button.clicked.connect(lambda: self.add_rock_layer(position=len(self.rock_layer_rows)))
        controls_row.addWidget(self.append_layer_button)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

        if not self.rock_layer_rows:
            self.add_rock_layer(position=0)

    def add_rock_layer(self, position: Optional[int] = None, state: Optional[Dict[str, Any]] = None):
        """Insert a new rock layer entry."""
        row = RockLayerRow()
        if state:
            row.set_state(state)
        row.moveRequested.connect(self.on_move_rock_layer)
        row.deleteRequested.connect(self.on_remove_rock_layer)
        row.browseRequested.connect(self.on_browse_rock_parameters)

        if position is None or position < 0 or position > len(self.rock_layer_rows):
            position = 0

        self.rock_layer_rows.insert(position, row)
        self.rock_layers_layout.insertWidget(position, row)
        self.update_rock_layer_indices()

    def on_move_rock_layer(self, row: RockLayerRow, direction: int):
        """Move a rock layer row up or down."""
        if row not in self.rock_layer_rows:
            return
        index = self.rock_layer_rows.index(row)
        new_index = index - 1 if direction < 0 else index + 1
        if new_index < 0 or new_index >= len(self.rock_layer_rows):
            return
        self.rock_layer_rows.pop(index)
        self.rock_layer_rows.insert(new_index, row)
        self.rock_layers_layout.removeWidget(row)
        self.rock_layers_layout.insertWidget(new_index, row)
        self.update_rock_layer_indices()

    def on_remove_rock_layer(self, row: RockLayerRow):
        """Remove a rock layer row."""
        if len(self.rock_layer_rows) <= 1:
            QMessageBox.information(
                self,
                "Cannot Remove Layer",
                "At least one rock layer must remain."
            )
            return
        if row in self.rock_layer_rows:
            self.rock_layer_rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self.update_rock_layer_indices()

    def on_browse_rock_parameters(self, row: RockLayerRow):
        """Select an erosion parameter file for a rock layer."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Erosion Parameter File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            row.set_path(filename)

    def update_rock_layer_indices(self):
        """Update numbering and control states for rock layers."""
        total = len(self.rock_layer_rows)
        for idx, row in enumerate(self.rock_layer_rows):
            row.set_index(idx)
            row.set_move_enabled(can_move_up=(idx > 0), can_move_down=(idx < total - 1))
            row.set_remove_enabled(total > 1)

    def collect_rock_layer_states(self) -> List[Dict[str, Any]]:
        """Return the current rock layer configuration from lowest to highest."""
        return [row.get_state() for row in reversed(self.rock_layer_rows)]

    def apply_rock_layer_states(self, layers: Optional[List[Dict[str, Any]]]):
        """Restore rock layer rows from stored state."""
        for row in list(self.rock_layer_rows):
            row.setParent(None)
            row.deleteLater()
        self.rock_layer_rows.clear()

        if not layers:
            self.add_rock_layer(position=0)
            return

        for state in reversed(layers):
            self.add_rock_layer(position=len(self.rock_layer_rows), state=state)

    def create_erosion_group(self, parent_layout):
        """Create erosion parameters group."""
        group = QGroupBox("Particle Erosion")
        layout = QVBoxLayout()
        
        # Enable erosion checkbox
        self.use_erosion_checkbox = QCheckBox("Enable Particle Erosion")
        self.use_erosion_checkbox.setChecked(True)
        self.use_erosion_checkbox.stateChanged.connect(self.toggle_erosion)
        layout.addWidget(self.use_erosion_checkbox)
        
        # Erosion controls
        self.erosion_controls = []
        
        # Basic erosion parameters
        basic_label = QLabel("<b>Basic Settings:</b>")
        self.erosion_controls.append(basic_label)
        layout.addWidget(basic_label)
        
        controls_data = [
            ('erosion_iterations', "Droplet Count", 10000, 200000, 80000, 5000, 0),
            ('erosion_inertia', "Flow Inertia", 0.0, 0.7, 0.3, 0.05, 2),
            ('erosion_step_size', "Step Size", 0.1, 1.0, 0.3, 0.05, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in controls_data:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            self.controls[name] = control
            self.erosion_controls.append(control)
            layout.addWidget(control)
        
        # Erosion/Deposition rates
        rates_label = QLabel("<b>Erosion & Deposition:</b>")
        self.erosion_controls.append(rates_label)
        layout.addWidget(rates_label)
        
        rate_controls = [
            ('erosion_capacity', "Carrying Capacity", 1.0, 20.0, 8.0, 0.5, 1),
            ('erosion_rate', "Erosion Rate", 0.1, 1.0, 0.4, 0.05, 2),
            ('erosion_deposition_rate', "Deposition Rate", 0.1, 1.0, 0.2, 0.05, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in rate_controls:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            self.controls[name] = control
            self.erosion_controls.append(control)
            layout.addWidget(control)
        
        # Physics parameters
        physics_label = QLabel("<b>Physics Settings:</b>")
        self.erosion_controls.append(physics_label)
        layout.addWidget(physics_label)
        
        physics_controls = [
            ('erosion_gravity', "Gravity Strength", 1.0, 20.0, 10.0, 0.5, 1),
            ('erosion_evaporation', "Water Retention", 0.9, 0.999, 0.98, 0.001, 3),
            ('erosion_max_lifetime', "Droplet Lifetime", 20, 100, 60, 5, 0),
            ('erosion_blur_iterations', "Smoothing Passes", 0, 3, 1, 1, 0),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in physics_controls:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            self.controls[name] = control
            self.erosion_controls.append(control)
            layout.addWidget(control)
        
        # Info label
        info = QLabel(
            "• Droplet Count: More droplets = smoother but slower\n"
            "• Flow Inertia: Higher = straighter channels\n"
            "• Carrying Capacity: Amount of sediment carried\n"
            "• Water Retention: 0.98 = 2% evaporation per step"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        self.erosion_controls.append(info)
        layout.addWidget(info)

        self.export_erosion_button = QPushButton("Export Erosion Parameters...")
        self.export_erosion_button.clicked.connect(self.export_erosion_parameters)
        layout.addWidget(self.export_erosion_button)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def toggle_erosion(self, state):
        """Toggle erosion controls."""
        enabled = (state == 2)  # Qt.Checked
        for control in self.erosion_controls:
            control.setEnabled(enabled)

    def build_erosion_parameter_snapshot(self) -> Dict[str, Any]:
        """Collect the current erosion-related parameters."""
        snapshot: Dict[str, Any] = {}
        snapshot['river_downcutting'] = self.controls['river_downcutting'].value()
        snapshot['max_delta'] = self.controls['max_delta'].value()
        snapshot['erosion_iterations'] = int(self.controls['erosion_iterations'].value())
        snapshot['erosion_inertia'] = self.controls['erosion_inertia'].value()
        snapshot['erosion_capacity'] = self.controls['erosion_capacity'].value()
        snapshot['erosion_deposition_rate'] = self.controls['erosion_deposition_rate'].value()
        snapshot['erosion_rate'] = self.controls['erosion_rate'].value()
        snapshot['erosion_evaporation'] = self.controls['erosion_evaporation'].value()
        snapshot['erosion_gravity'] = self.controls['erosion_gravity'].value()
        snapshot['erosion_max_lifetime'] = int(self.controls['erosion_max_lifetime'].value())
        snapshot['erosion_step_size'] = self.controls['erosion_step_size'].value()
        snapshot['erosion_blur_iterations'] = int(self.controls['erosion_blur_iterations'].value())
        return snapshot

    def export_erosion_parameters(self):
        """Export current erosion settings to a JSON file."""
        snapshot = self.build_erosion_parameter_snapshot()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Erosion Parameters",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not filename:
            return

        target_path = Path(filename)
        param_set = ErosionParameterSet.from_defaults(snapshot, name=target_path.stem)
        try:
            saved_path = save_erosion_parameters(target_path, param_set)
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not write file: {exc}")
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Erosion parameters saved to {saved_path}"
        )

    def create_export_group(self, parent_layout):
        """Create export group."""
        group = QGroupBox("Export Options")
        layout = QVBoxLayout()
        
        # Heightmap export
        heightmap_label = QLabel("<b>Heightmap Export</b>")
        layout.addWidget(heightmap_label)
        
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_layout.addWidget(format_label)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        format_layout.addWidget(self.export_format_combo)
        layout.addLayout(format_layout)
        
        self.export_button = QPushButton("Export Heightmap")
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)
        
        # Flow mask export
        layout.addSpacing(10)
        flow_label = QLabel("<b>Flow Mask Export</b>")
        layout.addWidget(flow_label)
        
        flow_format_layout = QHBoxLayout()
        flow_format_label = QLabel("Format:")
        flow_format_layout.addWidget(flow_format_label)
        
        self.export_flow_format_combo = QComboBox()
        self.export_flow_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        flow_format_layout.addWidget(self.export_flow_format_combo)
        layout.addLayout(flow_format_layout)
        
        self.export_flow_button = QPushButton("Export Flow Mask")
        self.export_flow_button.setEnabled(False)
        layout.addWidget(self.export_flow_button)

        # Watershed mask export
        layout.addSpacing(10)
        watershed_label = QLabel("<b>Watershed Mask Export</b>")
        layout.addWidget(watershed_label)

        watershed_format_layout = QHBoxLayout()
        watershed_format_label = QLabel("Format:")
        watershed_format_layout.addWidget(watershed_format_label)

        self.export_watershed_format_combo = QComboBox()
        self.export_watershed_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        watershed_format_layout.addWidget(self.export_watershed_format_combo)
        layout.addLayout(watershed_format_layout)

        self.export_watershed_button = QPushButton("Export Watershed Mask")
        self.export_watershed_button.setEnabled(False)
        layout.addWidget(self.export_watershed_button)
        
        # Deposition mask export (NEW)
        layout.addSpacing(10)
        deposition_label = QLabel("<b>Deposition Mask Export</b>")
        layout.addWidget(deposition_label)
        
        deposition_format_layout = QHBoxLayout()
        deposition_format_label = QLabel("Format:")
        deposition_format_layout.addWidget(deposition_format_label)
        
        self.export_deposition_format_combo = QComboBox()
        self.export_deposition_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        deposition_format_layout.addWidget(self.export_deposition_format_combo)
        layout.addLayout(deposition_format_layout)
        
        self.export_deposition_button = QPushButton("Export Deposition Mask")
        self.export_deposition_button.setEnabled(False)
        layout.addWidget(self.export_deposition_button)

        # Info about deposition
        deposition_info = QLabel(
            "Deposition mask: Bright = deposition, Dark = erosion"
        )
        deposition_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(deposition_info)

        layout.addSpacing(10)
        rock_label = QLabel("<b>Rock Map Export</b>")
        layout.addWidget(rock_label)

        rock_format_layout = QHBoxLayout()
        rock_format_label = QLabel("Format:")
        rock_format_layout.addWidget(rock_format_label)

        self.export_rock_format_combo = QComboBox()
        self.export_rock_format_combo.addItems([
            "PNG (8-bit)",
        ])
        rock_format_layout.addWidget(self.export_rock_format_combo)
        layout.addLayout(rock_format_layout)

        self.export_rock_button = QPushButton("Export Rock Map")
        self.export_rock_button.setEnabled(False)
        layout.addWidget(self.export_rock_button)

        rock_info = QLabel(
            "Each rock type is assigned a unique colour. Non-land cells export as black."
        )
        rock_info.setWordWrap(True)
        rock_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(rock_info)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_generation_buttons(self, parent_layout):
        """Create generation buttons."""
        button_row = QHBoxLayout()

        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.setEnabled(True)
        self.preview_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preview_button.setStyleSheet("""
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        button_row.addWidget(self.preview_button)

        self.generate_button = QPushButton("Generate Terrain")
        self.generate_button.setEnabled(True)
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_row.addWidget(self.generate_button)

        parent_layout.addLayout(button_row)

    def add_instructions(self, parent_layout):
        """Add instructions label."""
        instructions = QLabel(
            "Controls:\n"
            "• Adjust parameters with sliders or type values\n"
            "• Click Generate to create terrain\n"
            "• Left-click and drag to rotate view\n"
            "• Scroll wheel to zoom in/out"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        parent_layout.addWidget(instructions)



    def toggle_import_mode(self, state):
        """Toggle heightmap import mode."""
        import_enabled = (state == 2)  # Qt.Checked
        self.browse_button.setEnabled(import_enabled)
        self.heightmap_path_edit.setEnabled(import_enabled)
        self.blend_control.setEnabled(import_enabled)
    
    def browse_heightmap(self):
        """Open file dialog to select heightmap."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Heightmap Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*.*)"
        )
        
        if filename:
            self.heightmap_path_edit.setText(filename)
    
    def toggle_preview_mode(self, state):
        """Toggle preview mode."""
        preview_enabled = (state == Qt.Checked)

        if preview_enabled:
            self.generate_button.setText("Generate Full Terrain")
        else:
            self.generate_button.setText("Generate Terrain")

    def toggle_variable_max_delta(self, state):
        """Toggle variable max delta controls."""
        show_controls = (state == 2)
        for control in self.variable_controls:
            control.setVisible(show_controls)

    def get_parameters(self) -> TerrainParameters:
        """Get current parameters as TerrainParameters object."""
        heightmap_path = None
        if self.use_import_checkbox.isChecked():
            path_text = self.heightmap_path_edit.text()
            if path_text and path_text != "No file selected":
                heightmap_path = path_text
        
        # Get FBM parameters
        fbm_params = self.fbm_widget.get_values()

        # Get curves data if enabled
        use_height_curves = False
        height_curve_points = None
        if hasattr(self, 'use_curves_checkbox'):
            use_height_curves = self.use_curves_checkbox.isChecked()
            if use_height_curves and hasattr(self, 'curves_widget'):
                height_curve_points = self.curves_widget.get_control_points()

        use_max_delta_curves = False
        max_delta_curve_points = None
        if hasattr(self, 'use_max_delta_curves_checkbox'):
            use_max_delta_curves = self.use_max_delta_curves_checkbox.isChecked()
            if use_max_delta_curves and hasattr(self, 'max_delta_curves_widget'):
                max_delta_curve_points = self.max_delta_curves_widget.get_control_points()

        return TerrainParameters(
            dimension=int(self.controls['dimension'].value()),
            seed=int(self.controls['seed'].value()),
            disc_radius=self.controls['disc_radius'].value(),
            
            # Domain-warped FBM parameters
            fbm_scale=fbm_params['fbm_scale'],
            fbm_lower=fbm_params['fbm_lower'],
            fbm_upper=fbm_params['fbm_upper'],
            offset_scale=fbm_params['offset_scale'],
            offset_lower=fbm_params['offset_lower'],
            offset_upper=fbm_params['offset_upper'],
            offset_amplitude=fbm_params['offset_amplitude'],
            land_threshold=fbm_params['land_threshold'],
            blur_distance=fbm_params['blur_distance'],
            
            # Edge falloff parameters
            edge_falloff_distance=fbm_params.get('edge_falloff_distance', 50.0),
            edge_falloff_rate=fbm_params.get('edge_falloff_rate', 4.0),
            edge_smoothness=fbm_params.get('edge_smoothness', 0.1),
            
            # Height curves adjustment parameters
            use_height_curves=hasattr(self, 'use_curves_checkbox') and self.use_curves_checkbox.isChecked(),
            height_curve_points=height_curve_points,

            # Max delta curves adjustment
            use_max_delta_curves=use_max_delta_curves,
            max_delta_curve_points=max_delta_curve_points,

            # Heightmap import
            use_imported_heightmap=self.use_import_checkbox.isChecked(),
            imported_heightmap_path=heightmap_path,
            heightmap_blend_factor=self.controls['heightmap_blend_factor'].value(),
            
            # River parameters
            river_downcutting=self.controls['river_downcutting'].value(),
            default_water_level=self.controls['default_water_level'].value(),
            evaporation_rate=self.controls['evaporation_rate'].value(),
            directional_inertia=self.controls['directional_inertia'].value(),
            
            # Terrain parameters
            max_delta=self.controls['max_delta'].value(),
            use_variable_max_delta=self.variable_max_delta_checkbox.isChecked(),
            
            # Terrace parameters
            terrace_count=int(self.controls.get('terrace_count', 
                ParameterControl("", 0, 0, 3)).value()),
            terrace_thickness=self.controls.get('terrace_thickness',
                ParameterControl("", 0, 0, 0.5)).value(),
            terrace_flat_delta=self.controls.get('terrace_flat_delta',
                ParameterControl("", 0, 0, 0.02)).value(),
            terrace_steep_delta=self.controls.get('terrace_steep_delta',
                ParameterControl("", 0, 0, 0.12)).value(),
            terrace_strength_scale=self.controls.get('terrace_strength_scale',
                ParameterControl("", 0, 0, -1.0)).value(),
            terrace_min_strength=self.controls.get('terrace_min_strength',
                ParameterControl("", 0, 0, 0.0)).value(),
            terrace_max_strength=self.controls.get('terrace_max_strength',
                ParameterControl("", 0, 0, 0.8)).value(),

            rock_warp_strength=self.controls['rock_warp_strength'].value(),
            rock_warp_scale=self.controls['rock_warp_scale'].value(),
            rock_warp_lower=self.controls['rock_warp_lower'].value(),
            rock_warp_upper=(
                np.inf if self.controls['rock_warp_upper'].value() >=
                self.controls['rock_warp_upper'].spinbox.maximum() else
                self.controls['rock_warp_upper'].value()
            ),

            rock_layers=self.collect_rock_layer_states(),

            # Erosion parameters
            use_erosion=self.use_erosion_checkbox.isChecked(),
            erosion_iterations=int(self.controls.get('erosion_iterations',
                ParameterControl("", 0, 0, 80000)).value()),
            erosion_inertia=self.controls.get('erosion_inertia',
                ParameterControl("", 0, 0, 0.3)).value(),
            erosion_capacity=self.controls.get('erosion_capacity',
                ParameterControl("", 0, 0, 8.0)).value(),
            erosion_deposition_rate=self.controls.get('erosion_deposition_rate',
                ParameterControl("", 0, 0, 0.2)).value(),
            erosion_rate=self.controls.get('erosion_rate',
                ParameterControl("", 0, 0, 0.4)).value(),
            erosion_evaporation=self.controls.get('erosion_evaporation',
                ParameterControl("", 0, 0, 0.98)).value(),
            erosion_gravity=self.controls.get('erosion_gravity',
                ParameterControl("", 0, 0, 10.0)).value(),
            erosion_max_lifetime=int(self.controls.get('erosion_max_lifetime',
                ParameterControl("", 0, 0, 60)).value()),
            erosion_step_size=self.controls.get('erosion_step_size',
                ParameterControl("", 0, 0, 0.3)).value(),
            erosion_blur_iterations=int(self.controls.get('erosion_blur_iterations',
                ParameterControl("", 0, 0, 1)).value()),
        )

    def get_state(self) -> Dict[str, Any]:
        """Serialize all terrain-related GUI settings."""
        numeric_controls = {
            name: control.value()
            for name, control in self.controls.items()
        }

        fbm_values = self.fbm_widget.get_values()

        height_curve_points = None
        if self.use_curves_checkbox.isChecked():
            height_curve_points = [list(pair) for pair in self.curves_widget.get_control_points()]

        max_delta_curve_points = None
        if self.use_max_delta_curves_checkbox.isChecked():
            max_delta_curve_points = [list(pair) for pair in self.max_delta_curves_widget.get_control_points()]

        import_path = self.heightmap_path_edit.text().strip()
        if import_path == "No file selected":
            import_path = ""

        export_formats = {
            'heightmap': self.export_format_combo.currentText(),
            'flow': self.export_flow_format_combo.currentText(),
            'watershed': self.export_watershed_format_combo.currentText(),
            'deposition': self.export_deposition_format_combo.currentText(),
            'rock': self.export_rock_format_combo.currentText(),
        }

        return {
            'numeric_controls': numeric_controls,
            'fbm': fbm_values,
            'use_imported_heightmap': self.use_import_checkbox.isChecked(),
            'imported_heightmap_path': import_path,
            'preview_mode': self.preview_checkbox.isChecked(),
            'use_height_curves': self.use_curves_checkbox.isChecked(),
            'height_curve_points': height_curve_points,
            'use_max_delta_curves': self.use_max_delta_curves_checkbox.isChecked(),
            'max_delta_curve_points': max_delta_curve_points,
            'use_variable_max_delta': self.variable_max_delta_checkbox.isChecked(),
            'use_erosion': self.use_erosion_checkbox.isChecked(),
            'export_formats': export_formats,
            'rock_layers': self.collect_rock_layer_states(),
        }

    def apply_state(self, state: Optional[Dict[str, Any]]):
        """Restore GUI settings from serialized terrain preset data."""
        if not state:
            return

        numeric_controls = state.get('numeric_controls', {})
        for name, value in numeric_controls.items():
            control = self.controls.get(name)
            if control is None:
                continue
            resolved = self._resolve_control_value(control, value)
            if resolved is None:
                continue
            try:
                control.set_value(resolved)
            except Exception:
                continue

        fbm_values = state.get('fbm', {})
        if isinstance(fbm_values, dict):
            sanitized = {}
            for key, value in fbm_values.items():
                if isinstance(value, str):
                    sanitized[key] = value
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    sanitized[key] = "inf" if numeric > 0 else "-inf"
                else:
                    sanitized[key] = numeric
            self.fbm_widget.set_values(sanitized)

        use_import = bool(state.get('use_imported_heightmap', False))
        self.use_import_checkbox.setChecked(use_import)
        self.heightmap_path_edit.setText(state.get('imported_heightmap_path', "") or "")

        self.preview_checkbox.setChecked(bool(state.get('preview_mode', False)))

        use_height_curves = bool(state.get('use_height_curves', False))
        self.use_curves_checkbox.setChecked(use_height_curves)
        if use_height_curves and state.get('height_curve_points'):
            self.curves_widget.set_control_points(state['height_curve_points'])

        use_max_delta_curves = bool(state.get('use_max_delta_curves', False))
        self.use_max_delta_curves_checkbox.setChecked(use_max_delta_curves)
        if use_max_delta_curves and state.get('max_delta_curve_points'):
            self.max_delta_curves_widget.set_control_points(state['max_delta_curve_points'])

        self.variable_max_delta_checkbox.setChecked(bool(state.get('use_variable_max_delta', False)))
        self.use_erosion_checkbox.setChecked(bool(state.get('use_erosion', True)))

        self.apply_rock_layer_states(state.get('rock_layers'))

        export_formats = state.get('export_formats', {})
        if isinstance(export_formats, dict):
            mapping = {
                'heightmap': self.export_format_combo,
                'flow': self.export_flow_format_combo,
                'watershed': self.export_watershed_format_combo,
                'deposition': self.export_deposition_format_combo,
                'rock': self.export_rock_format_combo,
            }
            for key, combo in mapping.items():
                target = export_formats.get(key)
                if not target:
                    continue
                index = combo.findText(str(target))
                if index >= 0:
                    combo.setCurrentIndex(index)
    
    def get_export_format(self) -> str:
        """Get selected export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_format_combo.currentText(), "PNG_8")
    
    def get_flow_export_format(self) -> str:
        """Get selected flow export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_flow_format_combo.currentText(), "PNG_8")

    def get_watershed_export_format(self) -> str:
        """Get selected watershed export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_watershed_format_combo.currentText(), "PNG_8")
    
    def get_deposition_export_format(self) -> str:
        """Get selected deposition export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_deposition_format_combo.currentText(), "PNG_8")

    def get_rock_export_format(self) -> str:
        """Get selected rock map export format."""
        return "PNG_8"

    def set_generation_enabled(self, enabled: bool):
        """Enable/disable generation controls during generation."""
        self.generate_button.setEnabled(enabled)
        if hasattr(self, 'preview_button'):
            self.preview_button.setEnabled(enabled)

        if enabled:
            if hasattr(self, 'preview_checkbox') and self.preview_checkbox.isChecked():
                self.generate_button.setText("Generate Full Terrain")
            else:
                self.generate_button.setText("Generate Terrain")

    def set_export_enabled(self, enabled: bool, *, heightmap_only: bool = False):
        """Enable/disable export controls."""
        allow_heightmap = enabled or heightmap_only
        self.export_button.setEnabled(allow_heightmap)
        self.export_flow_button.setEnabled(enabled)
        self.export_watershed_button.setEnabled(enabled)
        self.export_deposition_button.setEnabled(enabled)
        self.export_rock_button.setEnabled(enabled)

class AnalysisPanel(QWidget):
    """Panel hosting visualization controls and heuristic generation."""

    visualization_changed = pyqtSignal(dict)
    overlay_selected = pyqtSignal(str)
    overlay_cleared = pyqtSignal()
    heuristics_requested = pyqtSignal(dict)
    computed_overlay_requested = pyqtSignal(str)
    export_computed_overlay_requested = pyqtSignal(str)
    export_all_computed_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.visual_controls = {}
        self.heuristics_controls = {}
        self.heuristic_checkboxes = {}
        self.heuristic_cellsizes = {}
        self.setup_ui()

    def setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        self.create_visualization_group(scroll_layout)
        self.create_heuristics_group(scroll_layout)

        scroll_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll_area, stretch=1)

        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(12, 8, 12, 12)
        button_layout.setSpacing(8)

        # Keep heuristic computation button visible outside scrollable region
        if hasattr(self, 'compute_heuristics_button'):
            self.compute_heuristics_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button_layout.addWidget(self.compute_heuristics_button)

        root_layout.addWidget(button_container)

    def create_visualization_group(self, parent_layout):
        """Create visualization group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        color_layout = QHBoxLayout()
        color_label = QLabel("Color Scheme:")
        color_layout.addWidget(color_label)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Terrain", "Grayscale", "Topographic"])
        self.color_combo.currentTextChanged.connect(self.color_scheme_changed)
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        height_control = ParameterControl(
            "Height Scale", 5, 50, 20, 1, 0
        )
        height_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'height_scale': v})
        )
        self.visual_controls['height_scale'] = height_control
        layout.addWidget(height_control)

        sun_control = ParameterControl(
            "Sun Altitude", 0, 90, 45, 1, 0
        )
        sun_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'sun_altitude': v})
        )
        self.visual_controls['sun_altitude'] = sun_control
        layout.addWidget(sun_control)

        self.show_rivers_checkbox = QCheckBox("Show Major Rivers")
        self.show_rivers_checkbox.stateChanged.connect(self.river_display_changed)
        layout.addWidget(self.show_rivers_checkbox)

        river_threshold_control = ParameterControl(
            "River Size Threshold (%)", 80, 99.9, 97.5, 0.1, 1
        )
        river_threshold_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'river_threshold': v})
        )
        river_threshold_control.setEnabled(False)
        self.visual_controls['river_threshold'] = river_threshold_control
        layout.addWidget(river_threshold_control)

        layout.addSpacing(10)

        overlay_label = QLabel("<b>Overlay Texture</b>")
        layout.addWidget(overlay_label)

        self.show_overlay_checkbox = QCheckBox("Show Overlay Texture")
        self.show_overlay_checkbox.setEnabled(False)
        self.show_overlay_checkbox.stateChanged.connect(self.overlay_visibility_changed)
        layout.addWidget(self.show_overlay_checkbox)

        overlay_file_layout = QHBoxLayout()
        self.overlay_path_edit = QLineEdit()
        self.overlay_path_edit.setPlaceholderText("No overlay selected")
        self.overlay_path_edit.setReadOnly(True)
        overlay_file_layout.addWidget(self.overlay_path_edit)

        self.overlay_browse_button = QPushButton("Browse...")
        self.overlay_browse_button.clicked.connect(self.browse_overlay_texture)
        overlay_file_layout.addWidget(self.overlay_browse_button)

        self.overlay_clear_button = QPushButton("Clear")
        self.overlay_clear_button.setEnabled(False)
        self.overlay_clear_button.clicked.connect(self.clear_overlay_texture)
        overlay_file_layout.addWidget(self.overlay_clear_button)

        layout.addLayout(overlay_file_layout)

        computed_layout = QHBoxLayout()
        computed_label = QLabel("Computed map:")
        computed_layout.addWidget(computed_label)

        self.computed_overlay_combo = QComboBox()
        self.computed_overlay_combo.setEnabled(False)
        computed_layout.addWidget(self.computed_overlay_combo)

        self.apply_computed_overlay_button = QPushButton("Apply")
        self.apply_computed_overlay_button.setEnabled(False)
        self.apply_computed_overlay_button.clicked.connect(self.on_apply_computed_overlay)
        computed_layout.addWidget(self.apply_computed_overlay_button)

        self.export_computed_overlay_button = QPushButton("Export")
        self.export_computed_overlay_button.setEnabled(False)
        self.export_computed_overlay_button.clicked.connect(self.on_export_computed_overlay)
        computed_layout.addWidget(self.export_computed_overlay_button)

        self.export_all_computed_button = QPushButton("Export All")
        self.export_all_computed_button.setEnabled(False)
        self.export_all_computed_button.clicked.connect(self.on_export_all_computed_overlays)
        computed_layout.addWidget(self.export_all_computed_button)

        layout.addLayout(computed_layout)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_heuristics_group(self, parent_layout):
        """Create heuristics selection group."""
        group = QGroupBox("Terrain Heuristics")
        layout = QVBoxLayout()

        description = QLabel(
            "Generate diagnostic maps (slope, flow, biomes, etc.) using the current heightmap."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(description)

        heuristics = [
            ("slope", "Slope (deg)"),
            ("aspect", "Aspect (deg)"),
            ("normal", "Surface normals"),
            ("curvature", "Curvature"),
            ("tpi", "Topographic position index"),
            ("flowacc", "Flow accumulation"),
            ("twi", "Topographic wetness"),
            ("svf", "Sky view factor"),
            ("climate", "Climate fields"),
            ("biome", "Biomes"),
            ("albedo", "Terrain albedo"),
            ("foliage", "Foliage color"),
            ("forest_density", "Forest density"),
            ("groundcover_density", "Groundcover density"),
            ("albedo_continuous", "Continuous albedo"),
        ]
        default_checked = {"slope", "aspect", "normal", "curvature", "tpi", "flowacc", "twi", "svf", "climate", "biome", "albedo", "albedo_continuous", "foliage", "forest_density", "groundcover_density"}
        for key, label_text in heuristics:
            row = QHBoxLayout()
            cb = QCheckBox(label_text)
            cb.setChecked(key in default_checked)
            self.heuristic_checkboxes[key] = cb
            row.addWidget(cb)

            row.addStretch()

            size_label = QLabel("Cell size")
            size_label.setStyleSheet("color: #666; font-size: 11px;")
            row.addWidget(size_label)

            override = QDoubleSpinBox()
            override.setRange(0.0, 50000.0)
            override.setDecimals(1)
            override.setSingleStep(50.0)
            override.setSpecialValueText("Default")
            override.setSuffix(" m")
            override.setValue(0.0)
            override.setMaximumWidth(110)
            override.setToolTip("Override the global cell size for this map (0 = default).")
            self.heuristic_cellsizes[key] = override
            row.addWidget(override)

            layout.addLayout(row)

        layout.addSpacing(6)

        params_box = QGroupBox("Computation Settings")
        params_layout = QVBoxLayout()

        cellsize_control = ParameterControl("Cell Size (m/pixel)", 1, 20000, 1500, 10, 0)
        self.heuristics_controls['cellsize'] = cellsize_control
        params_layout.addWidget(cellsize_control)

        z_min_control = ParameterControl("Elevation Minimum (m)", -4000, 4000, 0, 50, 0)
        self.heuristics_controls['z_min'] = z_min_control
        params_layout.addWidget(z_min_control)

        z_max_control = ParameterControl("Elevation Maximum (m)", 100, 10000, 6000, 50, 0)
        self.heuristics_controls['z_max'] = z_max_control
        params_layout.addWidget(z_max_control)

        sea_level_control = ParameterControl("Sea Level (m)", -1000, 4000, 0, 10, 0)
        self.heuristics_controls['sea_level_m'] = sea_level_control
        params_layout.addWidget(sea_level_control)

        lapse_control = ParameterControl("Temp Lapse Rate (°C/km)", 0.0, 15.0, 6.5, 0.1, 1)
        self.heuristics_controls['lapse_rate_c_per_km'] = lapse_control
        params_layout.addWidget(lapse_control)

        teq_control = ParameterControl("Temp @ Equator (°C)", -20.0, 60.0, 30.0, 0.5, 1)
        self.heuristics_controls['t_equator_c'] = teq_control
        params_layout.addWidget(teq_control)

        tpole_control = ParameterControl("Temp @ Poles (°C)", -80.0, 40.0, 0.0, 0.5, 1)
        self.heuristics_controls['t_pole_c'] = tpole_control
        params_layout.addWidget(tpole_control)

        coast_decay_control = ParameterControl("Coast Decay (km)", 0.1, 500.0, 1.75, 0.05, 2)
        self.heuristics_controls['coast_decay_km'] = coast_decay_control
        params_layout.addWidget(coast_decay_control)

        orographic_alpha_control = ParameterControl("Orographic Alpha", 0.0, 10.0, 4.0, 0.1, 1)
        self.heuristics_controls['orographic_alpha'] = orographic_alpha_control
        params_layout.addWidget(orographic_alpha_control)

        shadow_max_distance_control = ParameterControl("Shadow Max Distance (km)", 1.0, 2000.0, 400.0, 5.0, 1)
        self.heuristics_controls['shadow_max_distance_km'] = shadow_max_distance_control
        params_layout.addWidget(shadow_max_distance_control)

        shadow_decay_control = ParameterControl("Shadow Decay (km)", 1.0, 1000.0, 150.0, 5.0, 1)
        self.heuristics_controls['shadow_decay_km'] = shadow_decay_control
        params_layout.addWidget(shadow_decay_control)

        shadow_height_control = ParameterControl("Shadow Height Threshold (m)", 0.0, 5000.0, 150.0, 10.0, 1)
        self.heuristics_controls['shadow_height_threshold_m'] = shadow_height_control
        params_layout.addWidget(shadow_height_control)

        shadow_strength_control = ParameterControl("Shadow Strength", 0.0, 5.0, 1.0, 0.05, 2)
        self.heuristics_controls['shadow_strength'] = shadow_strength_control
        params_layout.addWidget(shadow_strength_control)

        svf_dirs_control = ParameterControl("SVF Directions", 4, 64, 16, 1, 0)
        self.heuristics_controls['svf_dirs'] = svf_dirs_control
        params_layout.addWidget(svf_dirs_control)

        svf_radius_control = ParameterControl("SVF Radius (m)", 10, 1000, 100, 5, 0)
        self.heuristics_controls['svf_radius'] = svf_radius_control
        params_layout.addWidget(svf_radius_control)

        biome_mix_control = ParameterControl("Biome Mixing Radius", 0, 10, 1, 1, 0)
        self.heuristics_controls['biome_mixing'] = biome_mix_control
        params_layout.addWidget(biome_mix_control)

        tpi_label = QLabel("TPI Radii (m, comma separated)")
        params_layout.addWidget(tpi_label)
        self.tpi_radii_edit = QLineEdit("25, 100")
        params_layout.addWidget(self.tpi_radii_edit)

        self.random_biomes_checkbox = QCheckBox("Use probabilistic biomes")
        params_layout.addWidget(self.random_biomes_checkbox)

        self.use_simulated_flow_checkbox = QCheckBox("Use simulated river flow for flow accumulation")
        self.use_simulated_flow_checkbox.setChecked(True)
        params_layout.addWidget(self.use_simulated_flow_checkbox)

        pattern_layout = QHBoxLayout()
        pattern_label = QLabel("Temperature pattern:")
        pattern_layout.addWidget(pattern_label)
        self.temperature_pattern_combo = QComboBox()
        self.temperature_pattern_combo.addItem("Polar", "polar")
        self.temperature_pattern_combo.addItem("Equatorial", "equatorial")
        self.temperature_pattern_combo.addItem("Gradient", "gradient")
        pattern_layout.addWidget(self.temperature_pattern_combo)
        params_layout.addLayout(pattern_layout)

        self.temperature_gradient_control = ParameterControl("Temperature gradient azimuth", 0, 360, 0, 5, 0)
        self.temperature_gradient_control.setEnabled(False)
        params_layout.addWidget(self.temperature_gradient_control)
        self.temperature_pattern_combo.currentIndexChanged.connect(self.update_temperature_gradient_enabled)

        precip_layout = QHBoxLayout()
        precip_label = QLabel("Precipitation pattern:")
        precip_layout.addWidget(precip_label)
        self.precip_pattern_combo = QComboBox()
        self.precip_pattern_combo.addItem("Two tropical bands", "two_bands")
        self.precip_pattern_combo.addItem("Single equatorial band", "single_band")
        self.precip_pattern_combo.addItem("Uniform", "uniform")
        self.precip_pattern_combo.addItem("Latitudinal gradient", "gradient")
        precip_layout.addWidget(self.precip_pattern_combo)
        params_layout.addLayout(precip_layout)

        self.precip_gradient_control = ParameterControl("Precip gradient azimuth", 0, 360, 0, 5, 0)
        self.precip_gradient_control.setEnabled(False)
        params_layout.addWidget(self.precip_gradient_control)
        self.precip_pattern_combo.currentIndexChanged.connect(self.update_precip_gradient_enabled)

        wind_layout = QHBoxLayout()
        wind_label = QLabel("Prevailing wind model:")
        wind_layout.addWidget(wind_label)
        self.prevailing_wind_combo = QComboBox()
        self.prevailing_wind_combo.addItem("Three-cell circulation", "three_cell")
        self.prevailing_wind_combo.addItem("Constant azimuth", "constant")
        wind_layout.addWidget(self.prevailing_wind_combo)
        params_layout.addLayout(wind_layout)

        self.constant_wind_control = ParameterControl("Constant wind azimuth", 0, 360, 25, 5, 0)
        self.constant_wind_control.setEnabled(False)
        params_layout.addWidget(self.constant_wind_control)
        self.prevailing_wind_combo.currentIndexChanged.connect(self.update_wind_angle_enabled)

        params_box.setLayout(params_layout)
        layout.addWidget(params_box)

        self.update_temperature_gradient_enabled()
        self.update_precip_gradient_enabled()
        self.update_wind_angle_enabled()

        self.compute_heuristics_button = QPushButton("Compute Heuristic Maps")
        self.compute_heuristics_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3F51B5;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #303F9F;
            }
            """
        )
        self.compute_heuristics_button.clicked.connect(self.request_heuristics_computation)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def color_scheme_changed(self, text):
        """Handle color scheme change."""
        scheme_map = {
            "Terrain": "terrain",
            "Grayscale": "grayscale",
            "Topographic": "topographic"
        }

        if text in scheme_map:
            self.visualization_changed.emit({'color_scheme': scheme_map[text]})

            is_terrain = (text == "Terrain")
            self.show_rivers_checkbox.setEnabled(is_terrain)
            if not is_terrain:
                self.show_rivers_checkbox.setChecked(False)

    def river_display_changed(self, state):
        """Handle river display toggle."""
        show_rivers = (state == 2)
        self.visual_controls['river_threshold'].setEnabled(show_rivers)
        self.visualization_changed.emit({'show_rivers': show_rivers})

    def overlay_visibility_changed(self, state):
        """Handle overlay visibility toggle."""
        visible = (state == 2)
        self.visualization_changed.emit({'overlay_visible': visible})

    def browse_overlay_texture(self):
        """Browse for an overlay texture image."""
        current_path = self.overlay_path_edit.text()
        start_dir = Path(current_path).parent if current_path else Path.home()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Overlay Texture",
            str(start_dir),
            "PNG Images (*.png);;All Files (*)"
        )

        if filename:
            self.overlay_path_edit.setText(filename)
            self.overlay_clear_button.setEnabled(True)
            self.show_overlay_checkbox.setEnabled(True)
            if not self.show_overlay_checkbox.isChecked():
                self.show_overlay_checkbox.blockSignals(True)
                self.show_overlay_checkbox.setChecked(True)
                self.show_overlay_checkbox.blockSignals(False)
                self.visualization_changed.emit({'overlay_visible': True})
            else:
                self.visualization_changed.emit({'overlay_visible': True})
            self.overlay_selected.emit(filename)

    def clear_overlay_texture(self):
        """Clear the currently selected overlay texture."""
        self.reset_overlay_controls()
        self.overlay_cleared.emit()
        self.visualization_changed.emit({'overlay_visible': False})

    def on_apply_computed_overlay(self):
        """Emit a request to apply the currently selected computed overlay."""
        if not self.computed_overlay_combo.isEnabled():
            return
        overlay_key = self.computed_overlay_combo.currentData()
        if overlay_key:
            self.computed_overlay_requested.emit(str(overlay_key))

    def on_export_computed_overlay(self):
        """Emit a request to export the currently selected computed overlay."""
        if not self.computed_overlay_combo.isEnabled():
            return
        overlay_key = self.computed_overlay_combo.currentData()
        if overlay_key:
            self.export_computed_overlay_requested.emit(str(overlay_key))

    def on_export_all_computed_overlays(self):
        """Emit a request to export all computed overlays."""
        if not self.export_all_computed_button.isEnabled():
            return
        self.export_all_computed_requested.emit()

    def set_computed_overlays(self, names, selected: str = None):
        """Populate the computed overlay combo box with available maps."""
        self.computed_overlay_combo.blockSignals(True)
        self.computed_overlay_combo.clear()
        for name in names:
            display = name.replace('_', ' ')
            self.computed_overlay_combo.addItem(display, name)
        has_items = bool(names)
        self.computed_overlay_combo.setEnabled(has_items)
        self.apply_computed_overlay_button.setEnabled(has_items)
        self.export_computed_overlay_button.setEnabled(has_items)
        self.export_all_computed_button.setEnabled(has_items)
        if has_items:
            target_index = 0
            if selected is not None:
                for idx in range(self.computed_overlay_combo.count()):
                    if self.computed_overlay_combo.itemData(idx) == selected:
                        target_index = idx
                        break
            self.computed_overlay_combo.setCurrentIndex(target_index)
        self.computed_overlay_combo.blockSignals(False)

    def clear_computed_overlays(self):
        """Clear computed overlays from the UI controls."""
        self.computed_overlay_combo.blockSignals(True)
        self.computed_overlay_combo.clear()
        self.computed_overlay_combo.blockSignals(False)
        self.computed_overlay_combo.setEnabled(False)
        self.apply_computed_overlay_button.setEnabled(False)
        self.export_computed_overlay_button.setEnabled(False)
        self.export_all_computed_button.setEnabled(False)

    def reset_overlay_controls(self):
        """Reset overlay controls to default state without emitting signals."""
        self.overlay_path_edit.clear()
        self.overlay_path_edit.setPlaceholderText("No overlay selected")
        self.overlay_clear_button.setEnabled(False)
        self.show_overlay_checkbox.blockSignals(True)
        self.show_overlay_checkbox.setChecked(False)
        self.show_overlay_checkbox.setEnabled(False)
        self.show_overlay_checkbox.blockSignals(False)

    def set_overlay_controls(self, path: str, visible: bool):
        """Programmatically apply overlay selection state without emitting signals."""
        self.overlay_path_edit.setText(path)
        self.overlay_clear_button.setEnabled(True)
        self.show_overlay_checkbox.blockSignals(True)
        self.show_overlay_checkbox.setEnabled(True)
        self.show_overlay_checkbox.setChecked(visible)
        self.show_overlay_checkbox.blockSignals(False)

    def update_temperature_gradient_enabled(self):
        enabled = self.temperature_pattern_combo.currentData() == 'gradient'
        self.temperature_gradient_control.setEnabled(enabled)

    def update_precip_gradient_enabled(self):
        enabled = self.precip_pattern_combo.currentData() == 'gradient'
        self.precip_gradient_control.setEnabled(enabled)

    def update_wind_angle_enabled(self):
        enabled = self.prevailing_wind_combo.currentData() == 'constant'
        self.constant_wind_control.setEnabled(enabled)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: Any):
        if value is None:
            return
        # Try matching stored data first
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        # Fall back to string comparison on displayed text
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def get_state(self) -> Dict[str, Any]:
        """Serialize heuristic-related GUI settings."""
        checkbox_state = {
            key: checkbox.isChecked()
            for key, checkbox in self.heuristic_checkboxes.items()
        }

        cellsizes = {
            key: spin.value()
            for key, spin in self.heuristic_cellsizes.items()
        }

        numeric_controls = {
            key: control.value()
            for key, control in self.heuristics_controls.items()
        }

        return {
            'heuristic_checkboxes': checkbox_state,
            'heuristic_cellsizes': cellsizes,
            'heuristic_controls': numeric_controls,
            'tpi_radii_text': self.tpi_radii_edit.text(),
            'use_random_biomes': self.random_biomes_checkbox.isChecked(),
            'use_simulated_flow': self.use_simulated_flow_checkbox.isChecked(),
            'temperature_pattern': self.temperature_pattern_combo.currentData(),
            'temperature_gradient_azimuth_deg': self.temperature_gradient_control.value(),
            'precip_pattern': self.precip_pattern_combo.currentData(),
            'precip_gradient_azimuth_deg': self.precip_gradient_control.value(),
            'prevailing_wind_model': self.prevailing_wind_combo.currentData(),
            'constant_wind_azimuth_deg': self.constant_wind_control.value(),
        }

    def apply_state(self, state: Optional[Dict[str, Any]]):
        """Restore heuristic-related settings from serialized data."""
        if not state:
            return

        checkbox_state = state.get('heuristic_checkboxes', {})
        for key, checkbox in self.heuristic_checkboxes.items():
            if key not in checkbox_state:
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(checkbox_state[key]))
            checkbox.blockSignals(False)

        cellsizes = state.get('heuristic_cellsizes', {})
        for key, spin in self.heuristic_cellsizes.items():
            if key not in cellsizes:
                continue
            try:
                spin.setValue(float(cellsizes[key]))
            except (TypeError, ValueError):
                continue

        numeric_controls = state.get('heuristic_controls', {})
        for key, control in self.heuristics_controls.items():
            if key not in numeric_controls:
                continue
            resolved = ControlPanel._resolve_control_value(control, numeric_controls[key])
            if resolved is None:
                continue
            try:
                control.set_value(resolved)
            except Exception:
                continue

        if 'tpi_radii_text' in state:
            self.tpi_radii_edit.setText(str(state['tpi_radii_text']))

        self.random_biomes_checkbox.setChecked(bool(state.get('use_random_biomes', False)))
        self.use_simulated_flow_checkbox.setChecked(bool(state.get('use_simulated_flow', True)))

        self._set_combo_value(self.temperature_pattern_combo, state.get('temperature_pattern'))
        self._set_combo_value(self.precip_pattern_combo, state.get('precip_pattern'))
        self._set_combo_value(self.prevailing_wind_combo, state.get('prevailing_wind_model'))

        # Ensure dependent controls update their enabled state before assigning values
        self.update_temperature_gradient_enabled()
        self.update_precip_gradient_enabled()
        self.update_wind_angle_enabled()

        if 'temperature_gradient_azimuth_deg' in state:
            try:
                self.temperature_gradient_control.set_value(float(state['temperature_gradient_azimuth_deg']))
            except (TypeError, ValueError):
                pass

        if 'precip_gradient_azimuth_deg' in state:
            try:
                self.precip_gradient_control.set_value(float(state['precip_gradient_azimuth_deg']))
            except (TypeError, ValueError):
                pass

        if 'constant_wind_azimuth_deg' in state:
            try:
                self.constant_wind_control.set_value(float(state['constant_wind_azimuth_deg']))
            except (TypeError, ValueError):
                pass

    def _parse_tpi_radii(self) -> List[float]:
        text = self.tpi_radii_edit.text().strip()
        if not text:
            return [25.0, 100.0]
        parts = [p for p in text.replace(' ', '').split(',') if p]
        radii: List[float] = []
        for part in parts:
            try:
                value = float(part)
                if value <= 0:
                    raise ValueError
                radii.append(value)
            except ValueError:
                raise ValueError(f"Invalid TPI radius: {part}") from None
        if not radii:
            raise ValueError("Provide at least one TPI radius")
        return radii

    def request_heuristics_computation(self):
        selections = []
        cellsize_overrides = {}
        for key, checkbox in self.heuristic_checkboxes.items():
            if checkbox.isChecked():
                selections.append(key)
                override_widget = self.heuristic_cellsizes.get(key)
                if override_widget is not None:
                    value = override_widget.value()
                    if value > 0.0:
                        cellsize_overrides[key] = value

        if not selections:
            QMessageBox.information(self, "No heuristics selected",
                                    "Select at least one map to compute.")
            return

        try:
            tpi_radii = self._parse_tpi_radii()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid TPI radii", str(exc))
            return

        settings = {
            'cellsize': self.heuristics_controls['cellsize'].value(),
            'z_min': self.heuristics_controls['z_min'].value(),
            'z_max': self.heuristics_controls['z_max'].value(),
            'sea_level_m': self.heuristics_controls['sea_level_m'].value(),
            'lapse_rate_c_per_km': self.heuristics_controls['lapse_rate_c_per_km'].value(),
            't_equator_c': self.heuristics_controls['t_equator_c'].value(),
            't_pole_c': self.heuristics_controls['t_pole_c'].value(),
            'coast_decay_km': self.heuristics_controls['coast_decay_km'].value(),
            'orographic_alpha': self.heuristics_controls['orographic_alpha'].value(),
            'shadow_max_distance_km': self.heuristics_controls['shadow_max_distance_km'].value(),
            'shadow_decay_km': self.heuristics_controls['shadow_decay_km'].value(),
            'shadow_height_threshold_m': self.heuristics_controls['shadow_height_threshold_m'].value(),
            'shadow_strength': self.heuristics_controls['shadow_strength'].value(),
            'svf_dirs': int(self.heuristics_controls['svf_dirs'].value()),
            'svf_radius': self.heuristics_controls['svf_radius'].value(),
            'biome_mixing': int(self.heuristics_controls['biome_mixing'].value()),
            'tpi_radii': tpi_radii,
            'use_random_biomes': self.random_biomes_checkbox.isChecked(),
            'temperature_pattern': self.temperature_pattern_combo.currentData(),
            'temperature_gradient_azimuth_deg': self.temperature_gradient_control.value(),
            'precip_lat_pattern': self.precip_pattern_combo.currentData(),
            'precip_gradient_azimuth_deg': self.precip_gradient_control.value(),
            'prevailing_wind_model': self.prevailing_wind_combo.currentData(),
            'constant_wind_azimuth_deg': self.constant_wind_control.value(),
        }

        expanded = []
        for key in selections:
            if key == 'tpi':
                for radius in tpi_radii:
                    expanded.append(f"tpi@{float(radius)}")
            else:
                expanded.append(key)

        request_payload = {
            'selections': expanded,
            'settings': settings,
            'use_simulated_flow': self.use_simulated_flow_checkbox.isChecked(),
            'cellsize_overrides': cellsize_overrides,
        }
        self.heuristics_requested.emit(request_payload)

    def set_heuristics_busy(self, busy: bool):
        self.compute_heuristics_button.setEnabled(not busy)
        for checkbox in self.heuristic_checkboxes.values():
            checkbox.setEnabled(not busy)
        for control in self.heuristics_controls.values():
            control.setEnabled(not busy)
        for override in self.heuristic_cellsizes.values():
            override.setEnabled(not busy)
        self.tpi_radii_edit.setEnabled(not busy)
        self.random_biomes_checkbox.setEnabled(not busy)
        self.use_simulated_flow_checkbox.setEnabled(not busy)
        self.temperature_pattern_combo.setEnabled(not busy)
        self.precip_pattern_combo.setEnabled(not busy)
        self.prevailing_wind_combo.setEnabled(not busy)
        self.temperature_gradient_control.setEnabled(not busy and self.temperature_pattern_combo.currentData() == 'gradient')
        self.precip_gradient_control.setEnabled(not busy and self.precip_pattern_combo.currentData() == 'gradient')
        self.constant_wind_control.setEnabled(not busy and self.prevailing_wind_combo.currentData() == 'constant')
        has_computed = self.computed_overlay_combo.count() > 0
        self.computed_overlay_combo.setEnabled((not busy) and has_computed)
        self.apply_computed_overlay_button.setEnabled((not busy) and has_computed)
        self.export_computed_overlay_button.setEnabled((not busy) and has_computed)
        self.export_all_computed_button.setEnabled((not busy) and has_computed)
