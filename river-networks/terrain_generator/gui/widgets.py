"""Custom widget components."""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QSlider,
                            QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal

class ParameterControl(QWidget):
    """Single parameter control with slider and numeric input."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, name: str, min_val: float, max_val: float,
                 default: float, step: float = 1.0, decimals: int = 0,
                 parent=None):
        super().__init__(parent)
        self.step = step
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        self.updating = False
        
        self.setup_ui(name, min_val, max_val, default, step, decimals)
    
    def setup_ui(self, name: str, min_val: float, max_val: float,
                 default: float, step: float, decimals: int):
        """Setup the control UI."""
        layout = QHBoxLayout()
        layout.setSpacing(5)
        
        # Label
        self.label = QLabel(f"{name}:")
        self.label.setMinimumWidth(140)
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val / step))
        self.slider.setMaximum(int(max_val / step))
        self.slider.setValue(int(default / step))
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setMinimumWidth(120)
        layout.addWidget(self.slider)
        
        # Numeric input
        if decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
            self.spinbox.setSingleStep(step)
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setSingleStep(int(step))
            # Convert float to int for QSpinBox
            self.spinbox.setMinimum(int(min_val))
            self.spinbox.setMaximum(int(max_val))
        
        self.spinbox.setValue(default if decimals > 0 else int(default))
        self.spinbox.setMinimumWidth(70)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)
        
        self.setLayout(layout)
    
    def on_slider_changed(self, value):
        """Handle slider value change."""
        if not self.updating:
            self.updating = True
            actual_value = value * self.step
            if self.decimals > 0:
                self.spinbox.setValue(actual_value)
            else:
                self.spinbox.setValue(int(actual_value))
            self.valueChanged.emit(actual_value)
            self.updating = False
    
    def on_spinbox_changed(self, value):
        """Handle spinbox value change."""
        if not self.updating:
            self.updating = True
            slider_value = int(value / self.step)
            self.slider.setValue(slider_value)
            self.valueChanged.emit(float(value))
            self.updating = False
    
    def value(self) -> float:
        """Get current value."""
        return float(self.spinbox.value())
    
    def set_value(self, value: float):
        """Set the control value."""
        if self.decimals > 0:
            self.spinbox.setValue(value)
        else:
            self.spinbox.setValue(int(value))


class RockLayerRow(QWidget):
    """Editable row representing a single rock layer entry."""

    moveRequested = pyqtSignal(object, int)
    deleteRequested = pyqtSignal(object)
    browseRequested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.index_label = QLabel("1")
        self.index_label.setFixedWidth(24)
        self.index_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.index_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Layer name")
        layout.addWidget(self.name_edit, stretch=1)

        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setDecimals(3)
        self.thickness_spin.setSingleStep(0.05)
        self.thickness_spin.setMinimum(0.0)
        self.thickness_spin.setMaximum(10.0)
        self.thickness_spin.setValue(0.25)
        self.thickness_spin.setToolTip("Thickness in normalised height units.")
        self.thickness_spin.setFixedWidth(90)
        layout.addWidget(self.thickness_spin)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No erosion parameter file")
        self.path_edit.setReadOnly(True)
        layout.addWidget(self.path_edit, stretch=2)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setFixedWidth(80)
        self.browse_button.clicked.connect(lambda: self.browseRequested.emit(self))
        layout.addWidget(self.browse_button)

        self.up_button = QPushButton("Up")
        self.up_button.setFixedWidth(50)
        self.up_button.clicked.connect(lambda: self.moveRequested.emit(self, -1))
        layout.addWidget(self.up_button)

        self.down_button = QPushButton("Down")
        self.down_button.setFixedWidth(60)
        self.down_button.clicked.connect(lambda: self.moveRequested.emit(self, 1))
        layout.addWidget(self.down_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setFixedWidth(70)
        self.remove_button.clicked.connect(lambda: self.deleteRequested.emit(self))
        layout.addWidget(self.remove_button)

    def set_index(self, index: int):
        self._index = index
        self.index_label.setText(str(index + 1))

    def set_move_enabled(self, *, can_move_up: bool, can_move_down: bool):
        self.up_button.setEnabled(can_move_up)
        self.down_button.setEnabled(can_move_down)

    def set_remove_enabled(self, enabled: bool):
        self.remove_button.setEnabled(enabled)

    def set_path(self, path: str):
        self.path_edit.setText(path or "")
        self.path_edit.setToolTip(path or "No erosion parameter file selected.")

    def get_state(self) -> dict:
        return {
            'name': self.name_edit.text().strip() or f'Layer {self._index + 1}',
            'thickness': float(self.thickness_spin.value()),
            'erosion_params_path': self.path_edit.text().strip() or None,
        }

    def set_state(self, state: dict):
        if not state:
            return
        name = state.get('name')
        if name:
            self.name_edit.setText(str(name))
        thickness = state.get('thickness')
        if thickness is not None:
            try:
                self.thickness_spin.setValue(float(thickness))
            except (TypeError, ValueError):
                pass
        path = state.get('erosion_params_path') or state.get('parameters_path')
        if path:
            self.set_path(str(path))
