"""Custom widgets for terrain nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..curves_widget import CurvesGraphWidget, DEFAULT_LINEAR_CURVE, parse_curve_points, serialize_curve_points


class FilePathNodeWidget(NodeBaseWidget):
    """Node widget that stores a file path and exposes a browse button."""

    def __init__(
        self,
        parent=None,
        name: str = "",
        label: str = "",
        *,
        text: str = "",
        placeholder_text: str = "",
        dialog_caption: str = "Select File",
        file_filter: str = "All Files (*)",
        directory: Optional[str] = None,
        button_text: str = "Browse...",
    ):
        super().__init__(parent, name, label)
        self._dialog_caption = dialog_caption
        self._file_filter = file_filter
        self._directory = directory or ""

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._path_edit = QLineEdit()
        self._path_edit.setText(text)
        self._path_edit.setPlaceholderText(placeholder_text)
        self._path_edit.editingFinished.connect(self._on_editing_finished)
        layout.addWidget(self._path_edit, stretch=1)

        self._browse_button = QPushButton(button_text)
        self._browse_button.setFixedWidth(72)
        self._browse_button.clicked.connect(self._browse_for_file)
        layout.addWidget(self._browse_button)

        self.set_custom_widget(container)
        self.widget().setMaximumWidth(220)
        self._sync_tooltips()

    def _default_directory(self) -> str:
        current_text = self.get_value()
        if current_text:
            current_path = Path(current_text).expanduser()
            if current_path.is_dir():
                return str(current_path)
            if current_path.parent != Path():
                return str(current_path.parent)
        return self._directory

    def _sync_tooltips(self):
        path = self.get_value()
        tooltip = path or "No file selected."
        self._path_edit.setToolTip(tooltip)
        self._browse_button.setToolTip(self._dialog_caption)

    def _on_editing_finished(self):
        self._sync_tooltips()
        self.on_value_changed()

    def _browse_for_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self._path_edit,
            self._dialog_caption,
            self._default_directory(),
            self._file_filter,
        )
        if not filename:
            return
        self.set_value(filename)

    def get_value(self):
        return self._path_edit.text().strip()

    def set_value(self, text=""):
        value = str(text or "").strip()
        if value == self.get_value():
            self._sync_tooltips()
            return
        self._path_edit.setText(value)
        self._sync_tooltips()
        self.on_value_changed()


class CurveEditorNodeWidget(NodeBaseWidget):
    """Compact interactive curve editor for node properties."""

    def __init__(
        self,
        parent=None,
        name: str = "",
        label: str = "",
        *,
        value: str = "",
    ):
        super().__init__(parent, name, label)
        self._updating = False

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._curve_editor = CurvesGraphWidget()
        self._curve_editor.setMinimumSize(220, 180)
        self._curve_editor.setMaximumSize(260, 220)
        self._curve_editor.curveChanged.connect(self._on_curve_changed)
        layout.addWidget(self._curve_editor)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_curve)
        controls_layout.addWidget(self._reset_button)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self._hint_label = QLabel("Click to add, drag to adjust, double-click to remove.")
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._hint_label)

        self.set_custom_widget(container)
        self.widget().setMaximumWidth(270)
        self.set_value(value or serialize_curve_points(DEFAULT_LINEAR_CURVE))

    def _on_curve_changed(self, _points):
        if not self._updating:
            self.on_value_changed()

    def _reset_curve(self):
        self.set_value(serialize_curve_points(DEFAULT_LINEAR_CURVE))

    def get_value(self):
        points = self._curve_editor.get_control_points()
        return serialize_curve_points(points)

    def set_value(self, value=""):
        points = parse_curve_points(value, DEFAULT_LINEAR_CURVE, clamp_output=True)
        serialized = serialize_curve_points(points)
        if serialized == self.get_value():
            return
        self._updating = True
        try:
            self._curve_editor.set_control_points(points)
        finally:
            self._updating = False
        self.on_value_changed()
