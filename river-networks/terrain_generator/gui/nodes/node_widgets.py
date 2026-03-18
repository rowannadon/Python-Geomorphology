"""Custom widgets for terrain nodes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..curves_widget import CurvesGraphWidget, DEFAULT_LINEAR_CURVE, parse_curve_points, serialize_curve_points


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_polygon_coord(value: float) -> str:
    text = f"{_clamp_unit(value):.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def regular_polygon_points(count: int) -> list[tuple[float, float]]:
    count = max(3, int(count))
    center_x = 0.5
    center_y = 0.5
    radius = 0.35
    start_angle = -math.pi / 2.0
    return [
        (
            _clamp_unit(center_x + radius * math.cos(start_angle + (2.0 * math.pi * idx / count))),
            _clamp_unit(center_y + radius * math.sin(start_angle + (2.0 * math.pi * idx / count))),
        )
        for idx in range(count)
    ]


def serialize_polygon_points(points: Sequence[tuple[float, float]]) -> str:
    normalized = [(_clamp_unit(x), _clamp_unit(y)) for x, y in points]
    return ", ".join(f"{_format_polygon_coord(x)}:{_format_polygon_coord(y)}" for x, y in normalized)


def _resample_polygon_points(points: Sequence[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    count = max(3, int(count))
    normalized = [(_clamp_unit(x), _clamp_unit(y)) for x, y in points]
    if len(normalized) < 3:
        return regular_polygon_points(count)
    if len(normalized) == count:
        return list(normalized)

    perimeter_lengths: list[float] = []
    total_length = 0.0
    for idx, start in enumerate(normalized):
        end = normalized[(idx + 1) % len(normalized)]
        segment_length = math.dist(start, end)
        perimeter_lengths.append(segment_length)
        total_length += segment_length
    if total_length <= 1e-6:
        return regular_polygon_points(count)

    targets = [total_length * idx / count for idx in range(count)]
    result: list[tuple[float, float]] = []
    segment_index = 0
    traversed = 0.0
    for target in targets:
        while segment_index < len(perimeter_lengths) - 1 and traversed + perimeter_lengths[segment_index] < target:
            traversed += perimeter_lengths[segment_index]
            segment_index += 1
        start = normalized[segment_index]
        end = normalized[(segment_index + 1) % len(normalized)]
        segment_length = perimeter_lengths[segment_index]
        if segment_length <= 1e-6:
            result.append(start)
            continue
        t = (target - traversed) / segment_length
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        result.append((_clamp_unit(x), _clamp_unit(y)))
    return result


DEFAULT_POLYGON_POINTS = regular_polygon_points(5)


def parse_polygon_points(
    value: str,
    fallback: Sequence[tuple[float, float]] = DEFAULT_POLYGON_POINTS,
    *,
    point_count: Optional[int] = None,
) -> list[tuple[float, float]]:
    text = (value or "").strip()
    points: list[tuple[float, float]] = []
    if text:
        for chunk in text.split(","):
            item = chunk.strip()
            if not item or ":" not in item:
                continue
            left, right = item.split(":", 1)
            try:
                points.append((_clamp_unit(float(left)), _clamp_unit(float(right))))
            except ValueError:
                continue
    if not points:
        points = [(_clamp_unit(x), _clamp_unit(y)) for x, y in fallback]
    if point_count is not None:
        points = _resample_polygon_points(points, point_count)
    elif len(points) < 3:
        points = _resample_polygon_points(points, max(3, len(fallback)))
    return points


class PolygonEditorWidget(QWidget):
    """Simple polygon editor with draggable control points."""

    polygonChanged = QtCore.pyqtSignal(list)

    def __init__(self, parent=None, *, point_count: int = 5):
        super().__init__(parent)
        self._points = regular_polygon_points(point_count)
        self._drag_index: Optional[int] = None
        self._handle_radius = 7.0
        self._margin = 14
        self.setMouseTracking(True)
        self.setMinimumSize(220, 180)

    def point_count(self) -> int:
        return len(self._points)

    def get_control_points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def set_control_points(self, points: Sequence[tuple[float, float]], *, emit_signal: bool = False):
        normalized = [(_clamp_unit(x), _clamp_unit(y)) for x, y in points]
        if len(normalized) < 3:
            normalized = regular_polygon_points(self.point_count())
        changed = normalized != self._points
        self._points = list(normalized)
        self.update()
        if emit_signal and changed:
            self.polygonChanged.emit(self.get_control_points())

    def set_point_count(self, count: int, *, emit_signal: bool = False):
        updated = _resample_polygon_points(self._points, count)
        self.set_control_points(updated, emit_signal=emit_signal)

    def reset_polygon(self, *, emit_signal: bool = False):
        self.set_control_points(regular_polygon_points(self.point_count()), emit_signal=emit_signal)

    def _editor_rect(self) -> QtCore.QRectF:
        rect = self.rect().adjusted(self._margin, self._margin, -self._margin, -self._margin)
        return QtCore.QRectF(rect)

    def _point_to_widget(self, point: tuple[float, float]) -> QtCore.QPointF:
        rect = self._editor_rect()
        return QtCore.QPointF(
            rect.left() + point[0] * rect.width(),
            rect.top() + point[1] * rect.height(),
        )

    def _point_from_widget(self, pos: QtCore.QPointF) -> tuple[float, float]:
        rect = self._editor_rect()
        width = max(rect.width(), 1.0)
        height = max(rect.height(), 1.0)
        x = (pos.x() - rect.left()) / width
        y = (pos.y() - rect.top()) / height
        return (_clamp_unit(x), _clamp_unit(y))

    def _nearest_point_index(self, pos: QtCore.QPointF) -> int:
        widget_points = [self._point_to_widget(point) for point in self._points]
        distances = [
            (widget_point.x() - pos.x()) ** 2 + (widget_point.y() - pos.y()) ** 2
            for widget_point in widget_points
        ]
        return min(range(len(distances)), key=distances.__getitem__)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self._editor_rect()

        painter.fillRect(self.rect(), QtGui.QColor(38, 38, 42))
        painter.setPen(QtGui.QPen(QtGui.QColor(72, 72, 78), 1))
        painter.drawRoundedRect(rect, 6, 6)

        grid_pen = QtGui.QPen(QtGui.QColor(58, 58, 64), 1)
        for step in range(1, 4):
            x = rect.left() + rect.width() * step / 4.0
            y = rect.top() + rect.height() * step / 4.0
            painter.setPen(grid_pen)
            painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))

        polygon = QtGui.QPolygonF([self._point_to_widget(point) for point in self._points])
        painter.setBrush(QtGui.QColor(70, 150, 220, 70))
        painter.setPen(QtGui.QPen(QtGui.QColor(135, 205, 255), 2))
        painter.drawPolygon(polygon)

        for idx, point in enumerate(self._points):
            widget_point = self._point_to_widget(point)
            fill = QtGui.QColor(245, 180, 90) if idx == self._drag_index else QtGui.QColor(245, 245, 245)
            painter.setBrush(fill)
            painter.setPen(QtGui.QPen(QtGui.QColor(25, 25, 25), 1.5))
            painter.drawEllipse(widget_point, self._handle_radius, self._handle_radius)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_index = self._nearest_point_index(event.localPos())
        self._move_dragged_point(event.localPos(), emit_signal=True)

    def mouseMoveEvent(self, event):
        if self._drag_index is not None:
            self._move_dragged_point(event.localPos(), emit_signal=True)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_index = None
            self.update()
            return
        super().mouseReleaseEvent(event)

    def _move_dragged_point(self, pos: QtCore.QPointF, *, emit_signal: bool):
        if self._drag_index is None:
            return
        updated = list(self._points)
        updated[self._drag_index] = self._point_from_widget(pos)
        self.set_control_points(updated, emit_signal=emit_signal)


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


class PolygonEditorNodeWidget(NodeBaseWidget):
    """Compact polygon editor for node properties."""

    def __init__(
        self,
        parent=None,
        name: str = "",
        label: str = "",
        *,
        value: str = "",
        point_count: int = 5,
    ):
        super().__init__(parent, name, label)
        self._updating = False

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._polygon_editor = PolygonEditorWidget(point_count=max(3, int(point_count)))
        self._polygon_editor.polygonChanged.connect(self._on_polygon_changed)
        layout.addWidget(self._polygon_editor)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_polygon)
        controls_layout.addWidget(self._reset_button)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self._hint_label = QLabel("Drag points to sketch the polygon. Point Count reshapes the loop.")
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._hint_label)

        self.set_custom_widget(container)
        self.widget().setMaximumWidth(270)
        initial_value = value or serialize_polygon_points(regular_polygon_points(max(3, int(point_count))))
        self.set_value(initial_value)

    def _on_polygon_changed(self, _points):
        if not self._updating:
            self.on_value_changed()

    def _reset_polygon(self):
        self._updating = True
        try:
            self._polygon_editor.reset_polygon()
        finally:
            self._updating = False
        self.on_value_changed()

    def get_value(self):
        points = self._polygon_editor.get_control_points()
        return serialize_polygon_points(points)

    def set_value(self, value=""):
        fallback = regular_polygon_points(self._polygon_editor.point_count())
        points = parse_polygon_points(value, fallback, point_count=self._polygon_editor.point_count())
        serialized = serialize_polygon_points(points)
        if serialized == self.get_value():
            return
        self._updating = True
        try:
            self._polygon_editor.set_control_points(points)
        finally:
            self._updating = False
        self.on_value_changed()

    def set_point_count(self, point_count: int):
        previous = self.get_value()
        self._updating = True
        try:
            self._polygon_editor.set_point_count(max(3, int(point_count)))
        finally:
            self._updating = False
        if self.get_value() != previous:
            self.on_value_changed()
