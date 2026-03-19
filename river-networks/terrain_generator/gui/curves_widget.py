"""Height adjustment curves widget."""

import json
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget
from scipy.interpolate import PchipInterpolator, interp1d


DEFAULT_LINEAR_CURVE = [(0.0, 0.0), (1.0, 1.0)]


def parse_curve_points(
    raw_value: Any,
    fallback: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    clamp_output: bool = False,
) -> List[Tuple[float, float]]:
    """Parse curve points from text or pair-like values."""
    fallback_points = list(fallback or DEFAULT_LINEAR_CURVE)
    parsed: List[Tuple[float, float]] = []

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return fallback_points
        if text.startswith("["):
            try:
                raw_value = json.loads(text)
            except json.JSONDecodeError:
                raw_value = text
        if isinstance(raw_value, str):
            for chunk in raw_value.split(","):
                item = chunk.strip()
                if ":" not in item:
                    continue
                left, right = item.split(":", 1)
                try:
                    parsed.append((float(left), float(right)))
                except ValueError:
                    continue
    if not parsed and isinstance(raw_value, (list, tuple)):
        for pair in raw_value:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                parsed.append((float(pair[0]), float(pair[1])))
            except (TypeError, ValueError):
                continue

    if not parsed:
        return fallback_points

    sanitized: List[Tuple[float, float]] = []
    for x, y in sorted(parsed, key=lambda item: item[0]):
        x_val = float(np.clip(x, 0.0, 1.0))
        y_val = float(np.clip(y, 0.0, 1.0)) if clamp_output else float(y)
        if sanitized and abs(sanitized[-1][0] - x_val) < 1e-6:
            sanitized[-1] = (x_val, y_val)
        else:
            sanitized.append((x_val, y_val))

    return sanitized if len(sanitized) >= 2 else fallback_points


def serialize_curve_points(points: Sequence[Tuple[float, float]]) -> str:
    """Serialize control points to the node's persisted text format."""
    return ", ".join(f"{float(x):.4f}:{float(y):.4f}" for x, y in points)


def apply_curve_points(values: np.ndarray, points: Any) -> np.ndarray:
    """Evaluate a smooth curve against normalized input values."""
    curve_points = parse_curve_points(points, DEFAULT_LINEAR_CURVE)
    x_coords = np.asarray([point[0] for point in curve_points], dtype=np.float64)
    y_coords = np.asarray([point[1] for point in curve_points], dtype=np.float64)
    sample_points = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)

    if len(curve_points) >= 3:
        try:
            interpolator = PchipInterpolator(x_coords, y_coords, extrapolate=True)
            return np.asarray(interpolator(sample_points), dtype=np.float64)
        except ValueError:
            pass

    linear = interp1d(
        x_coords,
        y_coords,
        kind="linear",
        bounds_error=False,
        fill_value=(y_coords[0], y_coords[-1]),
    )
    return np.asarray(linear(sample_points), dtype=np.float64)

class CurvesGraphWidget(QWidget):
    """Interactive curves adjustment graph."""
    
    curveChanged = pyqtSignal(list)  # Emits control points when curve changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMaximumSize(400, 400)
        
        # Control points: list of (x, y) tuples in [0, 1] range
        self.control_points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_point = None
        self.hover_point = None
        
        # Visual settings
        self.margin = 30
        self.grid_lines = 5
        self.point_radius = 6
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def add_default_points(self):
        """Add some default control points for common adjustments."""
        self.control_points = [
            (0.0, 0.0),
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
            (1.0, 1.0)
        ]
        self.update()
        self.curveChanged.emit(self.control_points)
    
    def reset_curve(self):
        """Reset to linear curve."""
        self.control_points = list(DEFAULT_LINEAR_CURVE)
        self.selected_point = None
        self.update()
        self.curveChanged.emit(self.control_points)
    
    def set_control_points(self, points: List[Tuple[float, float]]):
        """Set control points from external source."""
        self.control_points = parse_curve_points(points, DEFAULT_LINEAR_CURVE, clamp_output=True)
        self.selected_point = None
        self.hover_point = None
        self.update()
    
    def get_control_points(self) -> List[Tuple[float, float]]:
        """Get current control points."""
        return self.control_points.copy()
    
    def apply_curve(self, values: np.ndarray) -> np.ndarray:
        """Apply the curve transformation to input values."""
        if len(self.control_points) < 2:
            return np.asarray(values, dtype=np.float64)
        result = apply_curve_points(values, self.control_points)
        return np.clip(result, 0, 1)
    
    def paintEvent(self, event):
        """Paint the curves graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        graph_width = width - 2 * self.margin
        graph_height = height - 2 * self.margin
        
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Draw graph background
        graph_rect = QRectF(self.margin, self.margin, graph_width, graph_height)
        painter.fillRect(graph_rect, Qt.white)
        
        # Draw grid
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DotLine))
        for i in range(self.grid_lines + 1):
            x = self.margin + i * graph_width / self.grid_lines
            y = self.margin + i * graph_height / self.grid_lines
            
            # Vertical lines
            painter.drawLine(int(x), self.margin, int(x), height - self.margin)
            # Horizontal lines
            painter.drawLine(self.margin, int(y), width - self.margin, int(y))
        
        # Draw diagonal reference line
        painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
        painter.drawLine(self.margin, height - self.margin, 
                        width - self.margin, self.margin)
        
        # Draw axes
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(self.margin, self.margin, 
                        self.margin, height - self.margin)
        painter.drawLine(self.margin, height - self.margin,
                        width - self.margin, height - self.margin)
        
        # Draw labels
        painter.setPen(Qt.black)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        painter.drawText(5, height - self.margin + 5, "0")
        painter.drawText(5, self.margin - 5, "1")
        painter.drawText(self.margin - 5, height - 5, "0")
        painter.drawText(width - self.margin - 5, height - 5, "1")
        
        # Draw curve
        if len(self.control_points) >= 2:
            sorted_points = sorted(self.control_points, key=lambda p: p[0])
            
            # Create path for smooth curve
            path = QPainterPath()
            
            # Generate curve points
            curve_points = []
            for i in range(101):
                x = i / 100.0
                y = self.apply_curve(np.array([x]))[0]
                px = self.margin + x * graph_width
                py = self.margin + (1 - y) * graph_height
                curve_points.append(QPointF(px, py))
            
            # Draw the curve
            painter.setPen(QPen(QColor(50, 120, 200), 2))
            path.moveTo(curve_points[0])
            for point in curve_points[1:]:
                path.lineTo(point)
            painter.drawPath(path)
        
        # Draw control points
        for i, (x, y) in enumerate(self.control_points):
            px = self.margin + x * graph_width
            py = self.margin + (1 - y) * graph_height
            
            # Determine point appearance
            if i == self.selected_point:
                painter.setPen(QPen(QColor(255, 100, 0), 2))
                painter.setBrush(QBrush(QColor(255, 150, 50)))
                radius = self.point_radius + 2
            elif i == self.hover_point:
                painter.setPen(QPen(QColor(100, 150, 255), 2))
                painter.setBrush(QBrush(QColor(150, 200, 255)))
                radius = self.point_radius + 1
            else:
                painter.setPen(QPen(QColor(50, 50, 50), 1))
                painter.setBrush(QBrush(QColor(100, 100, 100)))
                radius = self.point_radius
            
            painter.drawEllipse(QPointF(px, py), radius, radius)
    
    def mousePressEvent(self, event):
        """Handle mouse press for selecting/adding points."""
        if event.button() != Qt.LeftButton:
            return
        
        pos = event.pos()
        graph_x, graph_y = self._screen_to_graph(pos.x(), pos.y())
        
        # Check if we're clicking on an existing point
        clicked_point = self._find_point_at(graph_x, graph_y)
        
        if clicked_point is not None:
            self.selected_point = clicked_point
        else:
            # Add new point if not at edges
            if 0.05 < graph_x < 0.95:
                self.control_points.append((graph_x, graph_y))
                self.control_points = sorted(self.control_points, key=lambda p: p[0])
                # Find and select the newly added point
                for i, (x, y) in enumerate(self.control_points):
                    if abs(x - graph_x) < 0.01 and abs(y - graph_y) < 0.01:
                        self.selected_point = i
                        break
                self.curveChanged.emit(self.control_points)
        
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for dragging points and hover effects."""
        pos = event.pos()
        graph_x, graph_y = self._screen_to_graph(pos.x(), pos.y())
        
        # Update hover point
        old_hover = self.hover_point
        self.hover_point = self._find_point_at(graph_x, graph_y)
        
        if old_hover != self.hover_point:
            self.update()
        
        # Handle dragging
        if self.selected_point is not None and event.buttons() & Qt.LeftButton:
            # Don't allow moving edge points horizontally
            if self.selected_point == 0:
                graph_x = 0.0
            elif self.selected_point == len(self.control_points) - 1:
                graph_x = 1.0
            else:
                # Prevent crossing neighboring points
                if self.selected_point > 0:
                    graph_x = max(graph_x, self.control_points[self.selected_point - 1][0] + 0.01)
                if self.selected_point < len(self.control_points) - 1:
                    graph_x = min(graph_x, self.control_points[self.selected_point + 1][0] - 0.01)
            
            # Clamp y to valid range
            graph_y = np.clip(graph_y, 0.0, 1.0)
            
            # Update point position
            self.control_points[self.selected_point] = (graph_x, graph_y)
            self.curveChanged.emit(self.control_points)
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.selected_point = None
            self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to remove points."""
        if event.button() != Qt.LeftButton:
            return
        
        pos = event.pos()
        graph_x, graph_y = self._screen_to_graph(pos.x(), pos.y())
        clicked_point = self._find_point_at(graph_x, graph_y)
        
        # Remove point if it's not an edge point
        if clicked_point is not None and clicked_point not in [0, len(self.control_points) - 1]:
            del self.control_points[clicked_point]
            self.selected_point = None
            self.curveChanged.emit(self.control_points)
            self.update()
    
    def _screen_to_graph(self, x: int, y: int) -> Tuple[float, float]:
        """Convert screen coordinates to graph coordinates [0, 1]."""
        graph_width = self.width() - 2 * self.margin
        graph_height = self.height() - 2 * self.margin
        
        graph_x = (x - self.margin) / graph_width
        graph_y = 1.0 - (y - self.margin) / graph_height
        
        return np.clip(graph_x, 0, 1), np.clip(graph_y, 0, 1)
    
    def _find_point_at(self, graph_x: float, graph_y: float) -> int:
        """Find control point at given graph coordinates."""
        threshold = 0.03  # Detection threshold in graph units
        
        for i, (px, py) in enumerate(self.control_points):
            if abs(px - graph_x) < threshold and abs(py - graph_y) < threshold:
                return i

        return None
