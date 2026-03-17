"""Height adjustment curves widget."""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from scipy.interpolate import CubicSpline, interp1d
from typing import List, Tuple, Optional

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
        self.control_points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_point = None
        self.update()
        self.curveChanged.emit(self.control_points)
    
    def set_control_points(self, points: List[Tuple[float, float]]):
        """Set control points from external source."""
        self.control_points = sorted(points, key=lambda p: p[0])
        self.update()
    
    def get_control_points(self) -> List[Tuple[float, float]]:
        """Get current control points."""
        return self.control_points.copy()
    
    def apply_curve(self, values: np.ndarray) -> np.ndarray:
        """Apply the curve transformation to input values."""
        if len(self.control_points) < 2:
            return values
        
        # Sort points by x coordinate
        sorted_points = sorted(self.control_points, key=lambda p: p[0])
        x_coords = [p[0] for p in sorted_points]
        y_coords = [p[1] for p in sorted_points]
        
        # Use cubic spline interpolation if we have enough points
        if len(sorted_points) >= 4:
            try:
                spline = CubicSpline(x_coords, y_coords, bc_type='clamped')
                result = spline(np.clip(values, 0, 1))
            except:
                # Fallback to linear interpolation
                interp = interp1d(x_coords, y_coords, kind='linear', 
                                bounds_error=False, fill_value=(y_coords[0], y_coords[-1]))
                result = interp(np.clip(values, 0, 1))
        else:
            # Use linear interpolation for fewer points
            interp = interp1d(x_coords, y_coords, kind='linear',
                            bounds_error=False, fill_value=(y_coords[0], y_coords[-1]))
            result = interp(np.clip(values, 0, 1))
        
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


class HeightCurvesWidget(QWidget):
    """Complete height curves adjustment widget."""

    curvesChanged = pyqtSignal()

    def __init__(self, parent=None, *, title_text: str = "Height Adjustment Curves",
                 default_curve: Optional[List[Tuple[float, float]]] = None):
        super().__init__(parent)
        self.title_text = title_text
        self.default_curve = default_curve
        self.setup_ui()
        if self.default_curve:
            self.curves_graph.set_control_points(self.default_curve)

    def setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"<b>{self.title_text}</b>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Curves graph
        self.curves_graph = CurvesGraphWidget()
        self.curves_graph.curveChanged.connect(lambda points: self.curvesChanged.emit())
        layout.addWidget(self.curves_graph)
        
        # Preset curves
        preset_label = QLabel("Preset Curves:")
        layout.addWidget(preset_label)
        
        presets_layout = QHBoxLayout()
        
        # Preset buttons
        self.linear_btn = QPushButton("Linear")
        self.linear_btn.clicked.connect(self.apply_linear_preset)
        presets_layout.addWidget(self.linear_btn)
        
        self.contrast_btn = QPushButton("Contrast")
        self.contrast_btn.clicked.connect(self.apply_contrast_preset)
        presets_layout.addWidget(self.contrast_btn)
        
        self.smooth_btn = QPushButton("Smooth")
        self.smooth_btn.clicked.connect(self.apply_smooth_preset)
        presets_layout.addWidget(self.smooth_btn)
        
        layout.addLayout(presets_layout)
        
        # More presets
        presets_layout2 = QHBoxLayout()
        
        self.terraced_btn = QPushButton("Terraced")
        self.terraced_btn.clicked.connect(self.apply_terraced_preset)
        presets_layout2.addWidget(self.terraced_btn)
        
        self.lowlands_btn = QPushButton("Lowlands")
        self.lowlands_btn.clicked.connect(self.apply_lowlands_preset)
        presets_layout2.addWidget(self.lowlands_btn)
        
        self.highlands_btn = QPushButton("Highlands")
        self.highlands_btn.clicked.connect(self.apply_highlands_preset)
        presets_layout2.addWidget(self.highlands_btn)
        
        layout.addLayout(presets_layout2)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_curves)
        controls_layout.addWidget(self.reset_btn)
        
        self.smooth_curve_btn = QPushButton("Add Points")
        self.smooth_curve_btn.clicked.connect(self.add_default_points)
        controls_layout.addWidget(self.smooth_curve_btn)
        
        layout.addLayout(controls_layout)
        
        # Instructions
        instructions = QLabel(
            "• Click to add control point\n"
            "• Drag points to adjust curve\n"
            "• Double-click to remove point\n"
            "• Edge points can only move vertically"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(instructions)
        
        layout.addStretch()
    
    def apply_linear_preset(self):
        """Apply linear (identity) curve."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def apply_contrast_preset(self):
        """Apply S-curve for increased contrast."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (0.25, 0.15),
            (0.5, 0.5),
            (0.75, 0.85),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def apply_smooth_preset(self):
        """Apply inverse S-curve for smoother transitions."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (0.25, 0.35),
            (0.5, 0.5),
            (0.75, 0.65),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def apply_terraced_preset(self):
        """Apply stepped curve for terraced terrain."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (0.2, 0.2),
            (0.25, 0.35),
            (0.45, 0.35),
            (0.5, 0.5),
            (0.7, 0.5),
            (0.75, 0.75),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def apply_lowlands_preset(self):
        """Apply curve that emphasizes lower elevations."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (0.3, 0.5),
            (0.6, 0.7),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def apply_highlands_preset(self):
        """Apply curve that emphasizes higher elevations."""
        self.curves_graph.set_control_points([
            (0.0, 0.0),
            (0.4, 0.3),
            (0.7, 0.5),
            (1.0, 1.0)
        ])
        self.curvesChanged.emit()
    
    def reset_curves(self):
        """Reset curves to default."""
        if self.default_curve:
            self.curves_graph.set_control_points(self.default_curve)
        else:
            self.curves_graph.reset_curve()
        self.curvesChanged.emit()
    
    def add_default_points(self):
        """Add default control points."""
        self.curves_graph.add_default_points()
        self.curvesChanged.emit()
    
    def get_control_points(self) -> List[Tuple[float, float]]:
        """Get current control points."""
        return self.curves_graph.get_control_points()

    def set_control_points(self, points: Optional[List[Tuple[float, float]]]):
        """Set control points from an iterable of pairs."""
        if not points:
            return
        sanitized = []
        for pair in points:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                sanitized.append((float(pair[0]), float(pair[1])))
            except (TypeError, ValueError):
                continue
        if not sanitized:
            return
        self.curves_graph.set_control_points(sanitized)
        self.curvesChanged.emit()
    
    def apply_to_heightfield(self, heightfield: np.ndarray) -> np.ndarray:
        """Apply curves adjustment to a heightfield."""
        # Normalize to [0, 1] if needed
        hmin = heightfield.min()
        hmax = heightfield.max()
        
        if hmax > hmin:
            normalized = (heightfield - hmin) / (hmax - hmin)
        else:
            return heightfield
        
        # Apply curve
        adjusted = self.curves_graph.apply_curve(normalized)
        
        # Scale back to original range
        return adjusted * (hmax - hmin) + hmin
