"""Visual widgets for node execution status."""

from Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QPropertyAnimation, QEasingCurve


class NodeProgressBar(QtWidgets.QGraphicsRectItem):
    """Progress bar that appears above a node during execution."""
    
    def __init__(self, node_view, parent=None):
        # CRITICAL: Make this a child of node_view so it moves with the node
        super().__init__(node_view)
        self.node_view = node_view
        self.progress = 0.0  # 0.0 to 1.0
        
        # Styling
        self.bar_height = 8
        self.bar_width = 160
        self.background_color = QtGui.QColor(60, 60, 60, 200)
        self.progress_color = QtGui.QColor(255, 200, 0, 255)  # Yellow
        self.border_color = QtGui.QColor(100, 100, 100, 255)
        
        # Position above node (in LOCAL coordinates relative to parent)
        self.setZValue(1000)  # High z-value to appear on top
        self.update_position()
        
        # Animation for indeterminate progress
        self.animation_offset = 0.0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate)
        self.is_indeterminate = False
    
    def update_position(self):
        """Position the bar above the node in LOCAL coordinates."""
        if self.node_view:
            node_rect = self.node_view.boundingRect()
            # Use LOCAL coordinates (relative to parent node_view)
            x = node_rect.center().x() - (self.bar_width / 2)
            y = node_rect.top() - self.bar_height - 8
            self.setPos(x, y)
    
    def set_progress(self, progress):
        """Set progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        self.is_indeterminate = False
        self.animation_timer.stop()
        self.update()
    
    def set_indeterminate(self, enabled=True):
        """Enable/disable indeterminate animation."""
        self.is_indeterminate = enabled
        if enabled:
            self.animation_timer.start(30)  # ~33 FPS
        else:
            self.animation_timer.stop()
        self.update()
    
    def _animate(self):
        """Animate the indeterminate progress bar."""
        self.animation_offset += 0.02
        if self.animation_offset > 1.0:
            self.animation_offset = 0.0
        self.update()
    
    def boundingRect(self):
        """Define the bounding rectangle."""
        return QtCore.QRectF(0, 0, self.bar_width, self.bar_height)
    
    def paint(self, painter, option, widget):
        """Draw the progress bar."""
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        rect = self.boundingRect()
        
        # Draw background
        painter.setPen(QtGui.QPen(self.border_color, 1))
        painter.setBrush(self.background_color)
        painter.drawRoundedRect(rect, 4, 4)
        
        # Draw progress
        if self.is_indeterminate:
            # Animated stripe pattern
            progress_rect = QtCore.QRectF(
                rect.x() + 2,
                rect.y() + 2,
                rect.width() - 4,
                rect.height() - 4
            )
            
            # Create a moving window
            window_width = progress_rect.width() * 0.3
            window_pos = self.animation_offset * (progress_rect.width() + window_width) - window_width
            
            # Clip to progress area
            painter.setClipRect(progress_rect)
            
            # Draw moving window
            window_rect = QtCore.QRectF(
                progress_rect.x() + window_pos,
                progress_rect.y(),
                window_width,
                progress_rect.height()
            )
            
            gradient = QtGui.QLinearGradient(window_rect.left(), 0, window_rect.right(), 0)
            gradient.setColorAt(0.0, QtGui.QColor(255, 200, 0, 0))
            gradient.setColorAt(0.5, QtGui.QColor(255, 200, 0, 255))
            gradient.setColorAt(1.0, QtGui.QColor(255, 200, 0, 0))
            
            painter.setBrush(gradient)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(progress_rect, 3, 3)
            
        else:
            # Determinate progress
            progress_width = (rect.width() - 4) * self.progress
            if progress_width > 0:
                progress_rect = QtCore.QRectF(
                    rect.x() + 2,
                    rect.y() + 2,
                    progress_width,
                    rect.height() - 4
                )
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(self.progress_color)
                painter.drawRoundedRect(progress_rect, 3, 3)


class NodeExecutionLabel(QtWidgets.QGraphicsTextItem):
    """Label showing execution time after node completes - PERSISTENT."""
    
    def __init__(self, node_view, execution_time, parent=None):
        # CRITICAL: Make this a child of node_view so it moves with the node
        super().__init__(node_view)
        self.node_view = node_view
        self.execution_time = execution_time
        
        # Format time
        if execution_time < 1.0:
            time_str = f"{execution_time*1000:.0f} ms"
        else:
            time_str = f"{execution_time:.2f} s"
        
        self.setPlainText(time_str)
        
        # Styling
        font = QtGui.QFont("Arial", 8)
        self.setFont(font)
        self.setDefaultTextColor(QtGui.QColor(180, 180, 180))
        
        # Add background
        self.background_color = QtGui.QColor(40, 40, 40, 220)
        self.border_color = QtGui.QColor(100, 100, 100, 255)
        self.padding = 4  # Store padding as instance variable
        
        # Position above node (in LOCAL coordinates relative to parent)
        self.setZValue(1000)
        self.update_position()
        
        # NO AUTO-FADE - Keep it persistent
        # User removed the fade timer entirely
    
    def boundingRect(self):
        """
        CRITICAL: Override to include padding/border so Qt knows the full area to redraw.
        Without this, moving the node leaves ghost borders behind.
        """
        # Get the text bounding rect from parent class
        text_rect = super().boundingRect()
        # Expand by padding to include the background border
        return text_rect.adjusted(-self.padding, -self.padding, 
                                   self.padding, self.padding)
    
    def update_position(self):
        """Position the label above the node in LOCAL coordinates."""
        if self.node_view:
            node_rect = self.node_view.boundingRect()
            label_rect = self.boundingRect()
            # Use LOCAL coordinates (relative to parent node_view)
            x = node_rect.center().x() - (label_rect.width() / 2)
            y = node_rect.top() - label_rect.height() - 8
            self.setPos(x, y)
    
    def paint(self, painter, option, widget):
        """Draw the label with background."""
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw background
        # Use the text rect (not the full bounding rect) for positioning
        text_rect = super().boundingRect()
        bg_rect = text_rect.adjusted(-self.padding, -self.padding, 
                                      self.padding, self.padding)
        
        painter.setPen(QtGui.QPen(self.border_color, 1))
        painter.setBrush(self.background_color)
        painter.drawRoundedRect(bg_rect, 4, 4)
        
        # Draw text
        super().paint(painter, option, widget)