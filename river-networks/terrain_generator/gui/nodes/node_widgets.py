"""Custom widgets for terrain nodes."""

from NodeGraphQt import BaseNode
from NodeGraphQt.constants import NodePropWidgetEnum
from NodeGraphQt.qgraphics.node_base import NodeItem
from PyQt5.QtWidgets import QPushButton, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt


class ExecuteButtonWidget(QWidget):
    """Widget with an execute button for nodes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.button = QPushButton("▶ Execute")
        self.button.setMaximumHeight(24)
        layout.addWidget(self.button)
    
    def get_value(self):
        return None
    
    def set_value(self, value):
        pass