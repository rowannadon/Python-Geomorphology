"""Main application window for the node-editor-only terrain workflow."""

from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow

from .node_editor import NodeEditorWidget


class TerrainGeneratorWindow(QMainWindow):
    """Host the node editor as the sole application workspace."""

    def __init__(self):
        super().__init__()
        self.node_editor = NodeEditorWidget(self)
        self.setCentralWidget(self.node_editor)
        self.setWindowTitle("3D Terrain Generator with River Networks")
        self.setGeometry(100, 100, 1400, 1000)
