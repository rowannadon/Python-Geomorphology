"""Main application window for the node-editor-only terrain workflow."""

from __future__ import annotations

from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .node_editor import NodeEditorWidget


class TerrainGeneratorWindow(QMainWindow):
    """Host the node editor as the sole application workspace."""

    def __init__(self):
        super().__init__()
        self.node_editor = NodeEditorWidget(self)
        self.setCentralWidget(self.node_editor)
        self.setWindowTitle("3D Terrain Generator with River Networks")
        self.setGeometry(100, 100, 1400, 1000)
        self._restore_autosaved_graph()

    def _restore_autosaved_graph(self):
        try:
            self.node_editor.restore_autosaved_graph()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Autosave Restore Failed",
                f"Could not restore the autosaved graph.\n{exc}",
            )

    def closeEvent(self, event: QCloseEvent):
        try:
            self.node_editor.autosave_graph()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Autosave Failed",
                f"Could not autosave the current graph before quitting.\n{exc}",
            )
        super().closeEvent(event)
