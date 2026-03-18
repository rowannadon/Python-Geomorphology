"""Custom widgets for terrain nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


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
