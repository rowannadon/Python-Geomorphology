#!/usr/bin/env python3
"""
3D Terrain Generator with River Networks
Main application entry point.
"""

import os
import sys

def configure_gui_scale():
    """Apply an optional startup-only UI scale override for this app."""
    for env_name in ("TERRAIN_GENERATOR_UI_SCALE", "TERRAIN_GENERATOR_GUI_SCALE"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue

        try:
            scale = float(raw_value)
        except ValueError:
            print(
                f"Warning: ignoring {env_name}={raw_value!r}; "
                "expected a positive number such as 0.85"
            )
            return

        if scale <= 0.0:
            print(
                f"Warning: ignoring {env_name}={raw_value!r}; "
                "scale must be greater than 0"
            )
            return

        os.environ["QT_SCALE_FACTOR"] = f"{scale:g}"
        print(f"Using GUI scale override from {env_name}={scale:g}")
        return


configure_gui_scale()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat

# Import from package structure
from terrain_generator.gui import TerrainGeneratorWindow

try:
    import qdarktheme
    DARK_THEME_AVAILABLE = True
except ImportError:
    DARK_THEME_AVAILABLE = False
    print("Note: qdarktheme not installed. Install with: pip install pyqtdarktheme")


def setup_opengl():
    """Configure OpenGL settings."""
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def main():
    """Run the terrain generator application."""
    # Create application
    app = QApplication(sys.argv)
    
    # Configure application
    app.setApplicationName("3D Terrain Generator")
    app.setOrganizationName("TerrainGen")
    
    # Apply dark theme if available
    if DARK_THEME_AVAILABLE:
        app.setStyleSheet(qdarktheme.load_stylesheet())
    
    # Setup OpenGL
    setup_opengl()
    
    # Create and show main window
    window = TerrainGeneratorWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
