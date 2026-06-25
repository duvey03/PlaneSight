"""Main PlaneSight plugin class: wires the plugin into the QGIS GUI.

Responsible only for lifecycle (initGui/unload) and showing the dockwidget. All
analytical work lives in ``planesight.core`` and runs in background QgsTasks.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .gui.dockwidget import PlaneSightDockWidget

PLUGIN_DIR = os.path.dirname(__file__)
ICON_PATH = os.path.join(PLUGIN_DIR, "resources", "icon.svg")


class PlaneSightPlugin:
    """QGIS plugin entry point for PlaneSight."""

    def __init__(self, iface):
        self.iface = iface
        self._action = None
        self._dock = None

    def initGui(self):
        """Create the toolbar/menu action (called by QGIS on load)."""
        self._action = QAction(QIcon(ICON_PATH), "PlaneSight", self.iface.mainWindow())
        self._action.setObjectName("PlaneSightOpen")
        self._action.setCheckable(True)
        self._action.triggered.connect(self._toggle)
        self.iface.addToolBarIcon(self._action)
        self.iface.addPluginToMenu("PlaneSight", self._action)

    def unload(self):
        """Remove the dock + action (called by QGIS on unload)."""
        if self._dock is not None:
            self.iface.removeDockWidget(self._dock)
            self._dock = None
        if self._action is not None:
            self.iface.removeToolBarIcon(self._action)
            self.iface.removePluginMenu("PlaneSight", self._action)
            self._action = None

    def _toggle(self, checked):
        """Show/hide the PlaneSight dock, creating it on first use."""
        if self._dock is None:
            self._dock = PlaneSightDockWidget(self.iface, self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self._dock)
            self._dock.visibilityChanged.connect(self._action.setChecked)
        self._dock.setVisible(checked)
