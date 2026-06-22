"""Placeholder main dialog for PlaneSight.

A minimal scaffold window proving the plugin loads and is reachable. The real
workflow UI (AOI selection, parameters, run controls) arrives in later phases.
"""

from __future__ import annotations

from qgis.PyQt.QtWidgets import QDialog, QLabel, QVBoxLayout

from .. import __version__


class PlaneSightDialog(QDialog):
    """Minimal placeholder dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PlaneSight")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"PlaneSight v{__version__}"))
        layout.addWidget(
            QLabel("Pre-alpha scaffold. Workflow UI coming in later phases.")
        )
