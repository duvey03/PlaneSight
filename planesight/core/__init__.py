"""PlaneSight pure-Python core.

Everything under ``planesight.core`` is QGIS-free and depends only on libraries
bundled with QGIS (numpy, scipy, GDAL) - see ARCHITECTURE.md decision D11. This
keeps the analytical code unit-testable in CI without a QGIS runtime, and keeps
v1 dependency-free for the end user.
"""
