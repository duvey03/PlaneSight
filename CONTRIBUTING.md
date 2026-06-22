# Contributing to PlaneSight

PlaneSight is in an early design/scaffold phase. Code contributions are not open
yet, but issues, ideas, and domain expertise (structural geology, remote sensing,
QGIS plugin development) are very welcome.

## Project layout

```
planesight/        The QGIS plugin package (this folder is what installs into QGIS)
  core/            Pure-Python, QGIS-free, numpy/scipy/GDAL only - unit-tested in CI
  gui/, tasks/     QGIS/Qt-facing code (loaded only inside QGIS)
  resources/       Icons and assets
tests/             pytest suite for the pure core
scripts/           Dev helpers (e.g. deploy the plugin into local QGIS)
ARCHITECTURE.md    Design, science, and decisions log (read this first)
data/              Seed training data (CC-BY-4.0)
.beads/            Issue tracker
```

## Dev setup

```bash
python -m pip install -r requirements-dev.txt
pytest                              # run the core test suite (no QGIS needed)
ruff check planesight tests scripts # lint
python scripts/deploy_to_qgis.py    # symlink the plugin into local QGIS to test
```

## Conventions

- `planesight/core/` must **never import QGIS** - keep it CI-testable.
- v1 stays **dependency-free** (numpy/scipy/GDAL, all bundled with QGIS); heavier
  deps (onnxruntime, etc.) are deferred to the optional ML upgrade.
- Follow PEP 8; ruff enforces style in CI. Add docstrings and type hints.

## License

Code contributions are licensed under **GPL-3.0**. Data contributions under
**CC-BY-4.0**.
