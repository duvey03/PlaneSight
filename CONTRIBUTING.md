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

### Headless GDAL (optional, for the raster I/O tests)

Most of the core is pure numpy/scipy and needs no GDAL. The thin raster-fetch
edge (`planesight/core/data/fetch.py`) does, and its tests
(`tests/test_fetch_gdal.py`) auto-skip when GDAL is absent. For a no-sudo
headless GDAL that mirrors QGIS's bundled GDAL:

```bash
scripts/setup_dev_gdal.sh 'gdal=3.10'   # pin to match your QGIS (Help > About)
MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD \
  $HOME/bin/micromamba run -n gdal pytest   # runs the GDAL tests too
```

Dev philosophy: develop the analytical core headless (fast loop, full CI), and
use QGIS for **integration checkpoints at phase boundaries** - not only at the
end. Keep GDAL at the I/O edges; keep algorithms on numpy arrays.

## Conventions

- `planesight/core/` must **never import QGIS** - keep it CI-testable.
- v1 stays **dependency-free** (numpy/scipy/GDAL, all bundled with QGIS); heavier
  deps (onnxruntime, etc.) are deferred to the optional ML upgrade.
- Follow PEP 8; ruff enforces style in CI. Add docstrings and type hints.

## License

Code contributions are licensed under **GPL-3.0**. Data contributions under
**CC-BY-4.0**.
