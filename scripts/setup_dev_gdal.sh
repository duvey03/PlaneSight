#!/usr/bin/env bash
# Set up a no-sudo headless GDAL dev environment via micromamba, so the
# analytical core (planesight.core, incl. the GDAL raster I/O) can be developed
# and tested OUTSIDE QGIS - fast inner loop. QGIS remains the integration target;
# this is just a faithful headless proxy for its bundled GDAL.
#
# Usage:
#   scripts/setup_dev_gdal.sh                 # recent GDAL from conda-forge
#   scripts/setup_dev_gdal.sh 'gdal=3.10'     # pin to match YOUR QGIS GDAL
#                                             # (QGIS: Help > About shows GDAL ver)
set -euo pipefail

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MM="$HOME/bin/micromamba"
GDAL_SPEC="${1:-gdal>=3.8}"

if [ ! -x "$MM" ]; then
  echo "Installing micromamba (static binary, no sudo)..."
  ( cd "$HOME" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba )
fi

echo "Creating 'gdal' env with: $GDAL_SPEC"
"$MM" create -y -n gdal -c conda-forge "$GDAL_SPEC" python=3.12 numpy scipy pytest ruff

echo
echo "[OK] Done. Run the full suite (incl. GDAL tests) with:"
echo "  MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX PYTHONPATH=\$PWD $MM run -n gdal pytest"
