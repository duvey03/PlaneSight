"""PlaneSight QGIS integration checkpoint - paste into the QGIS Python Console.

Proves the analytical core runs inside QGIS's own Python/GDAL and that a fetched
DEM + Sentinel-2 render on the canvas. This is the first cross-checkpoint between
the headless dev loop and QGIS (the deployment target).

How to run:
  1. Edit REPO and (optionally) BBOX below.
  2. QGIS: Plugins > Python Console.
  3. Paste this whole file and run.

Separately, to test the plugin shell itself: run scripts/deploy_to_qgis.py, then
restart QGIS and confirm the PlaneSight toolbar button opens its dialog.
"""

import os
import sys
import tempfile

REPO = r"C:\PlaneSight"                 # <- path to your local clone
BBOX = [82.40, 27.70, 82.50, 27.80]    # <- small AOI (default: Nepal foothills)

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from planesight.core.data import fetch_dem, fetch_sentinel2_band

out = tempfile.mkdtemp()
dem = os.path.join(out, "planesight_dem.tif")
red = os.path.join(out, "planesight_s2_red.tif")

print("Fetching Copernicus GLO-30 DEM + Sentinel-2 (dry-season, <5% cloud)...")
fetch_dem(BBOX, dem)
fetch_sentinel2_band(BBOX, "red", red, datetime="2023-11-01/2024-03-31", cloud_max=5.0)

iface.addRasterLayer(dem, "PlaneSight DEM")  # noqa: F821 (iface is a QGIS global)
iface.addRasterLayer(red, "PlaneSight S2 red")  # noqa: F821
print("[OK] layers loaded - check the QGIS canvas")
