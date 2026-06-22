# PlaneSight - Starting Data Manifest

Consolidated raw training data inherited from the 2020-2021 prototype work.
All files here are **copies**; originals remain untouched at their source paths.
Nothing in this folder has been reprojected, cleaned, or merged yet - it is the
raw starting point for the data-preparation work (Phase 0).

**Compiled:** 2026-06-22
**License:** CC-BY-4.0 (see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md))

---

## Summary

Three hand-labeled regions of geological trace polylines, plus AOI boundary
polygons for two of them.

| Region | File | Lines | Vertices | CRS | Attributes |
|---|---|---|---|---|---|
| Canada | `raw/canada/canada_traces.shp` | 653 | 5,933 | EPSG:4326 (geographic) | `id` |
| Canada (v2) | `raw/canada/canada_traces_v2_1160.itf.ili` | 1,160 | 10,155 | EPSG:4326 | none |
| Nepal | `raw/nepal/nepal_traces.shp` | 411 | 3,247 | EPSG:32644 (UTM 44N, metric) | `id`, `Class` |
| Pakistan | `raw/pakistan/pakistan_traces.shp` | 853 | 11,124 | EPSG:4326 (geographic) | `id` |

Boundary / AOI polygons (extent definitions, not training lines):

| Region | File | CRS | Extent (lon/lat) |
|---|---|---|---|
| Nepal | `raw/boundaries/nepal_aoi.shp` | EPSG:4326 | 82.00-83.00 E, 27.00-28.00 N |
| Pakistan | `raw/boundaries/pakistan_aoi.shp` | EPSG:4326 | 62.00-63.00 E, 25.00-26.00 N |

Canada has no boundary polygon; its extent is the data bbox: **-117.0 to -116.0
lon, 52.0 to 53.0 lat** (a 1-degree tile).

Total usable training lines (preferring Canada v2): **~2,424 polylines** across
three structurally distinct regions.

---

## Per-region detail & provenance

### Canada
- **`canada_traces.shp`** (653 lines) - copied from
  `F:\QGIS Dev\Strike Dip Predict\CanadaTrainingPolylines.shp`. Geographic CRS.
  Single `id` attribute (no classification). Identical feature/vertex count to
  the old `DECENT_CanadaPolylines.itf.ili` export.
- **`canada_traces_v2_1160.itf.ili`** (1,160 lines) - copied from
  `F:\Coding Projects\2020\Python\Strike Dip Predict\Raw Data\Canada_Polylines_Updated.itf.ili`.
  This is a **larger, richer superset** of the Canada labels but exists only in
  INTERLIS1 text format. **Open item:** convert to shapefile/GeoPackage and
  decide whether it supersedes `canada_traces.shp`. Likely the better Canada
  source.

### Nepal (most attributed)
- **`nepal_traces.shp`** (411 lines) - copied from
  `F:\QGIS Dev\Strike Dip Predict\Training Strike Dip Polylines.shp`. The
  generic original filename hid that this is **Nepal** data (confirmed by its
  UTM 44N coordinates falling at ~82 E, ~27.8 N, matching the Nepal AOI).
- **Projected CRS (UTM 44N, metres)** - the only region already in a metric CRS,
  which is convenient for the plane-fit math.
- Has a **`Class`** field, but it is **sparsely populated**: 54 lines labeled
  `Strike`, 357 blank. Classification was started, not finished.
- A superseded older backup exists at the source
  (`Training Strike Dip Polylines Backup 2020_11_20.shp`, 266 lines, same 54
  `Strike` labels) - **not copied** (older, fewer features).

### Pakistan (largest)
- **`pakistan_traces.shp`** (853 lines) - copied from
  `F:\QGIS Dev\Strike Dip Predict\Satellite Imagery\Pakistan\Pakistan Training Polygons.shp`.
  Despite the "Polygons" filename, the geometry is **PolyLine** (MultiLineString
  in QGIS). Geographic CRS, single `id` attribute. The largest and most recently
  edited (Feb 2021) hand-created set.

---

## Known issues / open items

1. **Mixed CRS.** Canada & Pakistan are geographic (EPSG:4326); Nepal is
   projected (EPSG:32644). Plane-fit math requires a metric CRS, so a per-region
   reprojection step is needed downstream. Do **not** naively merge across CRS.
2. **Inconsistent schema.** Only Nepal has a `Class` field, and it is mostly
   empty. A normalised schema (region, class, source, geometry) should be
   defined when building the unified training layer.
3. **No semantic labels for most data.** ~2,000+ lines have no
   bedding/contact/fault classification. This is the core gap the bedding-vs-fault
   problem depends on - these will need (re)classification, likely via the
   plugin's human-in-the-loop tools.
4. **Canada duplication.** Decide between the 653-line shapefile and the
   1,160-line INTERLIS superset (see Canada section).
5. **Imagery/DEM not consolidated here.** The source tree also contains Sentinel-2
   and Landsat imagery, DEM/TPI rasters, and band sets per region. Those are
   large and are intentionally left at the source for now; this folder is
   labels-only. Source roots:
   - `F:\QGIS Dev\Strike Dip Predict\` (rasters, DEM Files, Satellite Imagery)
   - `F:\Coding Projects\2020\Python\Strike Dip Predict\` (original chunker pipeline)

## Excluded (intentionally not copied)
- `OtherTests\TestOutputs\TestVectorSegments.shp` - segmentation **output**
  (1,832 polygons with mean/variance bands), not training input.
- Pakistan `Finished_Polygons.shp` (2 polygons) - appears to be a small
  reviewed-area marker, not training lines.

---

## Suggested next step (Phase 0)
Build a single normalised **GeoPackage** (`planesight_traces.gpkg`) with all
three regions reprojected to a common analysis CRS (or kept per-region UTM with
a region tag), a unified schema, and the Canada v2 superset converted from
INTERLIS. This requires GDAL/QGIS tooling and is tracked as a Phase 0 task, not
done here.
