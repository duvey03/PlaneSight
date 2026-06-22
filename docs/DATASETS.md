# Open Datasets for Training & Validation

A sourcing scan (Phase 1, issue `planesight-t22`) of openly-licensed datasets that
can bolster PlaneSight's **training** and especially **validation**. Most open
structural data is strike/dip *points* (field measurements), so its highest value
is **ground-truth validation** - directly addressing open question Q4 (where is
our ground truth?). Map-scale *trace* labels paired with DEMs are rarer.

Verify each dataset's license at the point of download before redistributing any
derived product or shipped model (see `ARCHITECTURE.md` S16).

## Shortlist (pursue first)

| Rank | Dataset | Why | License | Access |
|---|---|---|---|---|
| 1 | **Macrostrat orientation points** | Only open, no-auth, bulk-queryable API returning explicit `strike`/`dip`/`dip_dir`; continental scale | CC-BY-4.0 | REST, GeoJSON/CSV, bbox filter |
| 2 | **Geoscience Australia Structural Measurements DB** | National field-measured strike/dip via live WFS; clean license; one download | CC-BY-4.0 | WFS -> CSV/SHP |
| 3 | **Svalbox "Konusdalen West" DOM** | The only true labeled *benchmark*: point cloud + 1.6 cm DTM + 72 GNSS-measured strike/dip + digitized planes, one package | CC-BY-4.0 | Direct download |
| 4 | **USGS GeMS `OrientationPoints` + `ContactsAndFaults`** | Public-domain strike/dip + contact traces w/ per-point uncertainty; co-registers to 1 m 3DEP lidar | Public domain / CC0 | Per-map via ScienceBase/NGMDB |
| 5 | **map2loop / Loop3D (WA Hamersley + WAROX)** | The canonical analog pipeline: already drapes contacts+orientations over auto-fetched DEM; reusable code + ground truth | Code MIT; data CC-BY-4.0 | `pip install`, WFS |

Runner-up for raw terrain at scale: **OpenTopography** (lidar/DTMs; per-dataset
license, often CC-BY).

## By category

### 1. Strike/dip point databases (validation gold)
- **Macrostrat** (CC-BY-4.0) - `macrostrat.org/api/v2/geologic_units/map/points`; fields `strike, dip, dip_dir, point_type, certainty, source_id`. Caveat: many are map-symbol digitizations (null `dip_dir`, 0/90 end-members); QC by `certainty`/`source_id`.
- **Geoscience Australia** (CC-BY-4.0) - WFS `services.ga.gov.au/gis/field-geology/wfs`, typename `fieldsite:Structures`. Cleanest single national download.
- **GSWA WAROX** (WA; CC-BY-4.0, confirm at download) - the real-field strike/dip behind map2loop's case study; via DASC.
- **Geological Survey of Victoria** (CC-BY-4.0) - dip/dip-direction + plunge/azimuth, declination-corrected.
- **USGS GeMS `OrientationPoints`** (CC0) - `Azimuth`(RHR strike), `Inclination`(dip), `OrientationConfidenceDegrees`(usable as validation weight). Per-map harvest.
- **Arizona Geological Survey** (effectively open) - 120+ GeMS maps, often 1:24k, GeoPackage available.
- **Geological Survey of Canada** (OGL-Canada) - `AZIMUTH`/`DIPPLUNGE`; scattered per-publication.

### 2. Lidar / photogrammetry-derived orientations
- **Svalbox Konusdalen West** (CC-BY-4.0) - point cloud + 1.6 cm DTM + 72 GNSS strike/dip + digitized GeoPackage planes. The closest thing to a labeled benchmark.
- **Svalbox DMDb** (CC-BY per-DOI) - ~135 DOMs / 114 km^2 of vegetation-free Arctic outcrop.
- **OpenTopography** (per-dataset license) - best-in-class raw lidar/DTMs to derive structure from; mind the academic-only restriction on 3DEP/NOAA *via OT* (free direct from USGS/NOAA).
- **MethodsX MATLAB tool + synthetic cylinder data** (CC-BY) - tiny, ideal for **unit-testing our plane-fit math** and the synthetic-DEM gate (D14).

### 3. Contact/bedding traces paired with DEMs (training labels)
- **map2loop / Loop3D** (MIT code; CC-BY-4.0 data) - ingests contacts + strike/dip + auto-downloaded DEM and already drapes linework over terrain; sampling/projection utilities are directly reusable. Caveat: 1:500k traces co-register loosely to 30 m SRTM - prefer finer maps / 90 m DTM.
- **USGS GeMS `ContactsAndFaults` + 3DEP 1 m lidar** (CC0) - the highest-resolution trace<->DEM pairing available where 3DEP exists.
- **DARPA CriticalMAAS / AI4CMA** (CC0) - ~48 annotated maps with line ground truth for training the trace-*detection* model; no DEM bundled (upstream only).

## NOT openly usable - flagged
- **StraboSpot** - highest-fidelity field data, but no explicit reuse license, no public bulk API, private-by-default. Confirm per released project; do not assume open.
- **BGS DiGMapGB-50** - restricted BGS data licence (not OGL). Avoid. (BGS Geology 625k linework *is* OGL but has no orientation points; BGS strike/dip DB not yet released.)
- **OneGeology** - portrayal aggregator, per-provider/unclear licensing, no bulk points. Discovery only.
- **V3Geo** - per-model licenses, some CC-BY-**NC** (blocks commercial). Check each.
- **EarthChem/IEDA** - geochemistry only, no structure. Dead end.

## Two recurring engineering caveats
1. **Convention normalization.** GeMS stores strike as `Azimuth` (RHR); Macrostrat/GA/GSC each use different field names and some store dip-direction not strike. Build a converter and parse each source's glossary - this is the main data-cleaning hazard.
2. **Scale-aware validation.** Map-digitized points carry symbol-level positional/orientation generalization. Weight by `OrientationConfidenceDegrees`/`certainty`/`PlotAtScale`; do not validate fine DEM-derived dips against coarse map symbols. For field-precision validation prefer GNSS-located sets (Svalbox, WAROX, GA field DB).

## Recommended first moves
- **Validation harness:** pull Macrostrat (API) + Geoscience Australia (WFS) for broad coverage, plus **Svalbox Konusdalen West** as the high-precision benchmark.
- **Plane-fit unit tests:** use the MethodsX synthetic cylinder data alongside our own synthetic-plane DEM (D14).
- **Training-label bootstrap (gated by `planesight-9vt`):** evaluate USGS GeMS `ContactsAndFaults` over 3DEP-lidar areas and the map2loop WA datasets.
