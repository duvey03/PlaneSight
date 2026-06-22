# PlaneSight

**Derive geological strike and dip measurements from free, global remote-sensing
data - automatically, at regional scale, inside QGIS.**

> Status: **pre-alpha / design phase.** The architecture is defined and the work
> is planned; implementation has not started. This repository currently holds the
> design, the project plan, and the seed training data.

---

## What it is

Global Digital Elevation Models (Copernicus GLO-30) and Sentinel-2 imagery are
freely available for the entire planet. Where geological strata are exposed and
have topographic expression, the *trace* of a bedding plane or contact across the
land surface encodes its 3D orientation. PlaneSight is a planned QGIS plugin that:

1. **Aggregates** free global DEM + Sentinel-2 data for any area of interest, with
   zero account/signup friction.
2. **Detects** candidate bedding/contact traces automatically (classical computer
   vision first; an optional ML detector later).
3. **Computes** strike and dip *en masse* by fitting a plane to each trace where
   it intersects topography (PCA / best-fit-plane on the sampled 3D points), with
   a confidence estimate on every measurement.
4. Keeps a **human in the loop** to review and classify traces - because telling
   bedding from faults, drainage, and roads is the genuinely hard part.

## Why it's novel

A prior-art review found that the three pieces exist separately - automated
lineament detection, and mass strike/dip extraction from *pre-existing map
linework* - but **no tool chains an automated raster trace-detector to a
mass-attitude engine** from raw global data. The closest QGIS tool (GeoTrace) is
unmaintained; map2loop is map-fed, not raster-detected. PlaneSight aims to fill
that gap. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full positioning.

## How it works (pipeline)

```
AOI -> fetch DEM + Sentinel-2 -> derivative stack -> detect traces
    -> [human review & classification] -> sample DEM along traces
    -> PCA plane fit -> strike/dip + uncertainty -> symbolised output
```

## Project status & roadmap

| Phase | Goal |
|---|---|
| 0 | Foundations & data backbone (AOI -> data -> derivative stack) |
| 1 | Research spike (best inputs, detector, training-data bootstrap go/no-go) |
| 2 | Classical detection MVP (dependency-free) |
| 3 | Strike/dip engine (the core science) |
| 4 | Human-in-the-loop review + triage UX |
| 5 | Supervised ML detector (optional, ONNX) |
| 6 | Packaging & public release |

Work is tracked as issues in [`.beads/`](.beads/) (the
[Beads](https://github.com/steveyegge/beads) issue tracker).

## Repository layout

```
ARCHITECTURE.md        Design, science, decisions log, and roadmap (start here)
.beads/                Issue tracker (JSONL is the source of truth)
data/                  Seed training data (hand-labeled traces) + manifest [CC-BY-4.0]
```

## The science, briefly

Each detected trace is the intersection of a geological plane with the terrain.
Sampling the DEM along the trace yields 3D points; the best-fit plane (smallest
PCA eigenvector = plane normal) gives strike and dip. The fit is only meaningful
when the points span two dimensions, so every measurement carries a **conditioning
metric** (is the fit constrained?), a **planarity metric** (is the feature
planar?), and a **propagated uncertainty** driven by the DEM's vertical error.
Details in [`ARCHITECTURE.md`](ARCHITECTURE.md) Section 6.

## Contributing

The project is in its design phase and not yet open for code contributions, but
discussion, ideas, and domain expertise (structural geology, remote sensing, QGIS
plugin development) are very welcome via issues. Built to give back to the open
geoscience community.

## License

- **Code:** [GPL-3.0](LICENSE) - consistent with the QGIS plugin ecosystem.
- **Seed data** (`data/`): [CC-BY-4.0](data/LICENSE) - original interpretive trace
  labels for Canada, Nepal, and Pakistan, digitised over public remote-sensing
  data. See [`data/NOTICE.md`](data/NOTICE.md) for attribution and
  [`data/DATA_MANIFEST.md`](data/DATA_MANIFEST.md) for provenance.
