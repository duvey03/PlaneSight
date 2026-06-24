# Training-Bootstrap Go/No-Go Verdict (planesight-9vt)

**Question (ARCHITECTURE.md S8.3):** Can we drape published vector geology-map
linework over the DEM/imagery to auto-generate `(input stack, trace mask)` training
pairs *at scale*, unlocking the v2 supervised ML detector - or does registration
offset / concealed contacts poison the labels?

**Verdict: CONDITIONAL NO-GO.** Naive full-auto draping onto 30 m data is **not**
viable as a primary label source. A *snapped + filtered* bootstrap is viable as an
**auxiliary** label source, behind a hand-labeled seed + active learning. v1
classical ships now and needs no labels, so v2 is never on the critical path.

---

## Evidence

We measured the core risk directly (`scripts/bootstrap_probe.py`): for every
hand-drawn trace pixel, the distance to the nearest detected DEM edge. Our
hand-drawn traces are a **best case** for registration - they were digitized on the
imagery, so published 1:100k maps (positioned to ~50-100 m ≈ 2-3 GLO-30 px) are
strictly worse.

| Region | median offset | p90 | within 2 px | within 3 px | **concealed (> 5 px)** |
|---|---|---|---|---|---|
| Nepal | 2.0 px (60 m) | 5.1 px | 58% | 73% | 11% |
| Pakistan | 1.0 px (30 m) | 10.8 px | 68% | 77% | 18% |
| Canada | 1.4 px (42 m) | 8.5 px | 57% | 69% | 19% |

(Offset is to the nearest *detected* edge at the default budget - a slightly
conservative proxy for distance to DEM signal.)

## The four failure modes, scored

1. **Thin-feature registration catastrophe - CONFIRMED.** Even best-case labels sit
   a median 1-2 px from signal, and 30-43% are > 2 px off. Target features are
   1-2 px wide, so a 2 px offset puts the label *off* the feature. Published maps
   add ~2-3 px more → median ~3-5 px → labels mostly mis-registered. This alone
   sinks naive draping.
2. **Concealed / inferred contacts - CONFIRMED, 11-19%.** ~1 in 6 labels has no
   usable DEM signal within 150 m. Draping these teaches a DEM segmenter pure
   noise. They must be filtered (by map line-type attributes *and* by requiring
   nearby signal).
3. **Generalisation smooths the V's.** Map linework is generalized at its source
   scale, removing the valley sinuosity the strike/dip engine depends on (S6.3).
   A source problem we cannot fix downstream.
4. **Heterogeneous licensing / datums.** Real but tractable with a normalization
   layer (already a tracked concern).

**map2loop context:** map2loop draping works because it consumes *existing vetted
vector maps* for *regional unit boundaries* at coarse tolerance - it does not
demand pixel-accurate thin-feature labels for segmentation, which is our harder
regime.

---

## What makes a bootstrap viable (the salvage path)

The same probe points to the fix - the distance transform we computed **is** the
registration-correction tool:

- **Snap-to-signal.** Move draped linework to the nearest strong DEM/spectral edge
  within a small tolerance, dropping vertices with no edge nearby. This both
  corrects registration *and* filters concealed contacts in one pass.
- **Filter by attribute.** Drop dashed/dotted (inferred/concealed) line types
  before draping.
- **Higher-resolution inputs.** Sentinel-2 (10-20 m) and lidar where available cut
  the pixel offset directly.
- **Treat draped labels as weak/auxiliary**, with the ~2,400-line hand-labeled seed
  + in-plugin active learning (accepted/edited traces) as the trusted primary.

## Recommendation

- **v1 classical detector ships now** - validated end to end (PHASE1_REPORT.md),
  zero training data required. v2 ML is an optional upgrade, never assumed.
- **v2 label strategy:** hand-seed + active learning as primary; snapped + filtered
  draped maps as auxiliary only. **Do not bet v2 on naive full-auto draping.**
- **Recommended confirmation step:** probe one real published source (e.g.
  Macrostrat / a provincial map) over an AOI to ground the published-map offset
  estimate; and re-run this probe with Sentinel-2 edges included to quantify the
  resolution benefit. (Tracked, not blocking.)

This matches the architecture's stated fallback (S8.3) and is now backed by data.
