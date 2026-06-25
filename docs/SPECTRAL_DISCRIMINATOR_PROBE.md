# Spectral creek-vs-contact discriminator probe (planesight-l2c, option A)

> **CORRECTION / SUPERSEDED (see `docs/SPECTRAL_VALIDATE.md`, planesight-0bf).** This
> probe's headline AUC **0.94 was inflated by two confounds** and does NOT survive
> validation: (1) a **pixel proxy** - per-trace mean-NDMI over detected traces collapses
> to AUC ~0.64; (2) the Pakistan AOI is the **Makran coast** (Gwadar), so the high-NDMI
> "creek" pixels are partly **open-water shoreline**, not riparian moisture. The
> "complementary-regimes / NDMI-works-in-arid" conclusion below is therefore **NOT
> established** - the moisture hypothesis is unproven (a clean test needs a non-coastal
> arid AOI), and the per-trace discriminator does not work. Read this probe as the
> cautionary first half; `SPECTRAL_VALIDATE.md` is the verdict.

**Question.** The flow-accumulation drainage FILTER fails on low-relief arid Pakistan.
Can Sentinel-2 discriminate creek-from-contact there (its correct *secondary* use -
DEM-curvature is the strongest *detector* everywhere, Phase 1)?

**Method.** Clean labels, NOT the broken Pakistan flag: CONTACT = hand-traced contacts
(ground truth); CREEK = confident high-accumulation channel pixels (accum>=120). Compare
NDVI and NDMI (moisture) distributions; AUC = P(creek index > contact index), 0.5 = no
separation. Pakistan (arid target) + Nepal (vegetated control). `scripts/spectral_drainage_probe.py`.

## Result

| region | terrain | flow-filter | NDVI AUC | NDMI AUC |
|---|---|---|---|---|
| Nepal | steep, vegetated | works | 0.44 (none) | 0.58 (weak) |
| Pakistan | low-relief, arid | fails (blobs) | 0.18 | **0.94 (strong)** |

**The methods are COMPLEMENTARY across terrain regimes.** Humid Nepal: vegetation
blankets creek and contact alike, so spectral can't separate them - but the flow filter
works. Arid Pakistan: the flow filter blobs, but the moisture contrast is sharp (damp
wash-bottoms vs bone-dry contacts), so NDMI separates strongly. Where one fails, the
other works -> a terrain-adaptive drainage-exclusion: flow-accum for steep/vegetated,
NDMI-moisture for low-relief/arid.

## Caveats (do not over-read)
- **Proxy label:** channel *pixels* vs hand *contacts*, not the actual use case
  (classify detected *traces*). The 0.94 partly reflects "wet valley vs dry slope",
  which correlates with creek-vs-contact but is not identical - a contact crossing a
  moist valley reads creek-like (strike-valley case, recoverable as a review-flag).
- **NEXT (gate before building):** validate per-trace mean-NDMI on DETECTED Pakistan
  traces - does it flag creek-followers without eating valley-crossing contacts?
- Arid-specific: the moisture signal depends on the dry contrast; humid/transitional
  terrain may sit in between (Nepal already shows it washing out).
