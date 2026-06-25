# Per-trace NDMI validation — the build gate (planesight-0bf)

**Question.** The probe (`docs/SPECTRAL_DISCRIMINATOR_PROBE.md`) found NDMI separates
creek-from-contact on arid Pakistan at **AUC 0.94** — but on a PROXY label (channel
*pixels* vs hand *contacts*). Before building a terrain-adaptive discriminator, this gate
tests the ACTUAL use case: **per-trace mean-NDMI over DETECTED traces**. Does it flag
creek-following detections without eating valley-crossing contacts, and is it cleaner than
the flow-accumulation filter?

**Method.** `scripts/spectral_validate.py pakistan`. Detect traces
(`ClassicalTraceDetector`, 14 976 on Pakistan), fetch S2 (2026-05-13, 100% cover), compute
`NDMI = normalized_ratio(nir, swir16)`, and take the **mean NDMI along each detected
trace**. Label detections by the flow network (the only creek reference available):
CREEK-proxy = channel-buffer overlap ≥ 50% (4 657); OFF-CHANNEL = overlap 0 (7 009). Pick
the threshold by Youden's J (creek-proxy positive, hand contacts negative). Report flagged
fraction + the **conditioned false-negative** on the 853 hand contacts. Compare head-to-head
with the flow filter on the *same* detected set + contacts.

## Result — the proxy oversold it

| metric | value | note |
|---|---|---|
| AUC creek-proxy vs hand-contact (per-trace) | **0.64** | probe pixel-proxy was **0.94** |
| AUC creek-proxy vs off-channel detected (per-trace) | **0.58** | barely above chance |
| recommended threshold | NDMI ≥ **−0.165** | Youden's J = 0.20 |
| flagged fraction of detected set | 30% (4 443 / 14 976) | |
| TPR on creek-proxy traces | **34%** | catches only a third of creeks |
| conditioned-FN (all hand contacts eaten) | **14%** (120 / 853) | |
| conditioned-FN (valley-crossing contacts) | 13% (74 / 576) | |

Per-trace NDMI distributions are nearly **indistinguishable**: creek-proxy median −0.183
[−0.20, −0.15], off-channel −0.191 [−0.21, −0.16], hand-contact −0.195 [−0.21, −0.18]. The
whole AOI is arid — every population sits at NDMI ≈ −0.18 to −0.20. **Aggregating pixels to
a per-trace mean collapses the 0.94 pixel separation to 0.64 at the trace level**, and to
0.58 when compared within the detected set (creek-proxy vs off-channel).

## Verdict: per-trace NDMI does NOT hold on the real use case

The probe's 0.94 was a proxy artifact — "wet valley-floor pixels vs dry contact pixels,"
which is not "creek trace vs contact trace." On the units we actually act on (detected
traces), NDMI is a **near-chance discriminator** (AUC 0.58–0.64) that catches only 34% of
creek-following traces while flagging 30% of the whole map. **Do not build a per-trace NDMI
drainage discriminator on this evidence.**

### The head-to-head with the flow filter is a trap

| method | flagged (count) | flagged (length) | cond-FN (at-risk, N=132) |
|---|---|---|---|
| NDMI per-trace (≥ −0.165) | 30% | (n/a) | **5%** |
| flow-accum filter | 19% | 18% | **45%** |

At face value NDMI looks "cleaner" — 5% contact-FN on channel-overlapping contacts vs the
flow filter's 45%. **This is misleading.** NDMI's flags are uncorrelated with channel
membership (that is exactly why its AUC vs off-channel is only 0.58), so it doesn't
preferentially eat valley-overlapping contacts the way the flow filter does — but it also
doesn't preferentially flag creeks. A near-random flagger with a tuned threshold gets a low
FN on any sub-population for free; that is not discrimination. The flow filter, for all its
45% at-risk FN, at least flags by a real geomorphic property (`docs/DRAINAGE_SWEEP.md`:
no flow-accumulation knee exists on low-relief arid Pakistan — it is a blunt instrument
here, but it is *aimed*).

## Caveats (honest)

- **BIGGEST CAVEAT — water contamination, not creeks.** The high-NDMI end of the ranking is
  dominated by the **coastal water body** in the AOI (Makran coast, ~62.3°E 25°N near
  Gwadar). The audit panels (`debug/spectral_validate/pakistan/pakistan_ndmi_highNDMI_*`)
  show the top-NDMI "flagged creeks" (NDMI +0.12 to +0.45) sitting on the **land/water
  shoreline** of a dark S2 water surface — open water has strongly positive NDMI, so the
  threshold flags shorelines, not riparian washes. The low-NDMI kept traces (≈ −0.28) are
  over uniform dry terrain, as expected. So the little separation NDMI has is partly a
  **water artifact**, making the real creek-vs-contact AUC even weaker than 0.64.
- **Positive-unlabeled ground truth.** Hand contacts are an incomplete subset; the
  conditioned-FN measures whether NDMI eats the contacts we *do* have, not coverage. No
  hand-labeled creeks exist, so "creek" is itself the flow-network proxy — the very label
  whose reliability is in question. The trace-level AUC is therefore an upper bound on a
  noisy reference, and it is already only 0.64.
- **Per-trace mean is the right aggregate but kills the signal.** Mean (and median) NDMI
  averages the moist-pixel contrast away; the 0.94 lived in individual wet pixels, not in
  the trace summary the discriminator would actually use.
- **Arid-AOI / single-scene / single-region.** One Pakistan AOI, one S2 date (2026-05-13,
  late dry season). Nepal already washed out spectrally (probe). This gate does not rescue
  the arid case, so the complementary-methods story from the probe does **not** survive
  contact with the real use case.

## Recommendation

Drop the NDMI per-trace discriminator (and, by extension, the terrain-adaptive
flow-accum/NDMI selection that l2c/option-A would have built on it). For low-relief arid
terrain the open problem stands: flow-accumulation has no knee (`DRAINAGE_SWEEP.md`) and
per-trace NDMI is near-chance. A productive next probe is **elevation monotonicity along
the trace** (the sinuosity-robust signal that worked on Nepal, HANDOFF `4l8`) tested on
Pakistan, and/or masking S2 water (NDWI) before any moisture index is trusted.

---
*Reproduce:* `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$PWD $HOME/bin/micromamba run
-n gdal python scripts/spectral_validate.py pakistan`. Panels under
`debug/spectral_validate/pakistan/`.
