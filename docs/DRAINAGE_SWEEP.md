# Drainage-filter transferability sweep: Pakistan + Canada (planesight-amn)

**Question.** The whole drainage rule - the overlap flag, the `DRAIN_ACCUM = 15`
channel threshold (recalibrated 8->15 for the depression-filled network), the
along-flow cut, and the rescue rank - was calibrated and validated on **Nepal only**.
Nepal is steep and dissected. **Does the knee transfer to low-relief/arid Pakistan and
Cordilleran Canada, or do we need per-region tuning?**

**Verdict (one line).** The Nepal knee **transfers cleanly to Canada** (keep
`accum = 15`) but **does NOT transfer to Pakistan**, where `accum = 15` over-densifies
the channel network to ~33% of the map and over-flags real contacts; Pakistan needs a
**much higher threshold (~`accum = 120`)** and is still the weak case.

Method: ran the three existing harnesses unchanged (`scripts/drainage_test.py`,
`drainage_verify.py`, `drainage_residual.py`), each on `pakistan` and `canada`, plus a
fresh `nepal` baseline as the anchor. GLO-30 DEM, 30 m, flow downsampled 3x with
depression fill (the hardened `flow_network`). Audit panels under `debug/`. All numbers
below are from those runs (2026-06-24).

---

## 1. The headline: channel-network density vs `accum`

`scripts/drainage_test.py <region>` sweeps the accumulation threshold and prints, per
level, the **channel %** (fraction of the valid map flagged as channel), the
**drainage-removed %** of detected trace length, and the **hand-trace false-negative**
(fraction of the hand-traced contacts flagged - a *positive-unlabeled, incomplete*
yardstick; see caveats).

**Nepal (anchor - steep, dissected):**

```
accum  channel%   removed   hand-FN
  120     6.5%      13%      0.2%
   60     8.5%      18%      0.2%
   30    11.3%      24%      0.2%
   15    15.2%      31%      0.7%   <- calibrated operating point
    8    21.6%      39%      1.9%
    4    34.2%      47%      8.3%
```

**Canada (Cordilleran - steep, dissected):**

```
accum  channel%   removed   hand-FN
  120     5.3%      14%      0.6%
   60     7.2%      19%      1.0%
   30    10.1%      25%      1.4%
   15    14.8%      33%      1.7%   <- accum=15 lands at ~15% density, like Nepal
    8    23.0%      41%      1.7%
    4    40.5%      49%      2.3%
```

**Pakistan (low-relief, arid):**

```
accum  channel%   removed   hand-FN
  120    14.3%       6%      2.3%   <- need THIS far up the sweep to reach ~15% density
   60    21.8%       9%      3.7%
   30    27.8%      13%      5.7%
   15    33.2%      18%      8.3%   <- accum=15 flags a THIRD of the map as channel
    8    38.7%      25%     11.2%
    4    47.4%      32%     15.3%
```

### Reading the knee

The "knee" is where lowering `accum` starts adding channel **disproportionately** -
i.e. where the depression-filled flow network stops tracing real thalwegs and starts
flooding flat ground.

- **Nepal:** channel% accelerates sharply just below 15 (15.2 -> 21.6 -> 34.2 at
  15/8/4). The knee sits **at accum = 15**. This is the calibration, by construction.
- **Canada:** the same sharp acceleration below 15 (14.8 -> 23.0 -> 40.5). The knee is
  **at accum = 15**, and the absolute density (14.8%) is within a point of Nepal's
  15.2%. **The Nepal knee transfers.**
- **Pakistan:** there is **no sharp knee in the swept range**. The curve is gradual
  (14.3 -> 21.8 -> 27.8 -> 33.2 -> 38.7), and density is *already* ~33% at accum = 15.
  To reach a Nepal-equivalent ~15% network you must climb to **accum ~ 120** - roughly
  **8x** the Nepal threshold. The diffuse curve is itself the signal: on low-relief
  arid terrain, depression-filling over-connects the flow field, so flow accumulation
  is intrinsically a **blunter** channel discriminator than it is in steep terrain.

### High-end sweep (`scripts/drainage_accum_hi.py`) - pinning Pakistan, and proving "no knee"

`drainage_test.py` stops at accum = 120, exactly where the Pakistan operating point
lives, so I extended the sweep upward (240..60) and ran Nepal as a control:

```
        Pakistan                         Nepal (control)
accum  channel%  removed  d(ch%)     accum  channel%  removed  d(ch%)
  240    6.6%      4%               240    4.8%      9%
  180    9.8%      5%    +3.2       180    5.5%     10%    +0.7
  120   14.3%      6%    +4.5       120    6.5%     13%    +1.0
   90   17.6%      7%    +3.3        90    7.2%     14%    +0.8
   60   21.8%      9%    +4.2        60    8.5%     18%    +1.3
```

Two things fall out. (1) **Pakistan ~`accum = 115-120` is the right density target**:
14.3% at 120, 17.6% at 90, so ~15% interpolates to ~115. (2) **There is still no knee.**
Pakistan's channel% changes by **+3 to +4.5 points per threshold step** across the
*entire* high range, while Nepal's barely moves (**+0.7 to +1.3**) and sits at roughly
**half** the density at every threshold. Nepal concentrates accumulation into crisp
thalwegs (raising the cut barely shrinks the network); Pakistan's flow field is diffuse
(the network smoothly inflates/deflates with the cut, with no natural operating point).
**This is the strongest evidence that flow-accumulation is a blunt channel discriminator
on Pakistan's low-relief terrain** - a per-region `accum` reaches a sane density but
does not buy the clean, stable network that 15 buys on Nepal/Canada.

> The `>30% or <5%` recalibration trigger from the brief fires for Pakistan
> (**33.2%** at accum 15) and not for Canada (14.8%).

**Note on the harness auto-pick.** `drainage_test.py` auto-selects "the densest channel
net with hand-FN <= 5%" and printed `chosen threshold 60` (Pakistan), `4` (Canada),
`8` (Nepal). **Do not use that selector as the operating point.** Hand-FN is a weak,
positive-unlabeled signal (the hand traces are mostly cross-cutters that rarely sit on
a channel), so it stays low even when the network is visibly over-dense (Canada keeps
FN <= 2.3% all the way down to accum = 4 at 40% channel). The defensible operating
point is read off **channel density** (target a sane ~15% dendritic network) and the
**panels**, not off the hand-FN auto-pick.

---

## 2. Flag behaviour at the current operating point (`accum = 15`, `min_aligned = 50%`, `angle_tol = 30`)

From `scripts/drainage_verify.py`:

| | detected | flagged (count) | flagged (length) | at-risk hand-traces | conditioned FN |
|---|---|---|---|---|---|
| **Nepal** (ref, from HANDOFF/`61f`) | ~7063 | ~47% | ~31% | 24 | **25%** |
| **Canada** | 11443 | 3906 (34%) | **33%** | **14** | 71% (small N) |
| **Pakistan** | 14976 | 2797 (19%) | **18%** | **132** | **45%** |

- **Canada** removes ~33% of detected length - right on top of Nepal's ~31%. Its
  conditioned FN reads 71%, but on only **N = 14** at-risk hand-traces (10/14): too few
  to trust, flagged as small-N by the harness. Read it as "indicative, not alarming",
  and lean on the panels.
- **Pakistan** removes only **18%** of length - *less* than Nepal - yet its conditioned
  FN is **45% over a solid N = 132**. That combination is the tell: the over-dense
  network doesn't flag *more* total length (low-relief detections are scattered and
  often fail the 50% along-flow cut against broad, flow-ambiguous channel blobs), but
  the contacts that **do** overlap a channel are flagged almost half the time. The
  filter is both blunter and more damaging to real contacts in Pakistan.

**Parameter sensitivity** (removed% / conditioned-FN%, from the verify sweep) confirms
fragility in Pakistan and Canada: at `angle_tol` 30 / `min_aligned` 0.5, Pakistan is
18/45 and Canada 33/71, both climbing steeply with looser angle - whereas on Nepal the
operating point sits on the flat part of the curve. The Nepal-tuned (30, 0.5) point is
**not** on a plateau off Nepal.

---

## 3. Residual drainage in the KEPT set

From `scripts/drainage_residual.py` (does the filter leave creeks behind?):

| | KEPT on-channel (overlap>=50%) | KEPT just-under-cut (along-flow 0.3-0.5) |
|---|---|---|
| **Nepal** (ref, `4l8`) | ~10.6% of length | - |
| **Pakistan** | 1860 traces, **11.8%** of length | 1342 traces, 9.1% |
| **Canada** | 1275 traces, **10.3%** of length | 913 traces, 7.9% |

The residual recall gap is **~10-12% of length in all three regions** - consistent with
the Nepal finding that sinuosity-robust overlap still misses meandering on-channel
traces. This part of the behaviour **does** transfer: it is a property of the
flow-only filter, not of the terrain. (`61f`'s decision - flag aggressively, rescue
only big cross-cutters via the rank - is unaffected by region.)

---

## 4. What the panels show (the geologist's check)

Audit panels rendered under `debug/`:

- `debug/drainage_verify/<region>/<region>_audit_hillshade.png` and `_audit_s2.png` -
  30 random **flagged** traces (orange) over the channel network (blue); creek if the
  orange sits on blue, wrongly-removed contact if it crosses/ignores blue.
- `debug/drainage_residual/<region>/<region>_kept_drainage_like.png` - the 30 most
  drainage-like **KEPT** traces (the recall-gap eyeball check).
- `debug/drainage_test/<region>/<region>_drainage_window{1..4}.png` - kept (green) vs
  flagged (red) over hillshade + channel network, in the trace-dense windows.

**Canada (`accum = 15`):** the blue network is a **thin dendritic drainage pattern**
over clearly dissected Cordilleran relief; the orange flagged traces overwhelmingly run
**along genuine valley thalwegs**. These are real creeks - the flag is behaving as it
does on Nepal. *Verdict: the filter is sound on Canada at accum 15.*

**Pakistan (`accum = 15`):** the blue network is an **over-dilated blob** filling
roughly a third of every tile, often with no crisp valley under it in the hillshade.
Orange traces are flagged for merely **touching** that broad blue sheet, including on
near-flat ground (e.g. tiles #4, #5, #16, #29). Some are genuine creeks, but the
network is plainly too dense and is swallowing contacts. *Verdict: at accum 15 the
Pakistan channel mask is an artifact of depression-filling a flat DEM, not a drainage
network - recalibrate.*

These are for the geologist to confirm, but the visual story matches the numbers
cleanly in both directions.

---

## 5. Recommended per-region `accum`

| Region | Recommended `accum` | Why | Confidence |
|---|---|---|---|
| **Nepal** | **15** | Calibration anchor; knee at 15, ~15% density. | High |
| **Canada** | **15** (unchanged) | Knee at 15, density 14.8% ~ Nepal, panels show clean dendritic creeks. **Nepal knee transfers.** | High |
| **Pakistan** | **~115-120** (vs 15) | High-end sweep: 14.3% density at 120, 17.6% at 90 -> ~15% at ~115. Reins in the over-dense blob. | **Medium** - see caveats |

Operationally: **the `accum = 15` default is fine for steep/dissected terrain (Nepal,
Canada) and must be raised ~8x for low-relief arid terrain (Pakistan).** A density-
targeting rule ("pick `accum` so the channel network is ~15% of the valid map") would
generalize better than a fixed `accum`, and is the cleaner long-term fix than a
per-region constant.

---

## 6. Caveats (honest)

1. **Pakistan `accum ~ 115-120` matches density but buys no clean network.** The
   high-end sweep (`drainage_accum_hi.py`, Section 1) confirms ~15% density at ~115-120,
   but also confirms there is **no knee** anywhere from 4 to 240: channel% changes
   +3..+4.5 points *per step* across the whole range (vs Nepal's +0.7..+1.3). So `120`
   is a density match, **not** a stable operating point the way 15 is on Nepal/Canada -
   it sits on a steep slope and is fragile to the exact AOI/DEM. The deeper implication:
   **flow-accumulation alone may be a weak channel discriminator on low-relief arid
   terrain**, and the real fix could be a different channel definition there (or a
   density-targeting rule), not just a bigger number. **This is the single biggest
   caveat for verification.**
2. **Positive-unlabeled ground truth.** Hand traces are an *incomplete* subset of true
   contacts, biased toward cross-cutters. Every "false-negative" / "conditioned FN"
   number is therefore a property of *that subset*, not of all contacts. Judge the flag
   by the **panels** (are flagged traces creeks?), not by hand-FN coverage. This is
   exactly why the harness auto-pick (densest net under 5% hand-FN) misfires.
3. **Canada conditioned FN rests on N = 14.** 71% is 10/14 - too small to be more than
   indicative. The trustworthy Canada signals are the density (14.8%), the removal
   (33% ~ Nepal), and the clean panels, all of which agree that 15 is right.
4. **Single 1-degree slice per region.** Each region is one AOI
   (PK `[62,25,63,26]`, CA `[-117,52,-116,53]`); intra-region terrain varies, so treat
   these as representative, not exhaustive. A second arid AOI would harden the Pakistan
   recommendation.
5. **Lane / scope.** This was a measurement task only. No core code changed
   (`core/detect/drainage.py` untouched); `DRAIN_ACCUM` remains 15 in the harnesses.
   Acting on the Pakistan recommendation (a per-region `accum` or a density-targeting
   rule) is a **separate** change against `core/`, to be scoped on its own bead.

---

## 7. Report-back summary

- **Does the Nepal knee transfer?** **To Canada, yes** (accum 15, 14.8% density, clean
  creek panels). **To Pakistan, no** (accum 15 -> 33% channel, an over-dense blob, 45%
  conditioned FN over N=132).
- **Recommended per-region `accum`:** Nepal 15, Canada 15, **Pakistan ~120** (density-
  targeting ~15% would be the more robust rule).
- **Biggest caveat:** Pakistan has **no knee at all** (confirmed by the 240..60
  high-end sweep: density changes +3..+4.5 pts/step across the whole range vs Nepal's
  +0.7..+1.3). `~120` matches density but is a fragile point on a steep slope -
  flow-accumulation is likely just a blunt channel discriminator on low-relief arid
  terrain, so the durable fix is a density-targeting rule (or a different channel
  definition there), not a per-region constant.
