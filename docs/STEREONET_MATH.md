# Stereonet / structural-analysis math (M3, `planesight-8et`)

The pure-numpy "Analyze" core: lower-hemisphere equal-area projection, Fisher
statistics, the orientation tensor, axial means, fold-axis recovery, and rose
diagrams. Dependency-free (numpy only); **no plotting** - rendering (matplotlib
vs custom QPainter) is a separate, later GUI decision. Lives in
`planesight/core/structural/stereonet.py`, tested by `tests/test_stereonet.py`.

This module is the math backend for GUI M3 (epic `planesight-pzz`). It consumes
the `Attitude` objects M2 produces and must use the **same orientation
convention** as `core/attitude/plane_fit.py` (ARCHITECTURE.md S6) - see below.

## Conventions (the load-bearing part)

Coordinate frame: right-handed **`(x = East, y = North, z = Up)`**.

| Quantity | Definition |
|---|---|
| **Trend / azimuth** | degrees clockwise from North; horizontal dir = `(sin T, cos T, 0)`, i.e. `atan2(x, y)` - the same compass as `plane_fit`'s `dip_direction`. |
| **Plunge** | downward inclination from horizontal, `[0, 90]`; a line is `(cos p sin T, cos p cos T, -sin p)`, so **plunge is positive downward** (`z <= 0`). |
| **Strike / dip (RHR)** | right-hand rule: `dip_direction = (strike + 90) mod 360` - the plane dips down to the *right* of the strike azimuth. |
| **Plane upward normal** | `(sin d sin a, sin d cos a, cos d)` with `d = dip`, `a = dip_direction` - **exactly** `plane_fit._fit_core`'s `normal`. |
| **Pole** | the **downward** unit normal (lower hemisphere, `z <= 0`): pole trend `= (strike - 90) mod 360`, pole plunge `= 90 - dip`. Equals the *negated* `plane_fit` normal. |

**The convention trap (verified in tests).** A plane dipping due **East**
(strike 0, RHR) has a pole that plunges toward the **West** (trend 270), because
the pole points *opposite* the dip direction. A mirrored implementation (pole =
dip direction) round-trips fine internally but is geologically backwards;
`test_pole_points_opposite_to_dip_direction` and `test_pole_matches_plane_fit_normal`
pin this down against `plane_fit`'s actual normal.

## API

| Function | Purpose |
|---|---|
| `line_to_xyz(trend, plunge)` / `xyz_to_line(vec, lower_hemisphere=True)` | trend/plunge <-> unit vector. `xyz_to_line` treats the vector as an axis by default (flips upward vectors to their downward antipode). |
| `strike_dip_to_pole(strike, dip)` / `pole_to_strike_dip(pole)` | strike/dip <-> downward pole. Exact inverses (away from the dip-0 singularity, where strike is undefined). |
| `equal_area_xy(trend, plunge)` | lower-hemisphere **equal-area (Schmidt)** projection -> `(x, y)` in the unit disk, North up. |
| `great_circle(strike, dip, n=181)` | equal-area great-circle trace of a plane, for plotting. |
| `fisher_mean(vectors, confidence=0.95)` | Fisher (1953) mean of **directed** unit vectors -> `FisherStats` (mean, R, kappa, alpha95, csd). |
| `orientation_tensor(vectors, normalize=False)` | `sum v_i v_i^T` (sign-invariant). |
| `principal_orientations(vectors)` | sorted `eigh` eigen-decomposition -> normalised `S1>=S2>=S3`, eigenvectors, Woodcock K/C. |
| `axial_mean(vectors)` | mean of **axial** data via the principal eigenvector (the correct estimator for poles). |
| `fold_axis(strikes, dips)` | fold axis (beta) = girdle pole (min-eigenvalue eigenvector) + girdle-vs-cluster shape statistic. |
| `rose_bins(strikes, bin_deg=10)` | bidirectional strike-azimuth histogram. |

### Equal-area radius

Uses the identity `r = sqrt(1 - sin(plunge))` (equivalently
`sqrt(2) sin((90 - plunge)/2)`, equivalently `sqrt(1 + z)` for a lower-hemisphere
unit vector). This is **exact** at both end-points - a vertical line maps to the
disk centre `(0, 0)` and a horizontal line to the rim `r = 1` with no float
fuzz - and satisfies the equal-area property `r^2 = 1 - sin(plunge)`
(`test_equal_area_*`).

### Fisher (1953)

For `n` directed unit vectors with resultant length `R`:

- mean direction = `sum(v_i) / R`;
- precision `kappa = (n - 1) / (n - R)`;
- `cos(alpha) = 1 - (n - R)/R * ((1/(1-c))^(1/(n-1)) - 1)` for confidence `c`
  (alpha95 at `c = 0.95`);
- angular/circular std `csd = 81 / sqrt(kappa)`.

**Worked example reproduced** (`test_fisher_textbook_pmagpy`): the PmagPy
`fisher_mean` example, dec `[140, 127, 142, 136]`, inc `[21, 23, 19, 22]` ->
dec 136.31, inc 21.35, R 3.98121, k 159.69, alpha95 7.29, csd 6.41. We match all
to <= 1e-2 deg. Source: PmagPy documentation, "Fisher mean, a95"
(<https://pmagpy.github.io/PmagPy-docs>), implementing Fisher (1953); see also
Butler (1992), *Paleomagnetism*, App. A.

Fisher is for **directed** data on one hemisphere. For axial data (poles, fold
elements) a vector mean is wrong without hemisphere normalisation - use
`axial_mean` / the orientation tensor instead.

### Orientation tensor, axial mean, fold axis

The orientation tensor `T = sum v_i v_i^T` is sign-invariant (`v` and `-v`
contribute equally), so it is the right tool for axial data. `eigh` gives
principal axes with normalised eigenvalues `S1 >= S2 >= S3` (sum 1):

- **axial mean** = `v1` (largest eigenvalue). `test_axial_mean_sign_invariant`
  flips half the input to antipodes and the mean is unchanged.
- **fold axis (beta)** = `v3` (smallest eigenvalue) = the pole to the best-fit
  girdle. Poles to cylindrically folded bedding lie on a girdle whose pole is
  the fold axis. `test_fold_axis_recovers_known_axis` synthesises bedding by
  rotating a pole about a **known** axis (Rodrigues) and recovers it to < 1 deg.

**Shape statistic (Woodcock 1977)** so a point cluster is never mis-reported as a
fold: `K = ln(S1/S2) / ln(S2/S3)`, strength `C = ln(S1/S3)`. `fold_axis`
classifies `"girdle"` (`K < 1`, a real fold - trust beta), `"cluster"`
(`K > 1`, a point maximum - **do not** trust beta), or `"weak"` (`C` below a
threshold, no preferred orientation). `is_girdle` is True only for a trustworthy
fold.

### Rose diagram

`rose_bins(strikes, bin_deg)` returns `(counts, bin_edges)`. Strikes are axes
(mod 180), so each is counted in both its bin and the opposite bin; `counts` is
symmetric across the diameter. `bin_deg` must divide 360.

## Degenerate handling (all tested)

- `n = 1`: Fisher returns the vector as the mean with NaN dispersion; no crash.
- all identical / antipodal: Fisher resultant cancels -> NaN mean direction (not
  a divide-by-zero); the orientation tensor still yields a cluster.
- non-finite rows are dropped (`_clean_unit_vectors`); empty input -> `n = 0`
  results with NaN fields.
- `fold_axis` with `< 3` measurements -> `is_girdle = False`, NaN beta.

## Biggest caveat (for adversarial verification)

The whole module rests on the **RHR-strike / downward-pole / lower-hemisphere**
convention being byte-identical to `plane_fit`. The internal round-trips can't
catch a global mirror; the geological anchors are `test_pole_matches_plane_fit_normal`
(pole == `-plane_fit.normal` on random synthetic planes) and
`test_pole_points_opposite_to_dip_direction` (east-dip -> west-plunging pole).
Verify the fold-axis recovery and these convention anchors against a *known*
structure, not just internal consistency.
