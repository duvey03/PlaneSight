"""Label-free linearity scoring of a detection mask (pure numpy/scipy).

Recall against incomplete labels cannot see what the structure tensor is *for* -
producing long, clean, connected lines rather than scattered blobs (the
contact-vs-mound goal, ARCHITECTURE.md S8). This module measures that directly and
WITHOUT labels: threshold a response into a detection mask, find its connected
components, and ask how *elongated* each component is.

It's the same fabric/eigenvalue idea used for the plane fit and the structure
tensor, applied a third time - here to each component's cloud of pixel
COORDINATES. A long thin line has one large coordinate-variance axis and a tiny
one (elongation ~1); a blob is isotropic (elongation ~0); scattered speckle makes
many tiny components that score ~0. So a high linearity score means "my detections
look like contacts, not noise," independent of any ground truth. Pure numpy/scipy
(decision D11).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label, sum_labels

from .score import detect_at_budget

__all__ = [
    "linearity_metrics",
    "linearity_at_budget",
]

_EIGHT = np.ones((3, 3), dtype=int)  # 8-connectivity keeps diagonal lines whole


def linearity_metrics(mask, min_pixels: int = 5) -> dict:
    """Geometry of a boolean detection mask's connected components.

    Args:
        mask: 2D boolean detections.
        min_pixels: components smaller than this score 0 elongation - too small to
            be a meaningful line (this is what penalises scattered speckle).

    Returns:
        dict with ``linearity`` (size-weighted mean component elongation in [0, 1];
        the headline), ``n_components``, ``mean_size``, ``largest_size``.
    """
    mask = np.asarray(mask, dtype=bool)
    lab, n = label(mask, structure=_EIGHT)
    total = int(mask.sum())
    if n == 0 or total == 0:
        return {"linearity": 0.0, "n_components": 0, "mean_size": 0.0, "largest_size": 0}

    idx = np.arange(1, n + 1)
    ys, xs = np.indices(mask.shape)
    xs = xs.astype(float)
    ys = ys.astype(float)
    counts = np.bincount(lab.ravel())[1:].astype(float)  # pixels per component
    sx = sum_labels(xs, lab, idx)
    sy = sum_labels(ys, lab, idx)
    sxx = sum_labels(xs * xs, lab, idx)
    syy = sum_labels(ys * ys, lab, idx)
    sxy = sum_labels(xs * ys, lab, idx)

    mx, my = sx / counts, sy / counts
    cxx = sxx / counts - mx * mx          # coordinate covariance per component
    cyy = syy / counts - my * my
    cxy = sxy / counts - mx * my
    tr = cxx + cyy
    disc = np.sqrt(np.maximum(0.25 * tr * tr - (cxx * cyy - cxy * cxy), 0.0))
    l1, l2 = 0.5 * tr + disc, 0.5 * tr - disc
    with np.errstate(invalid="ignore", divide="ignore"):
        elong = np.where(l1 + l2 > 0, (l1 - l2) / (l1 + l2), 0.0)
    elong = np.where(counts >= min_pixels, elong, 0.0)  # tiny specks are not lines

    linearity = float(np.sum(elong * counts) / counts.sum())
    return {
        "linearity": linearity,
        "n_components": int(n),
        "mean_size": float(counts.mean()),
        "largest_size": int(counts.max()),
    }


def linearity_at_budget(response, budget: float = 0.05, valid_mask=None,
                        min_pixels: int = 5) -> float:
    """Linearity of the detections obtained by thresholding ``response`` at ``budget``.

    Parallel to ``recall_at_budget`` but label-free: it rewards methods whose
    flagged pixels form long connected lines over methods that scatter blobs.
    """
    det = detect_at_budget(response, budget, valid_mask=valid_mask)
    return linearity_metrics(det, min_pixels=min_pixels)["linearity"]
