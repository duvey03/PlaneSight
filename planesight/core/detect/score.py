"""Scoring detector/derivative output against INCOMPLETE ground truth.

The hand-drawn traces are a small, non-exhaustive subset of the truly mappable
contacts (user constraint). That makes precision/F1 misleading: a detection far
from any drawn line may be a real, un-digitised contact, not a false positive.
So scoring here is deliberately *positive-unlabeled* (ARCHITECTURE.md S6 /
issues planesight-bc1, planesight-c3r):

  - Rank candidates on RECALL of the drawn subset (with a tolerance buffer for
    registration slop), never on precision.
  - Control for detection volume: recall alone is gamed by flagging everything,
    so compare at an equal "ink budget" (a fixed fraction of valid pixels
    flagged) or over a recall-vs-budget curve.

Pure numpy/scipy (decision D11). Inputs are arrays: a continuous ``response``
(higher = more contact-like) and a boolean ``truth_mask`` (drawn traces
rasterised onto the same grid). Rasterisation itself needs GDAL and lives at the
I/O edge, not here, so this module stays CI-testable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

__all__ = [
    "disk",
    "detect_at_budget",
    "recall_at_budget",
    "recall_curve",
]


def disk(radius: int):
    """Boolean disk structuring element of the given pixel radius (radius >= 0)."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _valid(response, valid_mask):
    finite = np.isfinite(response)
    return finite if valid_mask is None else finite & np.asarray(valid_mask, dtype=bool)


def detect_at_budget(response, budget: float, valid_mask=None):
    """Flag the highest-response ``budget`` fraction of valid pixels.

    Returns a boolean detection mask with (close to) ``budget * n_valid`` pixels
    set - the equal-ink control that makes methods comparable. Selection is
    rank-based (top-k), not a percentile threshold, so a response with many tied
    values (e.g. flat zero background) cannot flood the scene; ties at the cutoff
    are broken arbitrarily but the flagged count stays at the budget.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    resp = np.asarray(response, dtype=float)
    valid = _valid(resp, valid_mask)
    det = np.zeros(resp.shape, dtype=bool)
    pos = np.flatnonzero(valid.ravel())
    n = pos.size
    if n == 0:
        return det
    k = min(n, max(1, int(round(budget * n))))
    vals = resp.ravel()[pos]
    if k >= n:
        sel = pos
    else:
        top = np.argpartition(vals, n - k)[n - k:]  # indices of the k largest
        sel = pos[top]
    det.ravel()[sel] = True
    return det


def recall_at_budget(response, truth_mask, budget: float = 0.05,
                     tolerance_px: int = 1, valid_mask=None) -> float:
    """Fraction of drawn-trace pixels recovered at a fixed detection budget.

    A truth pixel counts as recovered if any detection lies within
    ``tolerance_px`` of it (registration buffer). Returns NaN if there is no
    truth pixel inside the valid region.

    This is the headline, volume-controlled completeness metric: "of the contacts
    we *know* are real, how many did this band/detector recover while flagging
    only ``budget`` of the scene?"
    """
    truth = np.asarray(truth_mask, dtype=bool)
    resp = np.asarray(response, dtype=float)
    if truth.shape != resp.shape:
        raise ValueError("response and truth_mask must have the same shape")
    valid = _valid(resp, valid_mask)
    truth = truth & valid
    n_truth = int(truth.sum())
    if n_truth == 0:
        return float("nan")
    det = detect_at_budget(resp, budget, valid_mask=valid_mask)
    reached = binary_dilation(det, structure=disk(tolerance_px))
    return float((truth & reached).sum()) / n_truth


def recall_curve(response, truth_mask, budgets=(0.01, 0.02, 0.05, 0.1, 0.2),
                 tolerance_px: int = 1, valid_mask=None):
    """Recall at each budget - the recall-vs-ink-budget curve for one candidate.

    Returns a list of ``(budget, recall)`` pairs. Comparing curves (not single
    points) is the fair way to rank bands/detectors under incomplete labels: a
    method that reaches high recall at a *small* budget is genuinely sharper, not
    just noisier.
    """
    return [
        (float(b), recall_at_budget(response, truth_mask, budget=b,
                                    tolerance_px=tolerance_px, valid_mask=valid_mask))
        for b in budgets
    ]
