"""Multi-channel structure-tensor linear-feature response (pure numpy/scipy).

The contact-vs-mound problem (ARCHITECTURE.md S8): a geological contact is a
STRONG, ORIENTED, LATERALLY-CONTINUOUS line, ideally corroborated by a spectral
change across it; a smoothed erosional mound is an isotropic slope break with no
lithological signal. Plain gradient magnitude cannot tell them apart - it fires on
any local contrast. The structure tensor can:

  - Per pixel it accumulates local gradient ORIENTATION, then measures how
    consistent that orientation is. Elongated linear features score high; isotropic
    blobs, corners, and noise score low.
  - Accumulating the tensor across MULTIPLE co-registered channels (DEM
    derivatives AND Sentinel-2 bands) before the eigen-decomposition fuses the
    modalities: a line present in both topography and spectra reinforces, while a
    topographic edge with no spectral support (the eroded mound) is comparatively
    weak. This is "edge detection supported by Sentinel data" with no hand-tuned
    fusion rule - orientation agreement falls out of the tensor sum.

Caveat: the structure tensor is a LOCAL operator - it rewards local linearity, so a
large, gently-curved mound whose flank is locally straight is not rejected by this
alone. Global continuity (a Hough/elongation pass) and the downstream plane-fit
planarity/conditioning guard are the complementary filters. Pure numpy/scipy (D11).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "structure_tensor",
    "tensor_eigenvalues",
    "tensor_response",
    "linear_response",
]


def _as_channels(channels):
    """Normalize input to a list of 2D float arrays (accepts one 2D array, a list,
    or a 3D (H, W, C) cube)."""
    arr = channels
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2:
            return [arr.astype(float)]
        if arr.ndim == 3:  # (H, W, C)
            return [arr[..., c].astype(float) for c in range(arr.shape[-1])]
        raise ValueError("array channels must be 2D or 3D (H, W, C)")
    chans = [np.asarray(c, dtype=float) for c in channels]
    if not chans:
        raise ValueError("need at least one channel")
    shape = chans[0].shape
    if any(c.shape != shape or c.ndim != 2 for c in chans):
        raise ValueError("all channels must be 2D and the same shape")
    return chans


def structure_tensor(channels, sigma_d: float = 1.0, sigma_i: float = 3.0, weights=None):
    """Accumulated, smoothed structure tensor over one or more channels.

    Args:
        channels: a 2D array, list of 2D arrays, or 3D (H, W, C) cube - all on the
            same grid. Channels should be on a comparable scale (e.g. normalized).
        sigma_d: Gaussian derivative scale (px) - the edge scale.
        sigma_i: Gaussian integration scale (px) - the neighbourhood over which
            orientation consistency is measured. Should exceed ``sigma_d``.
        weights: optional per-channel weights (default equal).

    Returns:
        ``(Jxx, Jxy, Jyy, valid)`` - the three independent tensor components and a
        boolean mask of pixels finite in every channel. NaNs are filled with the
        channel median for the derivative step, then masked via ``valid``.
    """
    chans = _as_channels(channels)
    w = [1.0] * len(chans) if weights is None else list(weights)
    if len(w) != len(chans):
        raise ValueError("weights must match the number of channels")
    shape = chans[0].shape
    jxx = np.zeros(shape)
    jxy = np.zeros(shape)
    jyy = np.zeros(shape)
    valid = np.ones(shape, dtype=bool)
    for wt, arr in zip(w, chans):
        finite = np.isfinite(arr)
        valid &= finite
        fill = float(np.nanmedian(arr)) if finite.any() else 0.0
        a = np.where(finite, arr, fill)
        gx = gaussian_filter(a, sigma_d, order=(0, 1))  # d/dx (cols)
        gy = gaussian_filter(a, sigma_d, order=(1, 0))  # d/dy (rows)
        jxx += wt * gx * gx
        jxy += wt * gx * gy
        jyy += wt * gy * gy
    jxx = gaussian_filter(jxx, sigma_i)
    jxy = gaussian_filter(jxy, sigma_i)
    jyy = gaussian_filter(jyy, sigma_i)
    return jxx, jxy, jyy, valid


def tensor_eigenvalues(jxx, jxy, jyy):
    """Per-pixel eigenvalues ``(lambda1, lambda2)`` of the 2x2 tensor, l1 >= l2 >= 0.

    ``lambda1`` is the energy across the dominant edge; ``lambda2`` the energy
    along it. ``lambda1 >> lambda2`` means a strongly oriented (linear) feature;
    ``lambda1 ~ lambda2`` means isotropic (blob / flat / noise).
    """
    half_tr = 0.5 * (jxx + jyy)
    disc = np.sqrt(np.maximum((0.5 * (jxx - jyy)) ** 2 + jxy * jxy, 0.0))
    return half_tr + disc, half_tr - disc


def tensor_response(jxx, jxy, jyy, kind: str = "anisotropy", eps: float = 1e-12):
    """Scalar linear-feature response from the tensor components.

    Args:
        kind: ``"anisotropy"`` = lambda1 - lambda2 (oriented edge energy; needs
            both strong contrast AND consistent orientation - the default, best at
            suppressing isotropic mounds); ``"coherence"`` =
            (lambda1 - lambda2)/(lambda1 + lambda2) in [0, 1] (orientation purity,
            contrast-independent, noisier on flats); ``"lambda1"`` = raw dominant
            edge energy (closest to plain gradient magnitude).
    """
    l1, l2 = tensor_eigenvalues(jxx, jxy, jyy)
    if kind == "anisotropy":
        return l1 - l2
    if kind == "coherence":
        return (l1 - l2) / (l1 + l2 + eps)
    if kind == "lambda1":
        return l1
    raise ValueError("kind must be 'anisotropy', 'coherence', or 'lambda1'")


def linear_response(channels, sigma_d: float = 1.0, sigma_i: float = 3.0,
                    kind: str = "anisotropy", weights=None):
    """End-to-end multi-channel linear-feature response (NaN where any channel is).

    Convenience wrapper: build the accumulated structure tensor over ``channels``
    and return the chosen scalar response, with invalid pixels set to NaN so the
    scorer's validity mask excludes them.
    """
    jxx, jxy, jyy, valid = structure_tensor(channels, sigma_d, sigma_i, weights)
    resp = tensor_response(jxx, jxy, jyy, kind=kind)
    return np.where(valid, resp, np.nan)
