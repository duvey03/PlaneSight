"""Geology-aware cost penalties for live-wire tracing (assisted tracing, T1).

The differentiators vs a generic magnetic lasso (spec S6). Each returns a per-pixel
penalty in [0, 1] (high = avoid) that drops straight into
:func:`planesight.core.trace.cost.build_cost_surface`:

- ``drainage_penalty`` keeps the wire off creeks (our #1 contamination mode), reusing the
  detector's flow-accumulation channel network.
- ``orientation_incoherence`` discourages wandering into structureless terrain / onto
  crossing features, reusing the structure tensor (which lost on detection *recall* but is
  ideal for tracing *guidance*).

Pure numpy/scipy (+ existing detect helpers); no GDAL/Qt.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from planesight.core.detect.drainage import channel_network
from planesight.core.detect.structure import structure_tensor, tensor_response


def drainage_penalty(
    dem: np.ndarray,
    *,
    downsample: int = 3,
    min_accum_cells: int = 15,
    decay_px: float = 3.0,
) -> np.ndarray:
    """Creek-avoidance penalty in [0, 1]: 1 on a flow-accumulation channel, decaying off.

    A soft penalty (not a hard barrier) so the wire may still *cross* a creek where a
    contact does, but is pushed off *following* one. Defaults mirror the detector's
    drainage filter (planesight-61f) for consistency.

    Args:
        dem: 2D elevation array.
        downsample, min_accum_cells: channel-network parameters (see
            :func:`planesight.core.detect.drainage.channel_network`).
        decay_px: e-folding distance (px) of the penalty away from a channel.

    Returns:
        Float array, ``dem`` shape, ``exp(-dist_to_channel / decay_px)`` (0 where there
        are no channels at all).
    """
    mask, _flow_az = channel_network(dem, min_accum_cells=min_accum_cells,
                                     downsample=downsample)
    if not mask.any():
        return np.zeros(np.asarray(dem).shape, dtype=float)
    dist = distance_transform_edt(~mask)
    return np.exp(-dist / max(decay_px, 1e-6))


def orientation_incoherence(
    field: np.ndarray,
    *,
    sigma_d: float = 1.0,
    sigma_i: float = 3.0,
) -> np.ndarray:
    """Penalty in [0, 1]: 0 where a strong coherent fabric exists, 1 where isotropic.

    Reuses the structure tensor's orientation *coherence* (purity of local orientation):
    the wire is nudged to follow the dominant grain and away from structureless terrain
    where it would otherwise jump onto a crossing feature.

    Args:
        field: the signal the tensor is computed on (e.g. curvature magnitude or an
            elevation derivative); 2D, or (C, H, W) to combine channels.
        sigma_d, sigma_i: derivative and integration scales of the structure tensor.

    Returns:
        Float array (H, W), ``1 - coherence`` clipped to [0, 1].
    """
    jxx, jxy, jyy, valid = structure_tensor(field, sigma_d=sigma_d, sigma_i=sigma_i)
    coherence = tensor_response(jxx, jxy, jyy, kind="coherence")
    incoherence = np.clip(1.0 - coherence, 0.0, 1.0)
    return np.where(valid, incoherence, 1.0)   # no data -> maximally avoid
