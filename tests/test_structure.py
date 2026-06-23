"""Tests for the multi-channel structure-tensor linear-feature response.

The geologically important checks: a straight linear feature must out-score an
isotropic mound (the contact-vs-eroded-mound problem), and accumulating aligned
channels must reinforce while disagreeing orientations must cancel (the
DEM + Sentinel fusion mechanism). Pure numpy/scipy.
"""

import numpy as np
import pytest

from planesight.core.detect import (
    linear_response,
    structure_tensor,
    tensor_eigenvalues,
    tensor_response,
)


def _vertical_edge(n=60, col=None):
    col = n // 2 if col is None else col
    img = np.zeros((n, n))
    img[:, col:] = 1.0
    return img


def _gaussian_bump(n=60, sigma=3.0):
    y, x = np.mgrid[0:n, 0:n]
    c = n / 2.0
    return np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * sigma ** 2))


def _interior(a, m=6):
    return a[m:-m, m:-m]


def test_flat_input_has_near_zero_response():
    flat = np.full((40, 40), 0.7)
    resp = linear_response(flat, kind="anisotropy")
    assert np.nanmax(np.abs(resp)) < 1e-9


def test_eigenvalues_are_ordered_and_nonnegative():
    jxx, jxy, jyy, _ = structure_tensor(_vertical_edge(), sigma_d=1.0, sigma_i=3.0)
    l1, l2 = tensor_eigenvalues(jxx, jxy, jyy)
    assert np.all(l1 + 1e-9 >= l2)
    assert np.all(l2 >= -1e-9)


def test_straight_edge_is_highly_coherent():
    resp = linear_response(_vertical_edge(), kind="coherence")
    assert np.nanmax(_interior(resp)) > 0.9  # a clean line -> coherence near 1


def _feature_mean_coherence(img, sigma_i, frac=0.02):
    """Mean coherence over the highest-energy ``frac`` of pixels (the feature).

    This is the discriminator that matters: a mound's flank is *locally* straight,
    so its single best pixel is coherent - but averaged over the whole feature the
    rotating gradient field drags the mean down, while a line stays ~1 everywhere.
    """
    jxx, jxy, jyy, _ = structure_tensor(img, sigma_d=1.0, sigma_i=sigma_i)
    l1, l2 = tensor_eigenvalues(jxx, jxy, jyy)
    coh = (l1 - l2) / (l1 + l2 + 1e-12)
    sel = l1 >= np.nanpercentile(l1, 100 * (1 - frac))
    return float(np.nanmean(coh[sel]))


def test_line_outscores_isotropic_mound_and_noise():
    # The contact-vs-mound discriminator: averaged over the feature, a straight
    # line is far more coherent than a radial mound or random noise. (At the
    # integration scale the mound's all-directions gradients average to ~0.)
    line = np.zeros((60, 60))
    line[30, :] = 1.0  # a thin horizontal line
    bump = _gaussian_bump(60, sigma=3.0)
    noise = np.random.default_rng(0).random((60, 60))
    line_coh = _feature_mean_coherence(line, sigma_i=5.0)
    bump_coh = _feature_mean_coherence(bump, sigma_i=5.0)
    noise_coh = _feature_mean_coherence(noise, sigma_i=5.0)
    assert line_coh > bump_coh + 0.3
    assert line_coh > noise_coh + 0.3


def test_aligned_channels_reinforce():
    edge = _vertical_edge()
    single = np.nanmax(_interior(linear_response(edge, kind="anisotropy")))
    double = np.nanmax(_interior(linear_response([edge, edge], kind="anisotropy")))
    assert double > 1.5 * single  # two aligned channels roughly add in energy


def test_disagreeing_orientations_cancel():
    # Vertical edge in channel A, horizontal edge in channel B. Where only one is
    # present the response is coherent; where they cross, orientation disagreement
    # drives coherence down - the fusion 'veto' mechanism.
    a = np.zeros((60, 60))
    a[:, 30:] = 1.0   # vertical edge at col 30
    b = np.zeros((60, 60))
    b[30:, :] = 1.0   # horizontal edge at row 30
    coh = linear_response([a, b], kind="coherence")
    single_edge = float(np.nanmean(coh[8:16, 28:33]))   # only the vertical edge
    crossing = float(np.nanmean(coh[28:33, 28:33]))      # both edges meet
    assert single_edge > crossing + 0.1


def test_nan_propagates_to_response():
    edge = _vertical_edge()
    edge[10, 10] = np.nan
    resp = linear_response(edge)
    assert np.isnan(resp[10, 10])


def test_accepts_2d_list_and_cube():
    edge = _vertical_edge(40)
    r_2d = linear_response(edge)
    r_list = linear_response([edge])
    r_cube = linear_response(edge[..., None])  # (H, W, 1)
    assert r_2d.shape == (40, 40)
    assert np.allclose(np.nan_to_num(r_2d), np.nan_to_num(r_list))
    assert np.allclose(np.nan_to_num(r_2d), np.nan_to_num(r_cube))


def test_invalid_kind_and_shapes_raise():
    edge = _vertical_edge(20)
    with pytest.raises(ValueError):
        tensor_response(*structure_tensor(edge)[:3], kind="bogus")
    with pytest.raises(ValueError):
        structure_tensor([np.zeros((4, 4)), np.zeros((4, 5))])
    with pytest.raises(ValueError):
        structure_tensor(edge, weights=[1.0, 2.0])  # weight count mismatch
