"""Tests for Sentinel-2 spectral derivatives and stack assembly (pure numpy)."""

import numpy as np
import pytest

from planesight.core.derivatives import (
    assemble,
    build_spectral_stack,
    build_terrain_stack,
    normalize01,
    spectral,
)


def test_ratio_exact_and_divide_safe():
    out = spectral.ratio([2.0, 4.0, 6.0], [1.0, 2.0, 0.0])
    assert np.allclose(out[:2], [2.0, 2.0])
    assert np.isnan(out[2])  # divide by zero -> NaN, not inf


def test_normalized_ratio_value_and_range():
    out = spectral.normalized_ratio([3.0, 1.0], [1.0, 1.0])
    assert np.allclose(out, [0.5, 0.0])
    rng = np.random.default_rng(0)
    a, b = rng.random(100), rng.random(100)
    nr = spectral.normalized_ratio(a, b)
    assert nr.min() >= -1.0 and nr.max() <= 1.0


def test_normalized_ratio_zero_sum_is_nan():
    out = spectral.normalized_ratio([0.0], [0.0])
    assert np.isnan(out[0])


def test_ndvi_is_normalized_nir_red():
    nir, red = np.array([0.4]), np.array([0.1])
    assert np.allclose(spectral.ndvi(nir, red), (0.4 - 0.1) / (0.4 + 0.1))


def test_indices_use_expected_bands():
    one = np.ones((3, 3))
    assert np.allclose(spectral.clay_index(2 * one, one), 2.0)         # swir16/swir22
    assert np.allclose(spectral.iron_oxide_index(3 * one, one), 3.0)   # red/blue
    assert np.allclose(spectral.ferrous_index(4 * one, one), 4.0)      # swir16/nir


def test_percentile_stretch_maps_to_unit_and_clips():
    band = np.linspace(0.0, 1.0, 101)
    s = spectral.percentile_stretch(band, lo=2.0, hi=98.0)
    assert s.min() == 0.0 and s.max() == 1.0
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_percentile_stretch_handles_all_nan_and_constant():
    allnan = np.full(10, np.nan)
    assert np.all(np.isnan(spectral.percentile_stretch(allnan)))
    const = np.full((4, 4), 7.0)
    out = spectral.percentile_stretch(const)
    assert np.allclose(out, 0.0)


def test_percentile_stretch_preserves_nan():
    band = np.array([0.0, 0.5, np.nan, 1.0])
    s = spectral.percentile_stretch(band)
    assert np.isnan(s[2]) and np.isfinite(s[0])


# --- stack assembly ---

def _s2_bands(shape=(8, 8)):
    rng = np.random.default_rng(2)
    keys = ("blue", "green", "red", "nir", "swir16", "swir22")
    return {k: rng.random(shape) + 0.1 for k in keys}


def test_build_terrain_stack_returns_requested_names():
    dem = np.cumsum(np.ones((12, 12)), axis=0)
    names, layers = build_terrain_stack(dem, px=30.0, names=("slope", "tpi"))
    assert names == ("slope", "tpi")
    assert set(layers) == {"slope", "tpi"}
    assert layers["slope"].shape == dem.shape


def test_build_spectral_stack_and_missing_band():
    bands = _s2_bands()
    names, layers = build_spectral_stack(bands, names=("clay", "ndvi"))
    assert set(layers) == {"clay", "ndvi"}
    # ferrous needs nir+swir16 (present); a band index missing its input raises.
    with pytest.raises(KeyError):
        build_spectral_stack({"red": np.ones((4, 4))}, names=("clay",))


def test_unknown_band_name_raises():
    with pytest.raises(KeyError):
        build_terrain_stack(np.zeros((5, 5)), px=30.0, names=("nope",))


def test_assemble_stacks_and_shares_nan_mask():
    a = np.ones((6, 6))
    b = np.ones((6, 6)) * 2.0
    b[0, 0] = np.nan
    names, cube = assemble({"a": a, "b": b}, normalize=False)
    assert names == ["a", "b"]
    assert cube.shape == (6, 6, 2)
    # NaN in band b at (0,0) nulls the whole pixel across bands
    assert np.all(np.isnan(cube[0, 0, :]))
    assert np.all(np.isfinite(cube[1, 1, :]))


def test_assemble_normalizes_each_band():
    a = np.linspace(0, 100, 36).reshape(6, 6)
    names, cube = assemble({"a": a}, normalize=True)
    assert np.nanmin(cube) >= 0.0 and np.nanmax(cube) <= 1.0


def test_assemble_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        assemble({"a": np.ones((4, 4)), "b": np.ones((4, 5))})


def test_normalize01_alias():
    band = np.linspace(0, 10, 50)
    assert np.allclose(normalize01(band), spectral.percentile_stretch(band))
