"""The attitude engine is a scaffold stub until Phase 3; pin its contract."""

import pytest

from planesight.core.attitude import Attitude, fit_plane


def test_fit_plane_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        fit_plane([(0, 0, 0), (1, 0, 1), (0, 1, 1)])


def test_attitude_dataclass_fields():
    a = Attitude(
        strike=10.0,
        dip=20.0,
        dip_direction=100.0,
        conditioning=0.5,
        planarity=0.01,
        residual_rms=1.2,
        relief=50.0,
        n_samples=12,
    )
    assert a.dip == 20.0
    assert a.conditioning == 0.5
