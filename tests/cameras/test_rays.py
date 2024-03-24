"""
Test the ray classes.
"""

import pytest
import torch

from nerfstudio.cameras.rays import Frustums


def test_frustum_get_position():
    """Test position calculation"""

    origin = torch.tensor([0., 1., 2.])[None, ...]
    direction = torch.tensor([0., 1., 0.])[None, ...]
    frustum_start = torch.tensor([2.])[None, ...]
    frustum_end = torch.tensor([3.])[None, ...]

    target_position = torch.tensor([0., 3.5, 2.])[None, ...]

    frustum = Frustums(
        origins=origin,
        directions=direction,
        starts=frustum_start,
        ends=frustum_end,
        radii=torch.ones((1, 1)),
    )

    positions = frustum.get_positions()
    assert positions == pytest.approx(target_position, abs=1e-6)


def test_frustum_get_gaussian_blob():
    """Test gaussian blob calculation"""

    frustum = Frustums(
        origins=torch.ones((5, 3)),
        directions=torch.ones((5, 3)),
        starts=torch.ones((5, 1)),
        ends=torch.ones((5, 1)),
        radii=torch.ones((5, 1)),
    )

    gaussian_blob = frustum.get_conical_gaussian_blob()
    assert gaussian_blob.mean.shape == (5, 3)
    assert gaussian_blob.cov.shape == (5, 3, 3)


def test_frustum_apply_masks():
    """Test masking frustum"""
    frustum = Frustums(
        origins=torch.ones((5, 3)),
        directions=torch.ones((5, 3)),
        starts=torch.ones((5, 1)),
        ends=torch.ones((5, 1)),
        radii=torch.ones((5, 1)),
    )

    mask = torch.tensor([False, True, False, True, True], dtype=torch.bool)
    frustum = frustum[mask]

    assert frustum.origins.shape == (3, 3)
    assert frustum.directions.shape == (3, 3)
    assert frustum.starts.shape == (3, 1)
    assert frustum.ends.shape == (3, 1)
    assert frustum.radii.shape == (3, 1)


def test_get_mock_frustum():
    """Test creation of mock frustum"""
    Frustums.get_mock_frustum()


if __name__ == "__main__":
    test_frustum_get_gaussian_blob()
