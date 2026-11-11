"""from_nifti must reproduce the NIfTI geometry in either axes order."""

import nibabel as nib
import numpy as np
import pytest

from zarrnii import ZarrNii

AFFINES = {
    # anisotropic voxels and a different offset on every axis
    "RAS": np.array(
        [[0.1, 0, 0, -5.0], [0, 0.2, 0, -7.0], [0, 0, 0.3, -2.0], [0, 0, 0, 1]]
    ),
    # voxel axes permuted and flipped relative to world axes
    "PIR": np.array(
        [[0, 0, -0.3, 4.0], [-0.1, 0, 0, 6.0], [0, -0.2, 0, -3.0], [0, 0, 0, 1]]
    ),
}
SHAPE = (10, 12, 14)
MARK = (2, 7, 11)  # voxel (i, j, k) with a distinct value


@pytest.fixture(params=list(AFFINES))
def nifti(request, tmp_path):
    data = np.zeros(SHAPE, dtype=np.float32)
    data[MARK] = 1.0
    aff = AFFINES[request.param]
    path = tmp_path / f"{request.param}.nii.gz"
    nib.Nifti1Image(data, aff).to_filename(path)
    return str(path), aff


def _marked_world(znii):
    """World (x, y, z) position of the marked voxel in a ZarrNii."""
    idx = np.argwhere(np.asarray(znii.data[0]) == 1.0)
    assert len(idx) == 1
    aff = znii.get_affine_matrix()  # data-order voxel -> world
    return aff[:3, :3] @ idx[0] + aff[:3, 3]


@pytest.mark.parametrize("axes_order", ["XYZ", "ZYX"])
def test_affine_matches_nifti(nifti, axes_order):
    path, aff = nifti
    znii = ZarrNii.from_nifti(path, axes_order=axes_order)
    np.testing.assert_allclose(znii.get_affine_matrix("XYZ"), aff, atol=1e-6)


@pytest.mark.parametrize("axes_order", ["XYZ", "ZYX"])
def test_voxel_world_position(nifti, axes_order):
    path, aff = nifti
    znii = ZarrNii.from_nifti(path, axes_order=axes_order)
    expected = aff[:3, :3] @ np.array(MARK) + aff[:3, 3]
    np.testing.assert_allclose(_marked_world(znii), expected, atol=1e-6)


@pytest.mark.parametrize("axes_order", ["XYZ", "ZYX"])
def test_nifti_round_trip(nifti, axes_order, tmp_path):
    path, aff = nifti
    znii = ZarrNii.from_nifti(path, axes_order=axes_order)
    out = tmp_path / "out.nii.gz"
    znii.to_nifti(str(out))
    img = nib.load(out)
    np.testing.assert_allclose(img.affine, aff, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(img.dataobj), nib.load(path).get_fdata())
