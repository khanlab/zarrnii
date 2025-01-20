from __future__ import annotations

from abc import ABC, abstractmethod

import fsspec
import nibabel as nib
import numpy as np
from attrs import define
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from scipy.interpolate import interpn


@define
class Transform(ABC):
    """Base class for transformations"""

    @abstractmethod
    def apply_transform(self, vecs: np.array) -> np.array:
        """Apply transformation to an image"""

        pass


@define
class AffineTransform(Transform):

    matrix: np.array = None

    @classmethod
    def from_txt(cls, path, invert=False):
        matrix = np.loadtxt(path)
        if invert:
            matrix = np.linalg.inv(matrix)

        return cls(matrix=matrix)

    @classmethod
    def from_array(cls, matrix, invert=False):
        if invert:
            matrix = np.linalg.inv(matrix)

        return cls(matrix=matrix)

    @classmethod
    def identity(cls):
        return cls(matrix=np.eye(4, 4))

    def __array__(self):
        """
        Define how the object behaves when converted to a numpy array.
        Returns the matrix of the affine transform.
        """
        return self.matrix

    def __getitem__(self, key):
        """
        Enable array-like indexing on the matrix.
        """
        return self.matrix[key]

    def __setitem__(self, key, value):
        """
        Enable array-like assignment to the matrix.
        """
        self.matrix[key] = value

    def __matmul__(self, other):

        if isinstance(other, np.ndarray):
            if other.shape == (3,) or other.shape == (3, 1):
                # Convert 3D point/vector to homogeneous coordinates
                homog_point = np.append(other, 1)
                result = self.matrix @ homog_point
                # Convert back from homogeneous coordinates to 3D
                return result[:3] / result[3]
            elif other.shape == (4,) or other.shape == (4, 1):
                # Directly use 4D point/vector
                result = self.matrix @ other
                # Convert back from homogeneous coordinates to 3D
                return result[:3] / result[3]
            elif other.shape == (4, 4):
                # perform matrix multiplication, and return a Transform object
                return AffineTransform.from_array(self.matrix @ other)
            else:
                raise ValueError("Unsupported shape for multiplication.")
        else:
            raise TypeError("Unsupported type for multiplication.")

    def apply_transform(self, vecs: np.array) -> np.array:
        return self.matrix @ vecs

    def invert(self):
        """Return the inverse of the matrix transformation."""
        return AffineTransform.from_array(np.linalg.inv(self.matrix))

    def update_for_orientation(self, input_orientation, output_orientation):
        """
        Update the matrix to map from the input orientation to the output orientation.

        Parameters:
            input_orientation (str): Current anatomical orientation (e.g., 'RPI').
            output_orientation (str): Target anatomical orientation (e.g., 'RAS').
        """

        # Define a mapping of anatomical directions to axis indices and flips
        axis_map = {
            "R": (0, 1),
            "L": (0, -1),
            "A": (1, 1),
            "P": (1, -1),
            "S": (2, 1),
            "I": (2, -1),
        }

        # Parse the input and output orientations
        input_axes = [axis_map[ax] for ax in input_orientation]
        output_axes = [axis_map[ax] for ax in output_orientation]

        # Create a mapping from input to output
        reorder_indices = [None] * 3
        flip_signs = [1] * 3

        for out_idx, (out_axis, out_sign) in enumerate(output_axes):
            for in_idx, (in_axis, in_sign) in enumerate(input_axes):
                if out_axis == in_axis:  # Match axis
                    reorder_indices[out_idx] = in_idx
                    flip_signs[out_idx] = out_sign * in_sign
                    break

        # Reorder and flip the affine matrix
        reordered_matrix = np.zeros_like(self.matrix)
        for i, (reorder_idx, flip_sign) in enumerate(zip(reorder_indices, flip_signs)):
            if reorder_idx is None:
                raise ValueError(
                    f"Cannot match all axes from {input_orientation} to {output_orientation}."
                )
            reordered_matrix[i, :3] = flip_sign * self.matrix[reorder_idx, :3]
            reordered_matrix[i, 3] = flip_sign * self.matrix[reorder_idx, 3]
        reordered_matrix[3, :] = self.matrix[3, :]  # Preserve the homogeneous row

        return AffineTransform.from_array(reordered_matrix)


@define
class DisplacementTransform(Transform):

    disp_xyz: np.array = None
    disp_grid: np.array = None
    disp_affine: AffineTransform = None

    @classmethod
    def from_nifti(cls, path):
        disp_nib = nib.load(path)
        disp_xyz = disp_nib.get_fdata().squeeze()
        disp_affine = AffineTransform.from_array(disp_nib.affine)

        # convert from itk transform
        disp_xyz[:, :, :, 0] = -disp_xyz[:, :, :, 0]
        disp_xyz[:, :, :, 1] = -disp_xyz[:, :, :, 1]

        disp_grid = (
            np.arange(disp_xyz.shape[0]),
            np.arange(disp_xyz.shape[1]),
            np.arange(disp_xyz.shape[2]),
        )

        return cls(
            disp_xyz=disp_xyz,
            disp_grid=disp_grid,
            disp_affine=disp_affine,
        )

    def apply_transform(self, vecs: np.array) -> np.array:

        # we have the grid points, the volumes to interpolate displacements

        # first we need to transform points to vox space of the warp
        vox_vecs = self.disp_affine.invert() @ vecs

        # then interpolate the displacement in x, y, z:
        disp_vecs = np.zeros(vox_vecs.shape)

        for ax in range(3):
            disp_vecs[ax, :] = interpn(
                self.disp_grid,
                self.disp_xyz[:, :, :, ax].squeeze(),
                vox_vecs[:3, :].T,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )

        return vecs + disp_vecs
