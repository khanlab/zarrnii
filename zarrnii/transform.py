"""Transformation classes for spatial transformations in ZarrNii."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import nibabel as nib
import numpy as np
from attrs import define
from scipy.interpolate import interpn


@define
class Transform(ABC):
    """Abstract base class for spatial transformations.

    This class defines the interface that all transformation classes must implement
    to be used with ZarrNii. Transformations convert coordinates from one space
    to another (e.g., subject space to template space).
    """

    @abstractmethod
    def apply_transform(self, vecs: np.ndarray) -> np.ndarray:
        """Apply transformation to coordinate vectors.

        Args:
            vecs: Input coordinates as numpy array. Shape can be:
                - (3,) for single 3D point
                - (3, N) for N 3D points
                - (4,) for single homogeneous coordinate
                - (4, N) for N homogeneous coordinates

        Returns:
            Transformed coordinates with same shape as input

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass


@define
class AffineTransform(Transform):
    """Affine transformation for spatial coordinate mapping.

    Represents a 4x4 affine transformation matrix that can be used to transform
    3D coordinates between different coordinate systems. Supports various
    operations including matrix multiplication, inversion, and point transformation.

    Attributes:
        matrix: 4x4 affine transformation matrix
    """

    matrix: np.ndarray = None

    @classmethod
    def from_txt(
        cls, path: Union[str, bytes], invert: bool = False
    ) -> "AffineTransform":
        """Create AffineTransform from text file containing matrix.

        Args:
            path: Path to text file containing 4x4 affine matrix
            invert: Whether to invert the matrix after loading

        Returns:
            AffineTransform instance with loaded matrix

        Raises:
            OSError: If file cannot be read
            ValueError: If file does not contain valid 4x4 matrix
        """
        matrix = np.loadtxt(path)
        if invert:
            matrix = np.linalg.inv(matrix)
        return cls(matrix=matrix)

    @classmethod
    def from_array(cls, matrix: np.ndarray, invert: bool = False) -> "AffineTransform":
        """Create AffineTransform from numpy array.

        Args:
            matrix: 4x4 numpy array representing affine transformation
            invert: Whether to invert the matrix

        Returns:
            AffineTransform instance with the matrix

        Raises:
            ValueError: If matrix is not 4x4
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")

        if invert:
            matrix = np.linalg.inv(matrix)
        return cls(matrix=matrix)

    @classmethod
    def identity(cls) -> "AffineTransform":
        """Create identity transformation.

        Returns:
            AffineTransform representing identity transformation (no change)
        """
        return cls(matrix=np.eye(4, 4))

    def __array__(self) -> np.ndarray:
        """Convert to numpy array.

        Defines how the object behaves when converted to a numpy array.

        Returns:
            The 4x4 affine transformation matrix
        """
        return self.matrix

    def __getitem__(self, key) -> Union[np.ndarray, float]:
        """Enable array-like indexing on the matrix.

        Args:
            key: Index or slice for matrix access

        Returns:
            Element(s) from the transformation matrix
        """
        return self.matrix[key]

    def __setitem__(self, key, value: Union[float, np.ndarray]) -> None:
        """Enable array-like assignment to the matrix.

        Args:
            key: Index or slice for matrix assignment
            value: Value(s) to assign to matrix
        """
        self.matrix[key] = value

    def __matmul__(
        self, other: Union[np.ndarray, "AffineTransform"]
    ) -> Union[np.ndarray, "AffineTransform"]:
        """Perform matrix multiplication with another object.

        Args:
            other: The object to multiply with. Supported types:
                - (3,) or (3, 1): A 3D point or vector (voxel coordinates)
                - (3, N): A batch of N 3D points or vectors (voxel coordinates)
                - (4,) or (4, 1): A 4D point/vector in homogeneous coordinates
                - (4, N): A batch of N 4D points in homogeneous coordinates
                - (4, 4): Another affine transformation matrix
                - AffineTransform: Another affine transformation object

        Returns:
            Transformed coordinates as numpy array or new AffineTransform:
                - For coordinate inputs: transformed coordinates with same shape
                - For matrix/AffineTransform inputs: new AffineTransform object

        Raises:
            ValueError: If the shape of `other` is unsupported
            TypeError: If `other` is not an np.ndarray or AffineTransform
        """
        if isinstance(other, np.ndarray):
            if other.shape == (3,):
                # Single 3D point/vector
                homog_point = np.append(other, 1)  # Convert to homogeneous coordinates
                result = self.matrix @ homog_point
                return result[:3] / result[3]  # Convert back to 3D
            elif len(other.shape) == 2 and other.shape[0] == 3:
                # Batch of 3D points/vectors (3 x N)
                homog_points = np.vstack(
                    [other, np.ones((1, other.shape[1]))]
                )  # Add homogeneous row
                transformed_points = (
                    self.matrix @ homog_points
                )  # Apply affine transform
                return (
                    transformed_points[:3] / transformed_points[3]
                )  # Convert back to 3D
            elif other.shape == (4,):
                # Single 4D point/vector
                result = self.matrix @ other
                return result[:3] / result[3]
            elif len(other.shape) == 2 and other.shape[0] == 4:
                # Batch of 4D points in homogeneous coordinates (4 x N)
                transformed_points = self.matrix @ other  # Apply affine transform
                return transformed_points  # No conversion needed, stays in 4D space
            elif other.shape == (4, 4):
                # Matrix multiplication with another affine matrix
                return AffineTransform.from_array(self.matrix @ other)
            else:
                raise ValueError(f"Unsupported shape for multiplication: {other.shape}")
        elif isinstance(other, AffineTransform):
            # Matrix multiplication with another AffineTransform object
            return AffineTransform.from_array(self.matrix @ other.matrix)
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")

    def apply_transform(self, vecs: np.ndarray) -> np.ndarray:
        """Apply transformation to coordinate vectors.

        Args:
            vecs: Input coordinates to transform

        Returns:
            Transformed coordinates
        """
        return self @ vecs

    def invert(self) -> "AffineTransform":
        """Return the inverse of the matrix transformation.

        Returns:
            New AffineTransform with inverted matrix

        Raises:
            np.linalg.LinAlgError: If matrix is singular and cannot be inverted
        """
        return AffineTransform.from_array(np.linalg.inv(self.matrix))

    def update_for_orientation(
        self, input_orientation: str, output_orientation: str
    ) -> "AffineTransform":
        """Update the matrix to map from input orientation to output orientation.

        Args:
            input_orientation: Current anatomical orientation (e.g., 'RPI')
            output_orientation: Target anatomical orientation (e.g., 'RAS')

        Returns:
            New AffineTransform updated for orientation mapping

        Raises:
            ValueError: If orientations are invalid or cannot be matched
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
    """Non-linear displacement field transformation.

    Represents a displacement field transformation where each point in space
    has an associated displacement vector. Uses interpolation to compute
    displacements for arbitrary coordinates.

    Attributes:
        disp_xyz: Displacement vectors at grid points (4D array: x, y, z, vector_component)
        disp_grid: Grid coordinates for displacement field
        disp_affine: Affine transformation from world to displacement field coordinates
    """

    disp_xyz: np.ndarray = None
    disp_grid: Tuple[np.ndarray, ...] = None
    disp_affine: AffineTransform = None

    @classmethod
    def from_nifti(cls, path: Union[str, bytes]) -> "DisplacementTransform":
        """Create DisplacementTransform from NIfTI file.

        Args:
            path: Path to NIfTI displacement field file

        Returns:
            DisplacementTransform instance loaded from file

        Raises:
            OSError: If file cannot be read
            ValueError: If file format is invalid
        """
        disp_nib = nib.load(path)
        disp_xyz = disp_nib.get_fdata().squeeze()
        disp_affine = AffineTransform.from_array(disp_nib.affine)

        # Convert from ITK transform convention
        # ITK uses opposite sign convention for x and y displacements
        disp_xyz[:, :, :, 0] = -disp_xyz[:, :, :, 0]
        disp_xyz[:, :, :, 1] = -disp_xyz[:, :, :, 1]

        # Create grid coordinates
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

    def apply_transform(self, vecs: np.ndarray) -> np.ndarray:
        """Apply displacement transformation to coordinate vectors.

        Transforms input coordinates by interpolating displacement vectors
        from the displacement field and adding them to the input coordinates.

        Args:
            vecs: Input coordinates as numpy array. Shape should be (3, N) for
                N points or (3,) for single point

        Returns:
            Transformed coordinates with same shape as input

        Notes:
            Points outside the displacement field domain are filled with
            zero displacement (no transformation).
        """
        # Transform points to voxel space of the displacement field
        vox_vecs = self.disp_affine.invert() @ vecs

        # Initialize displacement vectors
        disp_vecs = np.zeros(vox_vecs.shape)

        # Interpolate displacement for each spatial dimension (x, y, z)
        for ax in range(3):
            disp_vecs[ax, :] = interpn(
                self.disp_grid,
                self.disp_xyz[:, :, :, ax].squeeze(),
                vox_vecs[:3, :].T,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )

        # Add displacement to original coordinates
        return vecs + disp_vecs
