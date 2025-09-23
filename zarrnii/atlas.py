"""Atlas handling module for ZarrNii.

This module provides functionality for working with brain atlases, specifically
BIDS-formatted dseg.nii.gz segmentation images with corresponding dseg.tsv
lookup tables. It enables region-of-interest (ROI) based analysis and
aggregation of data across atlas regions.

Based on functionality from SPIMquant:
https://github.com/khanlab/SPIMquant/blob/main/spimquant/workflow/scripts/
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from attrs import define, field

from .core import ZarrNii


@define
class Atlas:
    """Brain atlas with segmentation image and region lookup table.

    Represents a brain atlas consisting of a segmentation image (dseg) that
    assigns integer labels to brain regions, and a lookup table (tsv) that
    maps these labels to region names and other metadata.

    Attributes:
        dseg (ZarrNii): Segmentation image with integer labels for each region
        labels_df (pd.DataFrame): Lookup table mapping labels to region info
        label_column (str): Column name containing region labels/indices
        name_column (str): Column name containing region names
        abbrev_column (str): Column name containing region abbreviations
    """

    dseg: ZarrNii
    labels_df: pd.DataFrame
    label_column: str = field(default="index")
    name_column: str = field(default="name")
    abbrev_column: str = field(default="abbreviation")

    def __attrs_post_init__(self):
        """Validate atlas consistency after initialization."""
        self._validate_atlas()

    def _validate_atlas(self):
        """Validate that atlas components are consistent."""
        # Check that required columns exist
        required_cols = [self.label_column, self.name_column]
        missing_cols = [
            col for col in required_cols if col not in self.labels_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in labels DataFrame: {missing_cols}"
            )

        # Check that label column contains integers
        if not pd.api.types.is_integer_dtype(self.labels_df[self.label_column]):
            warnings.warn(
                f"Label column '{self.label_column}' should contain integers. "
                "Converting to int.",
                stacklevel=3,
            )
            self.labels_df[self.label_column] = self.labels_df[
                self.label_column
            ].astype(int)

        # Check for duplicate labels
        duplicates = self.labels_df[self.label_column].duplicated()
        if duplicates.any():
            dup_labels = self.labels_df[duplicates][self.label_column].tolist()
            raise ValueError(f"Duplicate labels found in atlas: {dup_labels}")

    @classmethod
    def from_files(
        cls,
        dseg_path: Union[str, Path],
        labels_path: Union[str, Path],
        label_column: str = "index",
        name_column: str = "name",
        abbrev_column: str = "abbreviation",
        **zarrnii_kwargs,
    ) -> "Atlas":
        """Load atlas from dseg image and labels TSV files.

        Args:
            dseg_path: Path to segmentation image (dseg.nii.gz or .ome.zarr)
            labels_path: Path to labels TSV file (dseg.tsv)
            label_column: Column name for region labels/indices
            name_column: Column name for region names
            abbrev_column: Column name for region abbreviations
            **zarrnii_kwargs: Additional arguments passed to ZarrNii.from_path()

        Returns:
            Atlas instance

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If files can't be loaded or are inconsistent
        """
        dseg_path = Path(dseg_path)
        labels_path = Path(labels_path)

        if not dseg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {dseg_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load segmentation image
        if str(dseg_path).endswith(".ome.zarr"):
            dseg = ZarrNii.from_ome_zarr(str(dseg_path), **zarrnii_kwargs)
        else:
            # Assume NIfTI format
            dseg = ZarrNii.from_nifti(str(dseg_path), **zarrnii_kwargs)

        # Load labels DataFrame
        try:
            labels_df = pd.read_csv(labels_path, sep="\t")
        except Exception as e:
            raise ValueError(f"Failed to load labels file {labels_path}: {e}")

        return cls(
            dseg=dseg,
            labels_df=labels_df,
            label_column=label_column,
            name_column=name_column,
            abbrev_column=abbrev_column,
        )

    @property
    def region_labels(self) -> np.ndarray:
        """Get unique region labels from the lookup table."""
        return self.labels_df[self.label_column].values

    @property
    def region_names(self) -> List[str]:
        """Get region names from the lookup table."""
        return self.labels_df[self.name_column].tolist()

    @property
    def region_abbreviations(self) -> List[str]:
        """Get region abbreviations from the lookup table (if available)."""
        if self.abbrev_column in self.labels_df.columns:
            return self.labels_df[self.abbrev_column].tolist()
        else:
            return [""] * len(self.labels_df)

    def _find_region_label(self, identifier: Union[int, str]) -> int:
        """Find region label from identifier (index, name, or abbreviation).

        Args:
            identifier: Region identifier - can be:
                - int: Direct label/index
                - str: Region name or abbreviation

        Returns:
            Region label/index as integer

        Raises:
            ValueError: If identifier not found in atlas
        """
        # If already an integer (including numpy integers), validate and return
        if isinstance(identifier, (int, np.integer)):
            identifier = int(identifier)  # Convert numpy integers to Python int
            if identifier not in self.region_labels:
                raise ValueError(f"Label {identifier} not found in atlas")
            return identifier

        # Search by name first
        name_mask = self.labels_df[self.name_column] == identifier
        if name_mask.any():
            return int(self.labels_df[name_mask].iloc[0][self.label_column])

        # Search by abbreviation if available
        if self.abbrev_column in self.labels_df.columns:
            abbrev_mask = self.labels_df[self.abbrev_column] == identifier
            if abbrev_mask.any():
                return int(self.labels_df[abbrev_mask].iloc[0][self.label_column])

        # Not found
        raise ValueError(
            f"Region '{identifier}' not found in atlas. "
            f"Must be a valid label, name, or abbreviation."
        )

    def get_region_info(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get information for a specific region.

        Args:
            identifier: Region identifier - can be:
                - int: Region label/index
                - str: Region name or abbreviation

        Returns:
            Dictionary with region information

        Raises:
            ValueError: If identifier not found in atlas
        """
        label = self._find_region_label(identifier)
        mask = self.labels_df[self.label_column] == label
        row = self.labels_df[mask].iloc[0]
        return row.to_dict()

    def get_region_mask(self, identifier: Union[int, str]) -> ZarrNii:
        """Get binary mask for a specific region.

        Args:
            identifier: Region identifier - can be:
                - int: Region label/index
                - str: Region name or abbreviation

        Returns:
            ZarrNii with binary mask (1 for region, 0 elsewhere)

        Raises:
            ValueError: If identifier not found in atlas
        """
        label = self._find_region_label(identifier)

        # Create binary mask
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_computed = dseg_data.compute()
        else:
            dseg_computed = dseg_data

        mask_data = (dseg_computed == label).astype(np.uint8)

        # Create new ZarrNii with the mask
        return ZarrNii.from_darr(
            mask_data,
            affine=self.dseg.affine,
            axes_order=self.dseg.axes_order,
            orientation=self.dseg.orientation,
        )

    def get_region_volume(self, identifier: Union[int, str]) -> float:
        """Calculate volume of a specific region in mm³.

        Args:
            identifier: Region identifier - can be:
                - int: Region label/index
                - str: Region name or abbreviation

        Returns:
            Region volume in mm³

        Raises:
            ValueError: If identifier not found in atlas
        """
        label = self._find_region_label(identifier)

        # Count voxels in region
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_computed = dseg_data.compute()
        else:
            dseg_computed = dseg_data

        voxel_count = np.sum(dseg_computed == label)

        # Calculate voxel volume from affine matrix
        # Voxel size is the determinant of the 3x3 spatial part
        if hasattr(self.dseg.affine, "matrix"):
            spatial_affine = self.dseg.affine.matrix[:3, :3]
        else:
            spatial_affine = self.dseg.affine[:3, :3]
        voxel_volume_mm3 = np.abs(np.linalg.det(spatial_affine))

        return float(voxel_count * voxel_volume_mm3)

    def aggregate_image_by_regions(
        self,
        image: ZarrNii,
        aggregation_func: str = "mean",
        regions: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Aggregate image values by atlas regions.

        Args:
            image: ZarrNii image to aggregate
            aggregation_func: Aggregation function
                ("mean", "sum", "std", "median", "min", "max")
            regions: List of region labels to include (None for all regions)

        Returns:
            DataFrame with region labels, names, and aggregated values

        Raises:
            ValueError: If images don't have compatible shapes/affines
        """
        # Validate that images are compatible
        if image.shape != self.dseg.shape:
            raise ValueError(
                f"Image shape {image.shape} doesn't match "
                f"atlas shape {self.dseg.shape}"
            )

        if not np.allclose(image.affine, self.dseg.affine):
            warnings.warn(
                "Image and atlas have different affine matrices. "
                "Results may not be spatially aligned.",
                stacklevel=3,
            )

        # Define aggregation functions
        agg_functions = {
            "mean": np.mean,
            "sum": np.sum,
            "std": np.std,
            "median": np.median,
            "min": np.min,
            "max": np.max,
        }

        if aggregation_func not in agg_functions:
            raise ValueError(f"Unknown aggregation function: {aggregation_func}")

        agg_func = agg_functions[aggregation_func]

        # Determine which regions to process
        if regions is None:
            regions = self.region_labels

        # Get image and atlas data
        img_data = image.data
        if hasattr(img_data, "compute"):
            img_data = img_data.compute()

        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        # Calculate aggregated values for each region
        results = []
        for label in regions:
            if label not in self.region_labels:
                warnings.warn(
                    f"Region {label} not found in atlas, skipping",
                    stacklevel=3,
                )
                continue

            # Get mask for this region
            mask = dseg_data == label

            if not mask.any():
                # Region not present in this atlas volume
                aggregated_value = np.nan
            else:
                # Extract values for this region and aggregate
                region_values = img_data[mask]
                aggregated_value = agg_func(region_values)

            # Get region info
            region_info = self.get_region_info(label)

            # Add to results
            result_row = {
                self.label_column: label,
                self.name_column: region_info[self.name_column],
                f"{aggregation_func}_value": aggregated_value,
                "volume_mm3": self.get_region_volume(label),
                "voxel_count": int(np.sum(mask)),
            }

            # Add abbreviation if available
            if self.abbrev_column in region_info:
                result_row[self.abbrev_column] = region_info[self.abbrev_column]

            results.append(result_row)

        return pd.DataFrame(results)

    def create_feature_map(
        self,
        feature_df: pd.DataFrame,
        feature_column: str,
        background_value: float = 0.0,
    ) -> ZarrNii:
        """Create a feature map by assigning values to atlas regions.

        Args:
            feature_df: DataFrame with region labels and feature values
            feature_column: Column name containing feature values to map
            background_value: Value to assign to voxels not in any region

        Returns:
            ZarrNii with feature values mapped to regions

        Raises:
            ValueError: If required columns are missing
        """
        if self.label_column not in feature_df.columns:
            raise ValueError(
                f"Feature DataFrame must contain '{self.label_column}' column"
            )
        if feature_column not in feature_df.columns:
            raise ValueError(
                f"Feature column '{feature_column}' not found in DataFrame"
            )

        # Initialize output volume with background value
        output_data = np.full(self.dseg.shape, background_value, dtype=np.float64)
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        # Map feature values to regions
        for _, row in feature_df.iterrows():
            label = row[self.label_column]
            feature_value = row[feature_column]

            if not pd.isna(feature_value):
                mask = dseg_data == label
                output_data[mask] = feature_value

        # Create new ZarrNii with feature map
        return ZarrNii.from_darr(
            output_data,
            affine=self.dseg.affine,
            axes_order=self.dseg.axes_order,
            orientation=self.dseg.orientation,
        )

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all atlas regions.

        Returns:
            DataFrame with region information and volume statistics
        """
        results = []

        for label in self.region_labels:
            region_info = self.get_region_info(label)
            volume = self.get_region_volume(label)

            # Count voxels for this region
            dseg_data = self.dseg.data
            if hasattr(dseg_data, "compute"):
                dseg_computed = dseg_data.compute()
            else:
                dseg_computed = dseg_data
            voxel_count = int(np.sum(dseg_computed == label))

            result_row = region_info.copy()
            result_row["volume_mm3"] = volume
            result_row["voxel_count"] = voxel_count

            results.append(result_row)

        return pd.DataFrame(results)

    def __repr__(self) -> str:
        """Return string representation of Atlas."""
        n_regions = len(self.region_labels)
        dseg_shape = self.dseg.shape
        return (
            f"Atlas(n_regions={n_regions}, "
            f"dseg_shape={dseg_shape}, "
            f"label_column='{self.label_column}', "
            f"name_column='{self.name_column}')"
        )


def import_lut_csv_as_tsv(
    csv_path: Union[str, Path],
    tsv_path: Union[str, Path],
    csv_columns: Optional[List[str]] = None,
) -> None:
    """Convert a CSV lookup table to TSV format for BIDS compatibility.

    This function replicates the functionality of SPIMquant's
    import_lut_csv_as_tsv.py script.

    Args:
        csv_path: Path to input CSV file
        tsv_path: Path to output TSV file
        csv_columns: Column names in CSV
            (default: ["abbreviation", "name", "index"])
    """
    if csv_columns is None:
        csv_columns = ["abbreviation", "name", "index"]

    csv_path = Path(csv_path)
    tsv_path = Path(tsv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV with specified or default column names
    df = pd.read_csv(csv_path, header=None, names=csv_columns)

    # Reorder columns to standard BIDS format
    output_columns = ["index", "name", "abbreviation"]

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in output_columns if col in df.columns]

    # Save as TSV
    df.to_csv(tsv_path, sep="\t", index=False, columns=available_columns)


def import_lut_itksnap_as_tsv(
    itksnap_path: Union[str, Path], tsv_path: Union[str, Path]
) -> None:
    """Convert ITK-SNAP label file to TSV format for BIDS compatibility.

    This function replicates the functionality of SPIMquant's
    import_lut_itksnap_as_tsv.py script.

    Args:
        itksnap_path: Path to ITK-SNAP label file
        tsv_path: Path to output TSV file
    """
    itksnap_path = Path(itksnap_path)
    tsv_path = Path(tsv_path)

    if not itksnap_path.exists():
        raise FileNotFoundError(f"ITK-SNAP file not found: {itksnap_path}")

    # Parse ITK-SNAP format
    # ITK-SNAP format: # IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
    # Example: 1    255    0    0   255    1   1   "Region 1"

    labels = []
    with open(itksnap_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue

            # Parse line
            parts = line.split()
            if len(parts) >= 8:
                try:
                    index = int(parts[0])
                    # Extract label name (everything after 7th column, remove quotes)
                    label_name = " ".join(parts[7:]).strip("\"'")

                    labels.append(
                        {
                            "index": index,
                            "name": label_name,
                            # Generate abbreviation
                            "abbreviation": f"R{index:03d}",
                        }
                    )
                except ValueError:
                    # Skip malformed lines
                    continue

    # Create DataFrame and save as TSV
    df = pd.DataFrame(labels)
    df.to_csv(tsv_path, sep="\t", index=False)


def get_builtin_atlas(name: str = "placeholder") -> Atlas:
    """Get a built-in atlas from the zarrnii package.

    This function provides access to atlases that are packaged with zarrnii,
    useful for testing, examples, and basic analysis workflows.

    Args:
        name: Name of the built-in atlas to retrieve.
            Available atlases:
            - "placeholder": A simple 4-region atlas for testing/examples

    Returns:
        Atlas instance loaded from built-in data

    Raises:
        ValueError: If the requested atlas name is not available

    Examples:
        >>> # Get the placeholder atlas
        >>> atlas = get_builtin_atlas("placeholder")
        >>> print(f"Atlas has {len(atlas.region_labels)} regions")

        >>> # Use in analysis workflows
        >>> atlas = get_builtin_atlas()  # defaults to "placeholder"
        >>> mask = atlas.get_region_mask("Region_A")
    """
    available_atlases = ["placeholder"]

    if name not in available_atlases:
        raise ValueError(
            f"Atlas '{name}' not found. Available built-in atlases: {available_atlases}"
        )

    if name == "placeholder":
        return _create_placeholder_atlas()

    # This should never be reached given the validation above
    raise ValueError(f"Atlas '{name}' not implemented")


def list_builtin_atlases() -> List[Dict[str, str]]:
    """List all available built-in atlases.

    Returns:
        List of dictionaries containing atlas information with keys:
        - 'name': Atlas identifier for use with get_builtin_atlas()
        - 'description': Brief description of the atlas
        - 'regions': Number of regions (approximate)
        - 'resolution': Spatial resolution information

    Examples:
        >>> atlases = list_builtin_atlases()
        >>> for atlas_info in atlases:
        ...     print(f"{atlas_info['name']}: {atlas_info['description']}")
    """
    return [
        {
            "name": "placeholder",
            "description": "Simple 4-region atlas for testing and examples",
            "regions": "5 (including background)",
            "resolution": "32x32x32 voxels, 1mm isotropic",
        }
    ]


def _create_placeholder_atlas() -> Atlas:
    """Create the placeholder atlas with synthetic data.

    Returns:
        Atlas instance with a simple 4-region segmentation
    """
    import tempfile
    from pathlib import Path

    # Create synthetic segmentation data
    shape = (32, 32, 32)
    dseg_data = np.zeros(shape, dtype=np.int32)

    # Create 4 distinct regions in a simple pattern
    # Region 1: Left half
    dseg_data[:, :, :16] = 1

    # Region 2: Right-front quarter
    dseg_data[:16, :16, 16:] = 2

    # Region 3: Right-back-top eighth
    dseg_data[:8, 16:, 16:] = 3

    # Region 4: Right-back-bottom eighth
    dseg_data[8:16, 16:, 16:] = 4

    # Create ZarrNii from synthetic data
    from .transform import AffineTransform

    affine = AffineTransform.identity()  # 1mm isotropic
    dseg = ZarrNii.from_darr(dseg_data, affine=affine)

    # Load labels from packaged TSV file
    labels_path = Path(__file__).parent / "data" / "atlases" / "placeholder_labels.tsv"

    try:
        labels_df = pd.read_csv(labels_path, sep="\t")
    except FileNotFoundError:
        # Fallback to creating labels in memory if file not found
        labels_df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "name": ["Background", "Region_A", "Region_B", "Region_C", "Region_D"],
                "abbreviation": ["BG", "RA", "RB", "RC", "RD"],
            }
        )

    return Atlas(dseg=dseg, labels_df=labels_df)
