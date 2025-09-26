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
class Template:
    """Brain template with anatomical image and associated atlases.

    A template represents an anatomical reference space (e.g., MNI152, ABA)
    with an anatomical image and multiple atlases that can be registered to it.

    Attributes:
        name (str): Template name/identifier
        description (str): Human-readable description
        anatomical_image (ZarrNii): Template anatomical image
        resolution (str): Spatial resolution description
        dimensions (tuple): Image dimensions
        metadata (Dict[str, Any]): Additional template metadata
    """

    name: str
    description: str
    anatomical_image: ZarrNii
    resolution: str = field(default="Unknown")
    dimensions: tuple = field(default=())
    metadata: Dict[str, Any] = field(factory=dict)

    @classmethod
    def from_directory(cls, template_path: Union[str, Path]) -> "Template":
        """Load template from directory containing template files.

        Supports both TemplateFlow standard and legacy formats:
        - TemplateFlow: template_description.json, tpl-{name}_*.nii.gz
        - Legacy: template_info.yaml, template.nii.gz

        Args:
            template_path: Path to template directory

        Returns:
            Template instance

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If template metadata is invalid
        """
        template_path = Path(template_path)

        # Try TemplateFlow format first
        templateflow_info = template_path / "template_description.json"
        legacy_info = template_path / "template_info.yaml"

        if templateflow_info.exists():
            # Load TemplateFlow JSON metadata
            import json

            with open(templateflow_info) as f:
                info = json.load(f)

            # Extract template name from directory or metadata
            if template_path.name.startswith("tpl-"):
                template_name = template_path.name[4:]  # Remove 'tpl-' prefix
            else:
                template_name = info.get("Name", template_path.name)

            # Find anatomical image following TemplateFlow convention
            # Look for tpl-{name}_*.nii.gz files (SPIM, T1w, T2w, etc.)
            anat_patterns = [
                f"tpl-{template_name}_SPIM.nii.gz",
                f"tpl-{template_name}_T1w.nii.gz",
                f"tpl-{template_name}_T2w.nii.gz",
                f"tpl-{template_name}_desc-brain_T1w.nii.gz",
            ]

            anat_file = None
            for pattern in anat_patterns:
                candidate = template_path / pattern
                if candidate.exists():
                    anat_file = candidate
                    break

            if anat_file is None:
                # Fallback: look for any tpl-{name}_*.nii.gz file
                candidates = list(template_path.glob(f"tpl-{template_name}_*.nii.gz"))
                # Filter out atlas files
                candidates = [
                    c
                    for c in candidates
                    if "_atlas-" not in c.name and "_dseg" not in c.name
                ]
                if candidates:
                    anat_file = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"No anatomical image found for template {template_name} in {template_path}"
                    )

            # Convert TemplateFlow metadata to our format
            converted_info = {
                "name": template_name,
                "description": info.get("Description", "TemplateFlow template"),
                "resolution": info.get("Resolution", "Unknown"),
                "anatomical_image": anat_file.name,
                "atlases": info.get("atlases", []),
                "_templateflow": True,
                "_template_dir": str(template_path),
            }

        elif legacy_info.exists():
            # Load legacy YAML metadata
            try:
                import yaml

                with open(legacy_info) as f:
                    converted_info = yaml.safe_load(f)
                converted_info["_templateflow"] = False
                converted_info["_template_dir"] = str(template_path)
            except ImportError:
                raise ImportError("PyYAML is required to load legacy template metadata")
        else:
            raise FileNotFoundError(
                f"No template metadata found. Expected either "
                f"{templateflow_info} or {legacy_info}"
            )

        # Load anatomical image
        anat_file = template_path / converted_info["anatomical_image"]
        if not anat_file.exists():
            raise FileNotFoundError(f"Anatomical image not found: {anat_file}")

        anatomical_image = ZarrNii.from_nifti(str(anat_file))

        return cls(
            name=converted_info["name"],
            description=converted_info["description"],
            anatomical_image=anatomical_image,
            resolution=converted_info.get("resolution", "Unknown"),
            dimensions=tuple(converted_info.get("dimensions", anatomical_image.shape)),
            metadata=converted_info,
        )

    def list_available_atlases(self) -> List[Dict[str, Any]]:
        """List all atlases available for this template.

        Returns:
            List of atlas metadata dictionaries
        """
        if "atlases" in self.metadata:
            return self.metadata["atlases"]
        return []

    def get_atlas(self, atlas_name: str) -> "Atlas":
        """Get a specific atlas for this template.

        Args:
            atlas_name: Name of the atlas to retrieve

        Returns:
            Atlas instance

        Raises:
            ValueError: If atlas not found
        """
        atlases = self.list_available_atlases()
        atlas_info = None

        for atlas in atlases:
            if atlas["name"] == atlas_name:
                atlas_info = atlas
                break

        if atlas_info is None:
            available_names = [a["name"] for a in atlases]
            raise ValueError(
                f"Atlas '{atlas_name}' not found for template '{self.name}'. "
                f"Available atlases: {available_names}"
            )

        # Construct paths relative to template directory
        template_dir = Path(self.metadata.get("_template_dir", "."))
        
        # Handle both TemplateFlow and legacy naming
        if self.metadata.get("_templateflow", False):
            # TemplateFlow naming: tpl-{template}_atlas-{atlas}_dseg.{nii.gz,tsv}
            dseg_path = template_dir / f"tpl-{self.name}_atlas-{atlas_name}_dseg.nii.gz"
            labels_path = template_dir / f"tpl-{self.name}_atlas-{atlas_name}_dseg.tsv"
        else:
            # Legacy naming: use metadata
            dseg_path = template_dir / atlas_info["dseg_file"]
            labels_path = template_dir / atlas_info["labels_file"]

        return Atlas.from_files(dseg_path, labels_path)


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
    """Load the placeholder atlas from the template system.

    Returns:
        Atlas instance with a simple 4-region segmentation
    """
    try:
        # Use the new template-based approach
        return get_builtin_template_atlas("placeholder", "regions")
    except Exception:
        # Fallback to synthetic data if template system fails
        warnings.warn(
            "Template system failed, creating synthetic placeholder atlas.",
            stacklevel=3,
        )
        return _create_synthetic_placeholder_atlas()


def _create_synthetic_placeholder_atlas() -> Atlas:
    """Fallback function to create synthetic placeholder atlas.

    This is used when the packaged atlas files are not available.

    Returns:
        Atlas instance with synthetic data
    """
    # Create synthetic segmentation data (same as before)
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

    # Create labels DataFrame
    labels_df = pd.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "name": ["Background", "Region_A", "Region_B", "Region_C", "Region_D"],
            "abbreviation": ["BG", "RA", "RB", "RC", "RD"],
        }
    )

    return Atlas(dseg=dseg, labels_df=labels_df)


def _install_template_to_templateflow(template_name: str) -> bool:
    """Install a zarrnii built-in template to TEMPLATEFLOW_HOME.
    
    This function copies zarrnii's built-in templates to the user's TemplateFlow
    directory, enabling unified access through the TemplateFlow API.
    Uses the @requires_layout decorator pattern and TF_HOME for proper setup.
    
    Args:
        template_name: Name of the template to install
        
    Returns:
        True if template was successfully installed, False otherwise
    """
    try:
        import shutil
        from templateflow import api as tflow
        from templateflow.conf import TF_HOME, requires_layout
        
        @requires_layout
        def _do_install():
            """Internal function that requires TemplateFlow layout to be set up."""
            # Get paths using proper TemplateFlow configuration
            templateflow_home = Path(TF_HOME)
            zarrnii_template_dir = Path(__file__).parent / "data" / "templates" / f"tpl-{template_name}"
            target_dir = templateflow_home / f"tpl-{template_name}"
            
            # Only copy if not already present
            if not target_dir.exists():
                # Copy template directory (TF_HOME should already exist via @requires_layout)
                shutil.copytree(zarrnii_template_dir, target_dir)
                
                # Verify templateflow can see it
                try:
                    # Test if TemplateFlow can now access this template
                    tflow.get(template_name, desc=None, raise_empty=True)
                    return True
                except Exception:
                    # If TemplateFlow can't see it, remove the copied directory
                    shutil.rmtree(target_dir, ignore_errors=True)
                    return False
            else:
                # Already exists, check if TemplateFlow can access it
                try:
                    tflow.get(template_name, desc=None, raise_empty=True)
                    return True
                except Exception:
                    return False
        
        return _do_install()
                
    except ImportError:
        # TemplateFlow not available
        return False
    except Exception:
        # Other errors during installation
        return False


def get_builtin_template(name: str = "placeholder") -> Template:
    """Get a built-in template from the zarrnii package.

    This function provides access to templates that are packaged with zarrnii,
    including their anatomical images and associated atlases.
    
    When the templateflow extra is installed, this function will lazily copy
    the template to TEMPLATEFLOW_HOME on first access, enabling unified access
    through the TemplateFlow API.

    Args:
        name: Name of the built-in template to retrieve.
            Available templates:
            - "placeholder": A simple synthetic template for testing/examples

    Returns:
        Template instance loaded from built-in data

    Raises:
        ValueError: If the requested template name is not available

    Examples:
        >>> # Get the placeholder template
        >>> template = get_builtin_template("placeholder")
        >>> print(f"Template: {template.name}")
        >>> print(f"Available atlases: {[a['name'] for a in template.list_available_atlases()]}")

        >>> # Get an atlas from the template
        >>> atlas = template.get_atlas("regions")
        
        >>> # After first call, template is available via TemplateFlow API (if installed)
        >>> import templateflow.api as tflow
        >>> # This will work after lazy loading: tflow.get("placeholder", suffix="SPIM")
    """
    available_templates = ["placeholder"]

    if name not in available_templates:
        raise ValueError(
            f"Template '{name}' not found. Available built-in templates: {available_templates}"
        )

    # Lazy loading: Install to TemplateFlow if available (only on first access)
    _install_template_to_templateflow(name)

    # Load template from packaged data (TemplateFlow naming)
    template_path = Path(__file__).parent / "data" / "templates" / f"tpl-{name}"
    return Template.from_directory(template_path)


def get_templateflow_template(template: str, suffix: str = "SPIM") -> Template:
    """Get a template from TemplateFlow.
    
    This function uses the TemplateFlow API to download and access templates
    from the official TemplateFlow repository.
    
    Args:
        template: Template identifier (e.g., 'MNI152NLin2009cAsym', 'ABA')
        suffix: Image suffix/modality (e.g., 'SPIM', 'T1w', 'T2w')
        
    Returns:
        Template instance loaded from TemplateFlow
        
    Raises:
        ImportError: If templateflow package is not available
        ValueError: If template is not found in TemplateFlow
        
    Examples:
        >>> # Get MNI152 template
        >>> template = get_templateflow_template("MNI152NLin2009cAsym", "T1w")
        
        >>> # Get Allen Brain Atlas template  
        >>> template = get_templateflow_template("ABA", "SPIM")
    """
    try:
        from templateflow import api as tflow
    except ImportError:
        raise ImportError(
            "templateflow is required for TemplateFlow integration. "
            "Install with: pip install zarrnii[templateflow] or pip install templateflow"
        )
    
    try:
        # Get template file from TemplateFlow
        template_file = tflow.get(template, suffix=suffix, extension=".nii.gz")
        
        # Create a temporary template directory structure
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix=f"tpl-{template}_"))
        
        # Copy template file with TemplateFlow naming
        template_path = temp_dir / f"tpl-{template}_{suffix}.nii.gz"
        import shutil
        shutil.copy2(template_file, template_path)
        
        # Create minimal template_description.json
        template_info = {
            "Name": template,
            "BIDSVersion": "1.8.0", 
            "TemplateFlowVersion": "0.8.0",
            "Description": f"Template {template} from TemplateFlow",
            "Authors": ["TemplateFlow"],
            "License": "Various",
            "ReferencesAndLinks": ["https://templateflow.org"],
            "Resolution": "Unknown",
            "SpatialReference": {
                "VoxelSizes": [1.0, 1.0, 1.0],
                "Units": "mm",
                "Origin": "center"
            },
            "atlases": []  # Could be expanded to query TemplateFlow for atlases
        }
        
        import json
        with open(temp_dir / "template_description.json", "w") as f:
            json.dump(template_info, f, indent=2)
        
        # Load template
        return Template.from_directory(temp_dir)
        
    except Exception as e:
        raise ValueError(f"Failed to load template '{template}' from TemplateFlow: {e}")


def list_templateflow_templates() -> List[str]:
    """List available templates from TemplateFlow.
    
    Returns:
        List of template identifiers available in TemplateFlow
        
    Raises:
        ImportError: If templateflow package is not available
        
    Examples:
        >>> templates = list_templateflow_templates()
        >>> print("Available TemplateFlow templates:", templates)
    """
    try:
        from templateflow import api as tflow
    except ImportError:
        raise ImportError(
            "templateflow is required for TemplateFlow integration. "
            "Install with: pip install zarrnii[templateflow] or pip install templateflow"
        )
    
    return list(tflow.templates())


def install_zarrnii_templates() -> Dict[str, bool]:
    """Manually install all zarrnii built-in templates to TEMPLATEFLOW_HOME.
    
    This function allows users to explicitly install zarrnii's built-in templates
    to their TemplateFlow directory for unified access through the TemplateFlow API.
    Uses the @requires_layout decorator to ensure proper TemplateFlow setup.
    
    Returns:
        Dictionary mapping template names to installation success status
        
    Raises:
        ImportError: If templateflow package is not available
        
    Examples:
        >>> results = install_zarrnii_templates()
        >>> print("Installation results:", results)
        >>> # {'placeholder': True}
        
        >>> # After installation, templates are accessible via TemplateFlow
        >>> import templateflow.api as tflow
        >>> template_files = tflow.get("placeholder", suffix="SPIM")
    """
    try:
        from templateflow import api as tflow
        from templateflow.conf import requires_layout
    except ImportError:
        raise ImportError(
            "templateflow is required for TemplateFlow integration. "
            "Install with: pip install zarrnii[templateflow] or pip install templateflow"
        )
    
    @requires_layout  
    def _install_all():
        """Install all templates with proper TemplateFlow layout setup."""
        available_templates = ["placeholder"]  # Add more as they become available
        results = {}
        
        for template_name in available_templates:
            results[template_name] = _install_template_to_templateflow(template_name)
        
        return results
    
    return _install_all()


def list_builtin_templates() -> List[Dict[str, str]]:
    """List all available built-in templates.

    Returns:
        List of dictionaries containing template information with keys:
        - 'name': Template identifier for use with get_builtin_template()
        - 'description': Brief description of the template
        - 'resolution': Spatial resolution information
        - 'atlases': Number of available atlases

    Examples:
        >>> templates = list_builtin_templates()
        >>> for template_info in templates:
        ...     print(f"{template_info['name']}: {template_info['description']}")
    """
    templates = []

    # Add placeholder template info
    try:
        placeholder = get_builtin_template("placeholder")
        templates.append(
            {
                "name": "placeholder",
                "description": placeholder.description,
                "resolution": placeholder.resolution,
                "atlases": str(len(placeholder.list_available_atlases())),
            }
        )
    except Exception:
        # Fallback if template can't be loaded
        templates.append(
            {
                "name": "placeholder",
                "description": "Simple synthetic template for testing and examples",
                "resolution": "32x32x32 voxels, 1mm isotropic",
                "atlases": "1",
            }
        )

    return templates


def get_builtin_template_atlas(
    template_name: str = "placeholder", atlas_name: str = "regions"
) -> Atlas:
    """Convenience function to get an atlas from a built-in template.

    Args:
        template_name: Name of the built-in template
        atlas_name: Name of the atlas within the template

    Returns:
        Atlas instance

    Examples:
        >>> # Get the default atlas from default template
        >>> atlas = get_builtin_template_atlas()

        >>> # Get specific atlas from specific template
        >>> atlas = get_builtin_template_atlas("placeholder", "regions")
    """
    template = get_builtin_template(template_name)
    return template.get_atlas(atlas_name)
