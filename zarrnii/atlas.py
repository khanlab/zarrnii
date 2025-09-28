"""Atlas handling module for ZarrNii.

This module provides functionality for working with brain atlases through TemplateFlow,
specifically BIDS-formatted dseg.nii.gz segmentation images with corresponding dseg.tsv
lookup tables. It enables region-of-interest (ROI) based analysis and
aggregation of data across atlas regions.

Based on functionality from SPIMquant:
https://github.com/khanlab/SPIMquant/blob/main/spimquant/workflow/scripts/
"""

import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from attrs import define, field

try:
    import templateflow.api as tflow
    from templateflow.conf import TF_HOME, requires_layout

    TEMPLATEFLOW_AVAILABLE = True
except ImportError:
    TEMPLATEFLOW_AVAILABLE = False
    TF_HOME = None
    requires_layout = lambda f: f  # noqa: E731

from .core import ZarrNii


class AmbiguousTemplateFlowQueryError(ValueError):
    """Raised when TemplateFlow query returns multiple files requiring more specific query."""

    def __init__(self, template: str, suffix: str, matching_files: List[str], **kwargs):
        """Initialize with template query details."""
        self.template = template
        self.suffix = suffix
        self.matching_files = matching_files
        self.query_kwargs = kwargs

        super().__init__(
            f"Query for template='{template}' suffix='{suffix}' returned {len(matching_files)} files. "
            f"Add more qualifiers like resolution=1, cohort='01', etc. to narrow the query. "
            f"Matching files: {matching_files[:3]}{'...' if len(matching_files) > 3 else ''}"
        )


# TemplateFlow wrapper functions
def get(template: str, **kwargs) -> Union[str, List[str]]:
    """Thin wrapper on templateflow.api.get() - preserves original signature.

    Args:
        template: Template name (e.g., "MNI152NLin2009cAsym")
        **kwargs: Additional TemplateFlow parameters (suffix, resolution, etc.)

    Returns:
        String path or list of string paths as returned by TemplateFlow

    Raises:
        ImportError: If TemplateFlow is not available
    """
    if not TEMPLATEFLOW_AVAILABLE:
        raise ImportError(
            "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
        )

    return tflow.get(template, **kwargs)


def get_template(template: str, suffix: str = "SPIM", **kwargs) -> "Template":
    """Get Template object from TemplateFlow with single file validation.

    Args:
        template: Template name (e.g., "MNI152NLin2009cAsym")
        suffix: File suffix (e.g., "T1w", "SPIM")
        **kwargs: Additional TemplateFlow parameters (resolution, cohort, etc.)

    Returns:
        Template object with anatomical image

    Raises:
        AmbiguousTemplateFlowQueryError: If query returns multiple files
        FileNotFoundError: If no matching files found
        ImportError: If TemplateFlow is not available
    """
    if not TEMPLATEFLOW_AVAILABLE:
        raise ImportError(
            "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
        )

    result = tflow.get(template, suffix=suffix, **kwargs)

    # Handle TemplateFlow's variable return behavior
    if isinstance(result, list):
        if len(result) == 0:
            raise FileNotFoundError(
                f"No files found for template='{template}' suffix='{suffix}'"
            )
        elif len(result) > 1:
            raise AmbiguousTemplateFlowQueryError(template, suffix, result, **kwargs)
        else:
            result = result[0]  # Single item list

    # Load anatomical image - determine format from file extension
    result_path = Path(str(result))
    if result_path.suffix.lower() in [".nii", ".gz"]:
        anatomical_image = ZarrNii.from_nifti(str(result))
    elif result_path.suffix.lower() == ".zarr" or "ome.zarr" in str(result_path):
        anatomical_image = ZarrNii.from_ome_zarr(str(result))
    else:
        # Default to NIfTI for unknown extensions
        anatomical_image = ZarrNii.from_nifti(str(result))

    return Template(
        name=template,
        description=f"TemplateFlow template: {template}",
        anatomical_image=anatomical_image,
        resolution=str(kwargs.get("resolution", "Unknown")),
        dimensions=anatomical_image.shape,
        metadata=kwargs,
    )


def get_atlas(template: str, atlas: str, **kwargs) -> "Atlas":
    """Load atlas directly from TemplateFlow by template and atlas name.
    
    Args:
        template: Template name (e.g., 'MNI152NLin2009cAsym') 
        atlas: Atlas name (e.g., 'DKT', 'Harvard-Oxford')
        **kwargs: Additional TemplateFlow query parameters
        
    Returns:
        Atlas object loaded from TemplateFlow
        
    Raises:
        ImportError: If templateflow is not available
        FileNotFoundError: If atlas files not found
    """
    if not TEMPLATEFLOW_AVAILABLE:
        raise ImportError(
            "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
        )
    
    # Get dseg file  
    dseg_result = tflow.get(template, suffix="dseg", atlas=atlas, **kwargs)
    if isinstance(dseg_result, list):
        if len(dseg_result) == 0:
            raise FileNotFoundError(f"No dseg files found for template '{template}' atlas '{atlas}'")
        dseg_file = dseg_result[0]  # Take first match
    else:
        dseg_file = dseg_result
        
    # Get corresponding TSV file
    tsv_result = tflow.get(template, suffix="dseg", atlas=atlas, extension=".tsv", **kwargs)
    if isinstance(tsv_result, list):
        if len(tsv_result) == 0:
            raise FileNotFoundError(f"No TSV files found for template '{template}' atlas '{atlas}'")
        tsv_file = tsv_result[0]  # Take first match
    else:
        tsv_file = tsv_result
        
    return Atlas.from_files(dseg_file, tsv_file)


def save_atlas_to_templateflow(atlas: "Atlas", template_name: str, atlas_name: str) -> str:
    """Save atlas to TemplateFlow directory as BIDS-compliant files.
    
    Args:
        atlas: Atlas object to save
        template_name: Template name (e.g., 'MyTemplate')
        atlas_name: Atlas name (e.g., 'MyAtlas')
        
    Returns:
        Path to created template directory
        
    Raises:
        ImportError: If templateflow is not available
    """
    if not TEMPLATEFLOW_AVAILABLE:
        raise ImportError(
            "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
        )
    
    template_dir = Path(TF_HOME) / f"tpl-{template_name}"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dseg.nii.gz file
    dseg_file = template_dir / f"tpl-{template_name}_atlas-{atlas_name}_dseg.nii.gz"
    atlas.image.to_nifti(str(dseg_file))
    
    # Save dseg.tsv file  
    tsv_file = template_dir / f"tpl-{template_name}_atlas-{atlas_name}_dseg.tsv"
    atlas.lookup_table.to_csv(str(tsv_file), sep='\t', index=False)
    
    return str(template_dir)


@requires_layout
def add_template_to_templateflow(
    template_path: str,
    labels_path: Optional[str] = None,
    template_name: Optional[str] = None,
    suffix: str = "T1w",
) -> str:
    """Add new template and optional atlas to TemplateFlow directory.

    This function copies template files to the TemplateFlow directory structure
    following TemplateFlow naming conventions.

    Args:
        template_path: Path to template NIfTI file
        labels_path: Optional path to atlas labels TSV file
        template_name: Template identifier (derived from filename if not provided)
        suffix: Template suffix (T1w, SPIM, etc.)

    Returns:
        Template name for use with TemplateFlow API

    Raises:
        ImportError: If TemplateFlow is not available
        FileNotFoundError: If input files don't exist
        ValueError: If files don't follow expected format
    """
    if not TEMPLATEFLOW_AVAILABLE:
        raise ImportError(
            "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
        )

    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Derive template name if not provided
    if template_name is None:
        template_name = template_path.stem.replace(".nii", "")
        if template_name.startswith("tpl-"):
            template_name = template_name[4:]  # Remove tpl- prefix

    # Set up TemplateFlow directory structure
    tf_home = Path(TF_HOME)
    template_dir = tf_home / f"tpl-{template_name}"
    template_dir.mkdir(exist_ok=True)

    # Copy template file with TemplateFlow naming
    template_target = template_dir / f"tpl-{template_name}_{suffix}.nii.gz"
    shutil.copy2(template_path, template_target)

    # Copy labels file if provided
    if labels_path:
        labels_path = Path(labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Derive atlas name from labels filename
        atlas_name = labels_path.stem.replace("_labels", "").replace("_dseg", "")
        if atlas_name.startswith("tpl-"):
            # Extract atlas name from TemplateFlow format
            parts = atlas_name.split("_atlas-")
            if len(parts) > 1:
                atlas_name = parts[1]

        # Copy with TemplateFlow atlas naming
        labels_target = (
            template_dir / f"tpl-{template_name}_atlas-{atlas_name}_dseg.tsv"
        )
        shutil.copy2(labels_path, labels_target)

    return template_name


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
    dimensions: Tuple = field(default=())
    metadata: Dict[str, Any] = field(factory=dict)

    def get_atlas(self, atlas_name: str) -> "Atlas":
        """Get a specific atlas for this template using TemplateFlow.

        Args:
            atlas_name: Name of the atlas to retrieve

        Returns:
            Atlas instance

        Raises:
            ImportError: If TemplateFlow is not available
            FileNotFoundError: If atlas files not found
        """
        if not TEMPLATEFLOW_AVAILABLE:
            raise ImportError(
                "TemplateFlow is required. Install with: pip install zarrnii[templateflow]"
            )

        try:
            # Get dseg file from TemplateFlow
            dseg_files = tflow.get(self.name, atlas=atlas_name, suffix="dseg")
            if isinstance(dseg_files, list):
                dseg_files = [f for f in dseg_files if f.endswith(".nii.gz")]
                if not dseg_files:
                    raise FileNotFoundError(
                        f"No dseg.nii.gz found for atlas {atlas_name}"
                    )
                dseg_path = dseg_files[0]
            else:
                dseg_path = dseg_files

            # Get labels TSV file
            labels_files = tflow.get(
                self.name, atlas=atlas_name, suffix="dseg", extension=".tsv"
            )
            if isinstance(labels_files, list):
                labels_path = labels_files[0] if labels_files else None
            else:
                labels_path = labels_files

            if not labels_path:
                raise FileNotFoundError(f"No dseg.tsv found for atlas {atlas_name}")

            return Atlas.from_files(dseg_path, labels_path)

        except Exception as e:
            raise FileNotFoundError(
                f"Could not load atlas '{atlas_name}' for template '{self.name}': {e}"
            )


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
            ValueError: If files are invalid or incompatible
        """
        dseg_path = Path(dseg_path)
        labels_path = Path(labels_path)

        if not dseg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {dseg_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load segmentation image - determine format from file extension
        if dseg_path.suffix.lower() in [".nii", ".gz"]:
            dseg = ZarrNii.from_nifti(str(dseg_path), **zarrnii_kwargs)
        elif dseg_path.suffix.lower() == ".zarr" or "ome.zarr" in str(dseg_path):
            dseg = ZarrNii.from_ome_zarr(str(dseg_path), **zarrnii_kwargs)
        else:
            # Default to NIfTI for unknown extensions
            dseg = ZarrNii.from_nifti(str(dseg_path), **zarrnii_kwargs)

        # Load labels DataFrame
        labels_df = pd.read_csv(labels_path, sep="\t")

        return cls(
            dseg=dseg,
            labels_df=labels_df,
            label_column=label_column,
            name_column=name_column,
            abbrev_column=abbrev_column,
        )

    def _resolve_region_identifier(self, region_id: Union[int, str]) -> int:
        """Resolve region identifier to integer label.

        Supports lookup by:
        - Integer label (returned as-is)
        - Region name (looked up in name column)
        - Region abbreviation (looked up in abbreviation column if available)

        Args:
            region_id: Region identifier (int, name, or abbreviation)

        Returns:
            Integer label for the region

        Raises:
            ValueError: If region identifier not found
            TypeError: If region_id is not int or str
        """
        if isinstance(region_id, (int, np.integer)):
            return int(region_id)
        elif isinstance(region_id, str):
            # Try name column first
            name_matches = self.labels_df[
                self.labels_df[self.name_column].str.lower() == region_id.lower()
            ]
            if not name_matches.empty:
                return int(name_matches.iloc[0][self.label_column])

            # Try abbreviation column if available
            if (
                self.abbrev_column in self.labels_df.columns
                and not self.labels_df[self.abbrev_column].isna().all()
            ):
                abbrev_matches = self.labels_df[
                    self.labels_df[self.abbrev_column].str.lower() == region_id.lower()
                ]
                if not abbrev_matches.empty:
                    return int(abbrev_matches.iloc[0][self.label_column])

            # No matches found
            raise ValueError(
                f"Region '{region_id}' not found in atlas. "
                f"Available names: {self.labels_df[self.name_column].tolist()}"
            )
        else:
            raise TypeError(
                f"Region identifier must be int or str, got {type(region_id)}"
            )

    def get_region_info(self, region_id: Union[int, str]) -> Dict[str, Any]:
        """Get information about a specific region.

        Args:
            region_id: Region identifier (int label, name, or abbreviation)

        Returns:
            Dictionary containing region information

        Raises:
            ValueError: If region not found in atlas
        """
        label = self._resolve_region_identifier(region_id)

        # Find the region in labels DataFrame
        region_row = self.labels_df[self.labels_df[self.label_column] == label]
        if region_row.empty:
            raise ValueError(f"Region with label {label} not found in atlas")

        return region_row.iloc[0].to_dict()

    def get_region_mask(self, region_id: Union[int, str]) -> ZarrNii:
        """Create binary mask for a specific region.

        Args:
            region_id: Region identifier (int label, name, or abbreviation)

        Returns:
            ZarrNii instance containing binary mask (1 for region, 0 elsewhere)

        Raises:
            ValueError: If region not found in atlas
        """
        label = self._resolve_region_identifier(region_id)

        # Validate that the region exists in our labels_df
        if not (self.labels_df[self.label_column] == label).any():
            raise ValueError(f"Region with label {label} not found in atlas")

        # Create binary mask
        mask_data = (self.dseg.data == label).astype(np.uint8)

        return ZarrNii.from_darr(
            mask_data, affine=self.dseg.affine, axes_order=self.dseg.axes_order
        )

    def get_region_volume(self, region_id: Union[int, str]) -> float:
        """Calculate volume of a specific region in mm³.

        Args:
            region_id: Region identifier (int label, name, or abbreviation)

        Returns:
            Volume in cubic millimeters

        Raises:
            ValueError: If region not found in atlas
        """
        label = self._resolve_region_identifier(region_id)

        # Count voxels with this label
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            voxel_count = int((dseg_data == label).sum().compute())
        else:
            voxel_count = int((dseg_data == label).sum())

        # Calculate volume using voxel size from affine
        # Volume per voxel = abs(det(affine[:3, :3]))
        voxel_volume = abs(np.linalg.det(self.dseg.affine[:3, :3]))

        return float(voxel_count * voxel_volume)

    def aggregate_image_by_regions(
        self,
        image: ZarrNii,
        aggregation_func: str = "mean",
        background_label: int = 0,
    ) -> pd.DataFrame:
        """Aggregate image values by atlas regions.

        Args:
            image: Image to aggregate (must be compatible with atlas)
            aggregation_func: Aggregation function ('mean', 'sum', 'std', 'median', 'min', 'max')
            background_label: Label value to treat as background (excluded from results)

        Returns:
            DataFrame with columns: label, name, aggregated_value, volume_mm3

        Raises:
            ValueError: If image and atlas are incompatible
        """
        # Validate image compatibility
        if not np.array_equal(image.shape, self.dseg.shape):
            raise ValueError(
                f"Image shape {image.shape} doesn't match atlas shape {self.dseg.shape}"
            )

        if not np.allclose(image.affine, self.dseg.affine, atol=1e-6):
            warnings.warn(
                "Image and atlas affines don't match exactly. "
                "Results may be spatially inconsistent."
            )

        # Get all unique labels (excluding background)
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()
        unique_labels = np.unique(dseg_data)
        unique_labels = unique_labels[unique_labels != background_label]

        results = []
        for label in unique_labels:
            # Create mask for this region
            mask = self.dseg.data == label

            # Extract image values for this region
            region_values = image.data[mask]

            # Skip if no voxels (shouldn't happen with unique labels)
            if region_values.size == 0:
                continue

            # Compute aggregation
            if hasattr(region_values, "compute"):
                # Dask array - need to compute
                if aggregation_func == "mean":
                    agg_value = float(region_values.mean().compute())
                elif aggregation_func == "sum":
                    agg_value = float(region_values.sum().compute())
                elif aggregation_func == "std":
                    agg_value = float(region_values.std().compute())
                elif aggregation_func == "median":
                    agg_value = float(np.median(region_values.compute()))
                elif aggregation_func == "min":
                    agg_value = float(region_values.min().compute())
                elif aggregation_func == "max":
                    agg_value = float(region_values.max().compute())
                else:
                    raise ValueError(
                        f"Unknown aggregation function: {aggregation_func}. "
                        "Supported: mean, sum, std, median, min, max"
                    )
            else:
                # NumPy array - direct computation
                if aggregation_func == "mean":
                    agg_value = float(region_values.mean())
                elif aggregation_func == "sum":
                    agg_value = float(region_values.sum())
                elif aggregation_func == "std":
                    agg_value = float(region_values.std())
                elif aggregation_func == "median":
                    agg_value = float(np.median(region_values))
                elif aggregation_func == "min":
                    agg_value = float(region_values.min())
                elif aggregation_func == "max":
                    agg_value = float(region_values.max())
                else:
                    raise ValueError(
                        f"Unknown aggregation function: {aggregation_func}. "
                        "Supported: mean, sum, std, median, min, max"
                    )

            # Get region info
            try:
                region_info = self.get_region_info(int(label))
                region_name = region_info[self.name_column]
            except ValueError:
                region_name = f"Unknown_Region_{label}"

            # Calculate volume
            volume = self.get_region_volume(int(label))

            results.append(
                {
                    "label": int(label),
                    "name": region_name,
                    f"{aggregation_func}_value": agg_value,
                    "volume_mm3": volume,
                }
            )

        return pd.DataFrame(results)

    def create_feature_map(
        self,
        feature_data: pd.DataFrame,
        feature_column: str,
        label_column: str = "index",
    ) -> ZarrNii:
        """Create feature map by assigning values to atlas regions.

        Args:
            feature_data: DataFrame with region labels and feature values
            feature_column: Column name containing feature values to map
            label_column: Column name containing region labels

        Returns:
            ZarrNii instance with feature values mapped to regions

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = [label_column, feature_column]
        missing_cols = [col for col in required_cols if col not in feature_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in feature_data: {missing_cols}")

        # Create output array initialized to zeros
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()
        feature_map = np.zeros_like(dseg_data, dtype=np.float32)

        # Map feature values to regions
        for _, row in feature_data.iterrows():
            label = int(row[label_column])
            feature_value = float(row[feature_column])

            # Set all voxels with this label to the feature value
            feature_map[dseg_data == label] = feature_value

        return ZarrNii.from_darr(
            feature_map, affine=self.dseg.affine, axes_order=self.dseg.axes_order
        )


# Utility functions for atlas format conversion (from SPIMquant)
def import_lut_csv_as_tsv(csv_path: str, output_path: str) -> None:
    """Convert CSV lookup table to BIDS TSV format.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output TSV file
    """
    df = pd.read_csv(csv_path)
    df.to_csv(output_path, sep="\t", index=False)


def import_lut_itksnap_as_tsv(itksnap_path: str, output_path: str) -> None:
    """Convert ITK-SNAP label file to BIDS TSV format.

    Args:
        itksnap_path: Path to ITK-SNAP label file
        output_path: Path to output TSV file
    """
    # Parse ITK-SNAP format (simplified version)
    data = []
    with open(itksnap_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 7:  # Should have index, R, G, B, A, VIS, MSH, "LABEL"
                    index = int(parts[0])
                    # Extract label from quoted string at the end
                    label_start = line.find('"')
                    label_end = line.rfind('"')
                    if (
                        label_start != -1
                        and label_end != -1
                        and label_start < label_end
                    ):
                        name = line[label_start + 1 : label_end]
                        # Create abbreviation from name (first letters of words)
                        abbrev = "".join([word[0].upper() for word in name.split()[:2]])
                        data.append(
                            {"index": index, "name": name, "abbreviation": abbrev}
                        )

    df = pd.DataFrame(data)
    df.to_csv(output_path, sep="\t", index=False)
