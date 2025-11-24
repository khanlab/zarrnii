"""Atlas handling module for ZarrNii.

This module provides functionality for working with brain atlases through TemplateFlow,
specifically BIDS-formatted dseg.nii.gz segmentation images with corresponding dseg.tsv
lookup tables. It enables region-of-interest (ROI) based analysis and
aggregation of data across atlas regions.

Based on functionality from SPIMquant:
https://github.com/khanlab/SPIMquant/blob/main/spimquant/workflow/scripts/
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ngff_zarr as nz
import numpy as np
import pandas as pd
from attrs import define, field
from scipy.interpolate import interpn

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


def get_atlas(template: str, atlas: str, **kwargs) -> "ZarrNiiAtlas":
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
            raise FileNotFoundError(
                f"No dseg files found for template '{template}' atlas '{atlas}'"
            )
        dseg_file = dseg_result[0]  # Take first match
    else:
        dseg_file = dseg_result

    # Get corresponding TSV file
    tsv_result = tflow.get(
        template, suffix="dseg", atlas=atlas, extension=".tsv", **kwargs
    )
    if isinstance(tsv_result, list):
        if len(tsv_result) == 0:
            raise FileNotFoundError(
                f"No TSV files found for template '{template}' atlas '{atlas}'"
            )
        tsv_file = tsv_result[0]  # Take first match
    else:
        tsv_file = tsv_result

    return ZarrNiiAtlas.from_files(dseg_file, tsv_file)


def save_atlas_to_templateflow(
    atlas: "ZarrNiiAtlas", template_name: str, atlas_name: str
) -> str:
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
    atlas.lookup_table.to_csv(str(tsv_file), sep="\t", index=False)

    return str(template_dir)


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

    def get_atlas(self, atlas_name: str) -> "ZarrNiiAtlas":
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

            return ZarrNiiAtlas.from_files(dseg_path, labels_path)

        except Exception as e:
            raise FileNotFoundError(
                f"Could not load atlas '{atlas_name}' for template '{self.name}': {e}"
            )


@define
class ZarrNiiAtlas(ZarrNii):
    """Brain atlas with segmentation image and region lookup table.

    Represents a brain atlas consisting of a segmentation image (dseg) that
    assigns integer labels to brain regions, and a lookup table (tsv) that
    maps these labels to region names and other metadata.


    Extension of ZarrNii to support atlas label tables.

    Inherits all functionality from ZarrNii and adds support for
    storing region/label metadata in a pandas DataFrame.

    Attributes
    ----------
    labels_df : pandas.DataFrame
        DataFrame containing label information for the atlas.
    label_column : str
        Name of the column in labels_df containing label indices.
    name_column : str
        Name of the column in labels_df containing region names.
    abbrev_column : str
        Name of the column in labels_df containing region abbreviations.

    """

    labels_df: pd.DataFrame = field(default=None)
    label_column: str = field(default="index")
    name_column: str = field(default="name")
    abbrev_column: str = field(default="abbreviation")

    @property
    def dseg(self) -> "ZarrNii":
        """Return self as the segmentation image (for compatibility with API)."""
        return self

    @classmethod
    def create_from_dseg(cls, dseg: ZarrNii, labels_df: pd.DataFrame, **kwargs):
        """Create ZarrNiiAtlas from a dseg ZarrNii and labels DataFrame.

        Args:
            dseg: ZarrNii segmentation image
            labels_df: DataFrame containing label information
            **kwargs: Additional keyword arguments for label/name/abbrev columns

        Returns:
            ZarrNiiAtlas instance
        """
        if not isinstance(dseg, ZarrNii):
            raise TypeError(f"dseg must be a ZarrNii instance, got {type(dseg)}")

        # Note: attrs strips leading underscore from _omero in __init__ signature
        # so we pass it as 'omero' instead of '_omero'
        return cls(
            ngff_image=dseg.ngff_image,
            axes_order=dseg.axes_order,
            xyz_orientation=dseg.xyz_orientation,
            omero=getattr(dseg, "_omero", None),
            labels_df=labels_df,
            **kwargs,
        )

    @staticmethod
    def _import_csv_lut(csv_path: str, **kwargs_read_csv) -> pd.DataFrame:
        """import CSV lookup table

        Args:
            csv_path: Path to input CSV file
            kwargs_read_csv: options to pandas read_csv
        """
        return pd.read_csv(csv_path, **kwargs_read_csv)

    @staticmethod
    def _import_tsv_lut(tsv_path: str) -> pd.DataFrame:
        """import TSV lookup table

        Args:
            tsv_path: Path to input TSV file
        """
        return pd.read_csv(tsv_path, sep="\t")

    @staticmethod
    def _import_labelmapper_lut(
        labelmapper_json: str,
        in_cols=["index", "color", "abbreviation", "name"],
        keep_cols=["name", "abbreviation", "color"],
    ) -> pd.DataFrame:
        """Import labelmapper json label lut

        Args:
            labelmapper_json: Path to the input json file
        """

        with open(labelmapper_json) as fp:
            lut = json.load(fp)

        return pd.DataFrame(lut, columns=in_cols).set_index("index")[keep_cols]

    @staticmethod
    def _import_itksnap_lut(itksnap_txt_path: str) -> pd.DataFrame:
        """Import ITK-SNAP label file

        Args:
            itksnap_path: Path to ITK-SNAP label file
        """

        # Function to convert 8‐bit R, G, B values to a hex string (e.g., "ff0000")
        def rgb_to_hex(r, g, b):
            return "{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

        # Read the ITK‐Snap LUT file.
        # The ITK‐Snap LUT contains a header (all lines starting with "#")
        # so we use the "comment" argument to skip those lines.
        # The data columns are (in order):
        #   index, R, G, B, A, VIS, MSH, LABEL
        columns = ["index", "R", "G", "B", "A", "VIS", "MSH", "LABEL"]
        df = pd.read_csv(
            itksnap_txt_path, sep=r"\s+", comment="#", header=None, names=columns
        )

        # Remove the surrounding quotes from the LABEL column to get the ROI name.
        df["name"] = df["LABEL"].str.strip('"')

        # Create the BIDS color column by converting the R, G, B values to a hex string.
        df["color"] = df.apply(
            lambda row: rgb_to_hex(row["R"], row["G"], row["B"]), axis=1
        )

        # For the BIDS LUT, we only need the columns "index", "name", and "color"
        return df[["index", "name", "color"]]

    # ---- New constructors for different LUT formats ----

    @classmethod
    def from_files(
        cls, dseg_path: Union[str, Path], labels_path: Union[str, Path], **kwargs
    ):
        """Load ZarrNiiAtlas from dseg image and labels TSV files.

        Args:
            dseg_path: Path to segmentation image (NIfTI or OME-Zarr)
            labels_path: Path to labels TSV file
            **kwargs: Additional arguments passed to ZarrNii.from_file()

        Returns:
            ZarrNiiAtlas instance
        """
        # Load segmentation image
        dseg = ZarrNii.from_file(str(dseg_path), **kwargs)

        # Load labels dataframe
        labels_df = pd.read_csv(str(labels_path), sep="\t")

        # Create atlas instance using create_from_dseg
        return cls.create_from_dseg(dseg, labels_df)

    @classmethod
    def from_itksnap_lut(cls, path, lut_path, **kwargs):
        """
        Construct from itksnap lut file.
        """
        znii = super().from_file(path, **kwargs)
        labels_df = cls._import_itksnap_lut(lut_path)
        return cls(
            ngff_image=znii.ngff_image,
            axes_order=znii.axes_order,
            xyz_orientation=znii.xyz_orientation,
            labels_df=labels_df,
            omero=getattr(znii, "_omero", None),
        )

    @classmethod
    def from_csv_lut(cls, path, lut_path, **kwargs):
        """
        Construct from csv lut file.
        """
        znii = super().from_file(path, **kwargs)
        labels_df = cls._import_csv_lut(lut_path)
        return cls(
            ngff_image=znii.ngff_image,
            axes_order=znii.axes_order,
            xyz_orientation=znii.xyz_orientation,
            labels_df=labels_df,
            omero=getattr(znii, "_omero", None),
        )

    @classmethod
    def from_tsv_lut(cls, path, lut_path, **kwargs):
        """
        Construct from tsv lut file.
        """
        znii = super().from_file(path, **kwargs)
        labels_df = cls._import_tsv_lut(lut_path)
        return cls(
            ngff_image=znii.ngff_image,
            axes_order=znii.axes_order,
            xyz_orientation=znii.xyz_orientation,
            labels_df=labels_df,
            omero=getattr(znii, "_omero", None),
        )

    @classmethod
    def from_labelmapper_lut(cls, path, lut_path, **kwargs):
        """
        Construct from labelmapper lut file.
        """
        znii = super().from_file(path, **kwargs)
        labels_df = cls._import_labelmapper_lut(lut_path)
        return cls(
            ngff_image=znii.ngff_image,
            axes_order=znii.axes_order,
            xyz_orientation=znii.xyz_orientation,
            labels_df=labels_df,
            omero=getattr(znii, "_omero", None),
        )

    def __attrs_post_init__(self):
        """Validate atlas consistency after initialization."""
        # Only validate if we have labels_df
        if self.labels_df is not None:
            self._validate_atlas()

    def _validate_atlas(self):
        """Validate that atlas components are consistent."""
        if self.labels_df is None:
            return  # Nothing to validate

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

        mask_ngff = nz.NgffImage(
            data=mask_data,
            dims=self.dseg.ngff_image.dims.copy(),
            scale=self.dseg.ngff_image.scale.copy(),
            translation=self.dseg.ngff_image.translation.copy(),
            name=f"{self.name}_masked",
        )

        return ZarrNii.from_ngff_image(
            mask_ngff,
            xyz_orientation=self.dseg.xyz_orientation,
            axes_order=self.dseg.axes_order,
            omero=self.dseg.omero,
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
        column_name: str = None,
        column_suffix: str = None,
    ) -> pd.DataFrame:
        """Aggregate image values by atlas regions.

        Args:
            image: Image to aggregate (must be compatible with atlas)
            aggregation_func: Aggregation function ('mean', 'sum', 'std', 'median', 'min', 'max')
            background_label: Label value to treat as background (excluded from results)
            column_name: String to use for column name. If None, uses f"{aggregation_func}_value"
            column_suffix: (Deprecated) String suffix to append to column name.
                Use column_name instead. If provided, column_name will be set to
                f"{aggregation_func}_{column_suffix}".

        Returns:
            DataFrame with columns: index, name, {column_name}, volume
            (e.g., with defaults: index, name, mean_value, volume)

        Raises:
            ValueError: If image and atlas are incompatible

        .. deprecated:: 0.2.0
            The `column_suffix` parameter is deprecated. Use `column_name` instead.
        """
        # Handle deprecated column_suffix parameter
        if column_suffix is not None:
            warnings.warn(
                "The 'column_suffix' parameter is deprecated and will be removed in a "
                "future version. Use 'column_name' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if column_name is None:
                column_name = f"{aggregation_func}_{column_suffix}"

        # Set default column name if not provided
        if column_name is None:
            column_name = f"{aggregation_func}_value"

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
                    self.label_column: int(label),
                    self.name_column: region_name,
                    column_name: agg_value,
                    "volume": volume,
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

        Notes:
            Labels present in the dseg image but not in feature_data will be
            mapped to 0.0. This can occur with downsampled dseg images or
            small ROIs where some regions are not represented.
        """
        # Validate input
        required_cols = [label_column, feature_column]
        missing_cols = [col for col in required_cols if col not in feature_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in feature_data: {missing_cols}")

        dseg_data = self.dseg.data.astype("int")  # dask array of labels

        # Get max label from feature_data
        max_label_features = int(feature_data[label_column].max())

        # Get max label from dseg image to ensure LUT is large enough
        # for all labels that actually exist in the image
        max_label_dseg = int(dseg_data.max().compute())

        # Use the larger of the two to size the LUT
        max_label = max(max_label_features, max_label_dseg)

        # make a dense lookup array sized to handle all labels in dseg
        lut = np.zeros(max_label + 1, dtype=np.float32)
        lut[feature_data[label_column].to_numpy(dtype=int)] = feature_data[
            feature_column
        ].to_numpy(dtype=float)

        # broadcast the mapping in one go
        feature_map = dseg_data.map_blocks(lambda block: lut[block], dtype=np.float32)

        feature_map_ngff = nz.NgffImage(
            data=feature_map,
            dims=self.dseg.ngff_image.dims.copy(),
            scale=self.dseg.ngff_image.scale.copy(),
            translation=self.dseg.ngff_image.translation.copy(),
            name=f"{self.name}_feature_map",
        )

        return ZarrNii.from_ngff_image(
            feature_map_ngff,
            xyz_orientation=self.dseg.xyz_orientation,
            axes_order=self.dseg.axes_order,
            omero=self.dseg.omero,
        )

    def get_region_bounding_box(
        self,
        region_ids: Union[int, str, List[Union[int, str]]] = None,
        regex: Optional[str] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get bounding box in physical coordinates for selected regions.

        This method computes the spatial extents (bounding box) of one or more
        atlas regions in physical/world coordinates. The returned bounding box
        can be used directly with the crop method to extract a subvolume
        containing the selected regions.

        Args:
            region_ids: Region identifier(s) to include in bounding box. Can be:
                - Single int: label index
                - Single str: region name or abbreviation
                - List[int/str]: multiple regions by index, name, or abbreviation
                - None: use regex parameter instead
            regex: Regular expression to match region names. If provided,
                region_ids must be None. Case-insensitive matching.

        Returns:
            Tuple of (bbox_min, bbox_max) where each is a tuple of (x, y, z)
            coordinates in physical/world space (mm). These can be passed
            directly to ZarrNii.crop() method with physical_coords=True.

        Raises:
            ValueError: If no regions match the selection criteria, or if both
                region_ids and regex are provided/omitted
            TypeError: If region_ids contains invalid types

        Examples:
            >>> # Get bounding box for single region
            >>> bbox_min, bbox_max = atlas.get_region_bounding_box("Hippocampus")
            >>> cropped = image.crop(bbox_min, bbox_max, physical_coords=True)
            >>>
            >>> # Get bounding box for multiple regions
            >>> bbox_min, bbox_max = atlas.get_region_bounding_box(["Hippocampus", "Amygdala"])
            >>>
            >>> # Use regex to select regions
            >>> bbox_min, bbox_max = atlas.get_region_bounding_box(regex="Hip.*")
            >>>
            >>> # Crop atlas itself to region
            >>> cropped_atlas = atlas.crop(bbox_min, bbox_max, physical_coords=True)

        Notes:
            - Bounding box is in physical coordinates (mm), not voxel indices
            - Axes ordering is relative to self.axes_order (e.g. ZYX for ome zarr)
            - The bounding box is the union of all selected regions
            - Use the returned values with crop(physical_coords=True)
        """
        import re

        import dask.array as da

        # Validate input parameters
        if region_ids is None and regex is None:
            raise ValueError("Must provide either region_ids or regex parameter")
        if region_ids is not None and regex is not None:
            raise ValueError("Cannot provide both region_ids and regex parameters")

        # Determine which labels to include
        selected_labels = []

        if regex is not None:
            # Match regions using regex
            pattern = re.compile(regex, re.IGNORECASE)
            for _, row in self.labels_df.iterrows():
                region_name = str(row[self.name_column])
                if pattern.search(region_name):
                    selected_labels.append(int(row[self.label_column]))

            if not selected_labels:
                raise ValueError(f"No regions matched regex pattern: {regex}")
        else:
            # Convert region_ids to list if single value
            if not isinstance(region_ids, list):
                region_ids = [region_ids]

            # Resolve each region identifier to label
            for region_id in region_ids:
                label = self._resolve_region_identifier(region_id)
                selected_labels.append(label)

        # Create union mask of all selected regions
        dseg_data = self.dseg.data
        mask = None
        for label in selected_labels:
            region_mask = dseg_data == label
            if mask is None:
                mask = region_mask
            else:
                mask = mask | region_mask

        # Find voxel coordinates where mask is True
        # da.where returns tuple of arrays (one per dimension in data array)
        indices = da.where(mask)

        # Compute the indices to get actual coordinates
        indices_computed = [idx.compute() for idx in indices]

        # Check if any voxels were found
        if any(idx.size == 0 for idx in indices_computed):
            raise ValueError(f"No voxels found for selected regions: {selected_labels}")

        # Get the spatial dimensions from dims (skip non-spatial like 'c', 't')
        spatial_dims_lower = [d.lower() for d in ["x", "y", "z"]]
        spatial_indices = []
        for i, dim in enumerate(self.dseg.dims):
            if dim.lower() in spatial_dims_lower:
                spatial_indices.append(i)

        # Extract spatial coordinates from indices
        # indices_computed has one array per dimension in data
        voxel_coords = []
        for spatial_idx in spatial_indices:
            voxel_coords.append(indices_computed[spatial_idx])

        # Get min and max for each spatial dimension
        voxel_mins = [int(coords.min()) for coords in voxel_coords]
        voxel_maxs = [
            int(coords.max()) + 1 for coords in voxel_coords
        ]  # +1 for inclusive max

        # Now we have voxel coordinates in the order they appear in dims
        # We don't need to convert to (x, y, z) order for physical coordinates
        #  since the affine should do this already..

        # make homog coords so we can matrix mult
        voxel_min_xyz = np.array(
            [
                voxel_mins[0],
                voxel_mins[1],
                voxel_mins[2],
                1.0,
            ]
        )
        voxel_max_xyz = np.array(
            [
                voxel_maxs[0],
                voxel_maxs[1],
                voxel_maxs[2],
                1.0,
            ]
        )

        # Transform to physical coordinates using affine
        affine_matrix = self.dseg.affine.matrix
        physical_min = affine_matrix @ voxel_min_xyz
        physical_max = affine_matrix @ voxel_max_xyz

        # Return as tuples of (x, y, z) in physical space
        bbox_min = tuple(physical_min[:3].tolist())
        bbox_max = tuple(physical_max[:3].tolist())

        return bbox_min, bbox_max

    def sample_region_patches(
        self,
        n_patches: int,
        region_ids: Union[int, str, List[Union[int, str]]] = None,
        regex: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[float, float, float]]:
        """Sample random coordinates (centers) within atlas regions.

        This method generates a list of center coordinates by randomly sampling
        voxels within the selected atlas regions. The returned coordinates are
        in physical/world space (mm) and can be used with crop_centered() to
        extract fixed-size patches for machine learning training or other workflows.

        Args:
            n_patches: Number of patch centers to sample
            region_ids: Region identifier(s) to sample from. Can be:
                - Single int: label index
                - Single str: region name or abbreviation
                - List[int/str]: multiple regions by index, name, or abbreviation
                - None: use regex parameter instead
            regex: Regular expression to match region names. If provided,
                region_ids must be None. Case-insensitive matching.
            seed: Random seed for reproducibility. If None, patches are sampled
                randomly each time.

        Returns:
            List of (x, y, z) coordinates in physical/world space (mm).
            Each coordinate represents the center of a potential patch and
            can be used with crop_centered() to extract fixed-size regions.

        Raises:
            ValueError: If no regions match the selection criteria, if both
                region_ids and regex are provided/omitted, or if n_patches is
                less than 1
            TypeError: If region_ids contains invalid types

        Examples:
            >>> # Sample 10 patch centers from hippocampus
            >>> centers = atlas.sample_region_patches(
            ...     n_patches=10,
            ...     region_ids="Hippocampus",
            ...     seed=42
            ... )
            >>> # Extract 256x256x256 voxel patches at each center
            >>> patches = image.crop_centered(centers, patch_size=(256, 256, 256))
            >>>
            >>> # Sample from multiple regions using list
            >>> centers = atlas.sample_region_patches(
            ...     n_patches=20,
            ...     region_ids=[1, 2, 3],
            ...     seed=42
            ... )
            >>>
            >>> # Sample using regex pattern
            >>> centers = atlas.sample_region_patches(
            ...     n_patches=5,
            ...     regex=".*cortex.*",
            ... )

        Notes:
            - Coordinates are in physical space (mm), not voxel indices
            - Centers are sampled uniformly from voxels within selected regions
            - Use crop_centered() to extract fixed-size patches around these centers
            - For ML training with fixed patch sizes (e.g., 256x256x256 voxels),
              use a lower-resolution atlas to define masks, then crop at higher
              resolution using physical coordinates
        """
        import re

        import dask.array as da

        # Validate input
        if n_patches < 1:
            raise ValueError(f"n_patches must be at least 1, got {n_patches}")

        if region_ids is None and regex is None:
            raise ValueError("Must provide either region_ids or regex parameter")
        if region_ids is not None and regex is not None:
            raise ValueError("Cannot provide both region_ids and regex parameters")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Determine which labels to include (reuse logic from get_region_bounding_box)
        selected_labels = []

        if regex is not None:
            # Match regions using regex
            pattern = re.compile(regex, re.IGNORECASE)
            for _, row in self.labels_df.iterrows():
                region_name = str(row[self.name_column])
                if pattern.search(region_name):
                    selected_labels.append(int(row[self.label_column]))

            if not selected_labels:
                raise ValueError(f"No regions matched regex pattern: {regex}")
        else:
            # Convert region_ids to list if single value
            if not isinstance(region_ids, list):
                region_ids = [region_ids]

            # Resolve each region identifier to label
            for region_id in region_ids:
                label = self._resolve_region_identifier(region_id)
                selected_labels.append(label)

        # Create union mask of all selected regions
        dseg_data = self.dseg.data
        mask = None
        for label in selected_labels:
            region_mask = dseg_data == label
            if mask is None:
                mask = region_mask
            else:
                mask = mask | region_mask

        # Find voxel coordinates where mask is True
        indices = da.where(mask)

        # Compute the indices to get actual coordinates
        indices_computed = [idx.compute() for idx in indices]

        # Check if any voxels were found
        if any(idx.size == 0 for idx in indices_computed):
            raise ValueError(f"No voxels found for selected regions: {selected_labels}")

        # Get number of valid voxels
        n_voxels = indices_computed[0].size

        # Sample random voxels
        # If n_patches > n_voxels, sample with replacement
        replace = n_patches > n_voxels
        sampled_indices = np.random.choice(n_voxels, size=n_patches, replace=replace)

        # Get spatial dimensions (skip non-spatial like 'c', 't')
        spatial_dims_lower = [d.lower() for d in ["x", "y", "z"]]
        spatial_indices = []
        for i, dim in enumerate(self.dseg.dims):
            if dim.lower() in spatial_dims_lower:
                spatial_indices.append(i)

        # Build voxel coordinates for sampled centers
        sampled_coords = []
        for spatial_idx in spatial_indices:
            sampled_coords.append(indices_computed[spatial_idx][sampled_indices])

        # Map to x, y, z order
        dim_to_coords = {}
        for i, spatial_idx in enumerate(spatial_indices):
            dim_name = self.dseg.dims[spatial_idx].lower()
            dim_to_coords[dim_name] = sampled_coords[i]

        # Get affine matrix
        affine_matrix = self.dseg.affine.matrix

        # Generate center coordinates in physical space
        centers = []
        for i in range(n_patches):
            # Get center voxel coordinates in (x, y, z) order
            center_voxel_xyz = np.array(
                [
                    dim_to_coords["x"][i],
                    dim_to_coords["y"][i],
                    dim_to_coords["z"][i],
                    1.0,
                ]
            )

            # Transform to physical coordinates
            center_physical = affine_matrix @ center_voxel_xyz
            center_xyz = center_physical[:3]

            # Convert to tuple
            center = tuple(center_xyz.tolist())
            centers.append(center)

        return centers

    def label_centroids(
        self,
        centroids: np.ndarray,
        include_names: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Map centroids to atlas labels using nearest neighbor interpolation.

        This method takes a set of centroids (typically from compute_centroids)
        and determines which atlas region each centroid falls into. It uses
        nearest neighbor interpolation to assign labels, making it robust to
        small coordinate mismatches.

        Args:
            centroids: Nx3 numpy array of centroid coordinates in physical space
                (typically output from compute_centroids). Each row is [x, y, z]
                in physical/world coordinates (mm). Can also be an empty array (0, 3).
            include_names: If True, includes region names from the labels dataframe
                in the output (default: True).

        Returns:
            tuple of two pandas DataFrames:
                1. centroids DataFrame with columns:
                    - x, y, z: Physical coordinates (in mm) of each centroid
                    - index: Integer label index from the atlas
                    - name (optional): Region name if include_names=True
                2. counts DataFrame with columns:
                    - index: Integer label index from the atlas
                    - name (optional): Region name if include_names=True
                    - count: Number of centroids in each region

        Notes:
            - Input centroids must be in the same physical space as the atlas
            - Points outside the atlas bounds receive index=0 (background)
            - Uses scipy.interpolate.interpn with method='nearest' for label lookup

        Examples:
            >>> # Compute centroids from a segmentation
            >>> centroids = binary_seg.compute_centroids()
            >>>
            >>> # Map centroids to atlas labels
            >>> df_centroids, df_counts = atlas.label_centroids(centroids)
            >>> print(df_centroids)
            >>> print(df_counts)
            >>>
            >>> # Filter to specific regions
            >>> hippocampus_points = df_centroids[
            ...     df_centroids['name'] == 'Hippocampus'
            ... ]
        """
        # Handle empty centroids array
        if centroids.shape[0] == 0:
            columns_centroids = ["x", "y", "z", "index"]
            columns_counts = ["index"]
            if include_names:
                columns_centroids.append("name")
                columns_counts.append("name")
            columns_counts.append("count")
            return (
                pd.DataFrame(columns=columns_centroids),
                pd.DataFrame(columns=columns_counts),
            )

        # Validate input shape
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"centroids must be Nx3 array, got shape {centroids.shape}"
            )

        # Get atlas data and affine
        dseg_data = self.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        affine_matrix = self.dseg.affine.matrix

        # Convert physical coordinates to voxel coordinates
        # centroids are in (x, y, z) order
        # Create homogeneous coordinates
        n_points = centroids.shape[0]
        centroids_homogeneous = np.column_stack(
            [centroids, np.ones((n_points, 1), dtype=np.float64)]
        )

        # Apply inverse affine transform
        affine_inv = np.linalg.inv(affine_matrix)
        voxel_coords_homogeneous = centroids_homogeneous @ affine_inv.T
        voxel_coords = voxel_coords_homogeneous[:, :3]

        # Create grid for interpn
        # Grid should be in the order of the data array dimensions
        # For ZarrNii, this is typically (z, y, x) or (c, z, y, x)
        # Remove channel dimension if present
        if dseg_data.ndim == 4:
            dseg_data = dseg_data[0]  # Remove channel dimension

        # Create coordinate grids for each dimension
        # interpn expects a tuple of 1D arrays representing the grid coordinates
        shape = dseg_data.shape
        grid = tuple(np.arange(s) for s in shape)

        # The inverse affine transform already converted physical (x, y, z) coordinates
        # to voxel coordinates in the order matching the data array axes_order.
        # So voxel_coords is already in the correct order for interpn!

        # Use interpn to get labels at the centroid locations
        label_at_points = interpn(
            grid,
            dseg_data,
            voxel_coords,
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        # Convert to integer labels
        label_at_points = label_at_points.astype(int)

        # Create DataFrame with x, y, z columns
        # centroids are in (x, y, z) order, so we use that order for output
        df_data = {
            "x": centroids[:, 0],  # x from centroids
            "y": centroids[:, 1],  # y from centroids
            "z": centroids[:, 2],  # z from centroids
            "index": label_at_points,
        }

        # Add region names if requested
        if include_names and self.labels_df is not None:
            # Create a lookup from index to name
            label_to_name = dict(
                zip(
                    self.labels_df[self.label_column],
                    self.labels_df[self.name_column],
                )
            )
            # Map labels to names, use 'Unknown' for labels not in lookup table
            df_data["name"] = [
                label_to_name.get(label, f"Unknown_Label_{label}")
                for label in label_at_points
            ]

        df_centroids = pd.DataFrame(df_data)

        # Create counts DataFrame - group by index and optionally name
        if include_names and self.labels_df is not None:
            df_counts = (
                df_centroids.groupby(["index", "name"]).size().reset_index(name="count")
            )
        else:
            df_counts = df_centroids.groupby(["index"]).size().reset_index(name="count")

        return (df_centroids, df_counts)
