"""Tests for atlas module functionality."""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from zarrnii import (
    AffineTransform,
    AmbiguousTemplateFlowQueryError,
    Atlas,
    ZarrNii,
    add_template_to_templateflow,
    get,
    get_template,
    import_lut_csv_as_tsv,
    import_lut_itksnap_as_tsv,
)


class TestAtlas:
    """Test suite for Atlas class."""

    @pytest.fixture
    def sample_atlas_data(self):
        """Create sample atlas data for testing."""
        # Create a simple 3D atlas with 3 regions plus background
        shape = (10, 10, 10)
        dseg_data = np.zeros(shape, dtype=np.int32)

        # Region 1: left half
        dseg_data[:, :, :5] = 1
        # Region 2: right half top
        dseg_data[:5, :, 5:] = 2
        # Region 3: right half bottom
        dseg_data[5:, :, 5:] = 3

        # Create affine matrix (1mm isotropic)
        affine = AffineTransform.from_array(np.eye(4))

        # Create ZarrNii from the data
        dseg = ZarrNii.from_darr(dseg_data, affine=affine)

        # Create labels DataFrame
        labels_df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "name": ["Background", "Left Region", "Right Top", "Right Bottom"],
                "abbreviation": ["BG", "LR", "RT", "RB"],
            }
        )

        return dseg, labels_df

    @pytest.fixture
    def sample_atlas(self, sample_atlas_data):
        """Create Atlas instance from sample data."""
        dseg, labels_df = sample_atlas_data
        return Atlas(dseg=dseg, labels_df=labels_df)

    def test_atlas_creation(self, sample_atlas_data):
        """Test basic Atlas creation."""
        dseg, labels_df = sample_atlas_data
        atlas = Atlas(dseg=dseg, labels_df=labels_df)

        assert atlas.dseg is dseg
        assert atlas.labels_df is labels_df
        assert atlas.label_column == "index"
        assert atlas.name_column == "name"
        assert atlas.abbrev_column == "abbreviation"

    def test_atlas_validation(self, sample_atlas_data):
        """Test atlas validation during creation."""
        dseg, labels_df = sample_atlas_data

        # Test missing required column
        bad_df = labels_df.drop(columns=["name"])
        with pytest.raises(ValueError, match="Missing required columns"):
            Atlas(dseg=dseg, labels_df=bad_df)

        # Test duplicate labels
        dup_df = labels_df.copy()
        dup_df.loc[len(dup_df)] = [1, "Duplicate", "DUP"]  # Duplicate index 1
        with pytest.raises(ValueError, match="Duplicate labels found"):
            Atlas(dseg=dseg, labels_df=dup_df)

    def test_atlas_properties(self, sample_atlas):
        """Test Atlas property methods."""
        atlas = sample_atlas

        # Test region_labels
        expected_labels = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(atlas.region_labels, expected_labels)

        # Test region_names
        expected_names = ["Background", "Left Region", "Right Top", "Right Bottom"]
        assert atlas.region_names == expected_names

        # Test region_abbreviations
        expected_abbrevs = ["BG", "LR", "RT", "RB"]
        assert atlas.region_abbreviations == expected_abbrevs

    def test_get_region_info(self, sample_atlas):
        """Test getting information for specific regions."""
        atlas = sample_atlas

        # Test valid region
        info = atlas.get_region_info(1)
        assert info["index"] == 1
        assert info["name"] == "Left Region"
        assert info["abbreviation"] == "LR"

        # Test invalid region
        with pytest.raises(ValueError, match="Label 999 not found"):
            atlas.get_region_info(999)

    def test_get_region_info_by_name_and_abbreviation(self, sample_atlas):
        """Test getting region information by name and abbreviation."""
        atlas = sample_atlas

        # Test lookup by name
        info_by_name = atlas.get_region_info("Left Region")
        assert info_by_name["index"] == 1
        assert info_by_name["name"] == "Left Region"
        assert info_by_name["abbreviation"] == "LR"

        # Test lookup by abbreviation
        info_by_abbrev = atlas.get_region_info("RT")
        assert info_by_abbrev["index"] == 2
        assert info_by_abbrev["name"] == "Right Top"
        assert info_by_abbrev["abbreviation"] == "RT"

        # Test lookup by index (backward compatibility)
        info_by_index = atlas.get_region_info(3)
        assert info_by_index["index"] == 3
        assert info_by_index["name"] == "Right Bottom"
        assert info_by_index["abbreviation"] == "RB"

        # Test invalid name
        with pytest.raises(ValueError, match="Region 'Invalid Name' not found"):
            atlas.get_region_info("Invalid Name")

        # Test invalid abbreviation
        with pytest.raises(ValueError, match="Region 'XX' not found"):
            atlas.get_region_info("XX")

    def test_get_region_mask(self, sample_atlas):
        """Test creating binary masks for regions."""
        atlas = sample_atlas

        # Test region 1 mask (left half)
        mask = atlas.get_region_mask(1)
        assert isinstance(mask, ZarrNii)
        assert mask.shape == atlas.dseg.shape

        # Check that mask is correct
        mask_data = mask.data
        if hasattr(mask_data, "compute"):
            mask_data = mask_data.compute()

        dseg_data = atlas.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        expected_mask = (dseg_data == 1).astype(np.uint8)
        np.testing.assert_array_equal(mask_data, expected_mask)

        # Test invalid region
        with pytest.raises(ValueError, match="Label 999 not found"):
            atlas.get_region_mask(999)

    def test_aggregate_image_by_regions(self, sample_atlas):
        """Test aggregating image values by atlas regions."""
        atlas = sample_atlas

        # Create test image with different values in each region
        img_data = np.zeros(atlas.dseg.shape, dtype=np.float64)

        dseg_data = atlas.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        img_data[dseg_data == 1] = 10.0  # Left region = 10
        img_data[dseg_data == 2] = 20.0  # Right top = 20
        img_data[dseg_data == 3] = 30.0  # Right bottom = 30

        test_image = ZarrNii.from_darr(img_data, affine=atlas.dseg.affine)

        # Test mean aggregation
        result = atlas.aggregate_image_by_regions(test_image, aggregation_func="mean")

        assert len(result) == 4  # 4 regions including background
        assert "mean_value" in result.columns
        assert "volume_mm3" in result.columns
        assert "voxel_count" in result.columns

        # Check values for specific regions
        left_row = result[result["index"] == 1].iloc[0]
        assert left_row["mean_value"] == 10.0
        assert left_row["volume_mm3"] == 500.0

    def test_create_feature_map(self, sample_atlas):
        """Test creating feature maps from region values."""
        atlas = sample_atlas

        # Create feature DataFrame
        feature_df = pd.DataFrame(
            {"index": [1, 2, 3], "feature_value": [100.0, 200.0, 300.0]}
        )

        # Create feature map
        feature_map = atlas.create_feature_map(feature_df, "feature_value")

        assert isinstance(feature_map, ZarrNii)
        assert feature_map.shape == atlas.dseg.shape

        # Check feature values
        map_data = feature_map.data
        if hasattr(map_data, "compute"):
            map_data = map_data.compute()

        dseg_data = atlas.dseg.data
        if hasattr(dseg_data, "compute"):
            dseg_data = dseg_data.compute()

        assert np.all(map_data[dseg_data == 1] == 100.0)
        assert np.all(map_data[dseg_data == 2] == 200.0)
        assert np.all(map_data[dseg_data == 3] == 300.0)
        assert np.all(map_data[dseg_data == 0] == 0.0)  # Background value


class TestAtlasFileIO:
    """Test suite for Atlas file I/O operations."""

    def test_from_files(self):
        """Test loading Atlas from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            dseg_path = tmpdir / "test_dseg.nii.gz"
            labels_path = tmpdir / "test_labels.tsv"

            # Create test segmentation image - create a simple nibabel image directly
            shape = (5, 5, 5)
            dseg_data = np.random.randint(0, 4, shape, dtype=np.int32)
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(dseg_data, affine)
            nib.save(nifti_img, str(dseg_path))

            # Create test labels file
            labels_df = pd.DataFrame(
                {
                    "index": [0, 1, 2, 3],
                    "name": ["Background", "Region1", "Region2", "Region3"],
                    "abbreviation": ["BG", "R1", "R2", "R3"],
                }
            )
            labels_df.to_csv(labels_path, sep="\t", index=False)

            # Load Atlas
            atlas = Atlas.from_files(dseg_path, labels_path)

            assert isinstance(atlas, Atlas)
            # ZarrNii from NIfTI may add a channel dimension
            expected_shapes = [shape, (1,) + shape]
            assert atlas.dseg.shape in expected_shapes
            assert len(atlas.labels_df) == 4


class TestLUTConversion:
    """Test suite for lookup table conversion functions."""

    def test_import_lut_csv_as_tsv(self):
        """Test CSV to TSV conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            csv_path = tmpdir / "test.csv"
            tsv_path = tmpdir / "test.tsv"

            # Create test CSV file (no header, as per SPIMquant format)
            csv_data = "LR,Left Region,1\nRT,Right Top,2\nRB,Right Bottom,3\n"
            with open(csv_path, "w") as f:
                f.write(csv_data)

            # Convert to TSV
            import_lut_csv_as_tsv(csv_path, tsv_path)

            # Check result
            assert tsv_path.exists()
            result_df = pd.read_csv(tsv_path, sep="\t")

            assert len(result_df) == 3
            assert list(result_df.columns) == ["index", "name", "abbreviation"]
            assert result_df["index"].tolist() == [1, 2, 3]

    def test_import_lut_itksnap_as_tsv(self):
        """Test ITK-SNAP to TSV conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            itksnap_path = tmpdir / "test.txt"
            tsv_path = tmpdir / "test.tsv"

            # Create test ITK-SNAP format file
            itksnap_data = """# ITK-SNAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Background
1    255    0    0   255    1   1   "Left Region"
2      0  255    0   255    1   1   "Right Top"
3      0    0  255   255    1   1   "Right Bottom"
"""
            with open(itksnap_path, "w") as f:
                f.write(itksnap_data)

            # Convert to TSV
            import_lut_itksnap_as_tsv(itksnap_path, tsv_path)

            # Check result
            assert tsv_path.exists()
            result_df = pd.read_csv(tsv_path, sep="\t")

            assert len(result_df) == 3
            assert list(result_df.columns) == ["index", "name", "abbreviation"]
            assert result_df["index"].tolist() == [1, 2, 3]


class TestTemplateFlowWrappers:
    """Test suite for TemplateFlow wrapper functionality."""

    def test_get_function_without_templateflow(self):
        """Test get() function raises ImportError when TemplateFlow unavailable."""
        with pytest.raises(ImportError, match="TemplateFlow is required"):
            get("MNI152NLin2009cAsym", suffix="T1w")

    def test_get_template_function_without_templateflow(self):
        """Test get_template() function raises ImportError when TemplateFlow unavailable."""
        with pytest.raises(ImportError, match="TemplateFlow is required"):
            get_template("MNI152NLin2009cAsym", "T1w")

    def test_add_template_to_templateflow_without_templateflow(self):
        """Test add_template_to_templateflow() raises ImportError when TemplateFlow unavailable."""
        with pytest.raises(ImportError, match="TemplateFlow is required"):
            add_template_to_templateflow("test.nii.gz", "test.tsv")


class TestAmbiguousTemplateFlowQueryError:
    """Test suite for AmbiguousTemplateFlowQueryError exception."""

    def test_ambiguous_templateflow_query_error_creation(self):
        """Test creating AmbiguousTemplateFlowQueryError."""
        # Test that AmbiguousTemplateFlowQueryError can be created
        files = ["/path/1.nii.gz", "/path/2.nii.gz"]
        error = AmbiguousTemplateFlowQueryError("MNI152", "T1w", files)

        assert "2 files" in str(error)
        assert error.template == "MNI152"
        assert error.suffix == "T1w"
        assert error.matching_files == files

    def test_ambiguous_templateflow_query_error_with_kwargs(self):
        """Test AmbiguousTemplateFlowQueryError with additional query parameters."""
        files = ["/path/1.nii.gz", "/path/2.nii.gz", "/path/3.nii.gz"]
        error = AmbiguousTemplateFlowQueryError(
            "MNI152", "T1w", files, resolution=1, cohort="01"
        )

        assert error.template == "MNI152"
        assert error.suffix == "T1w"
        assert error.matching_files == files
        assert error.query_kwargs == {"resolution": 1, "cohort": "01"}

        # Check error message truncation for many files
        assert "..." in str(error)  # Should truncate file list
