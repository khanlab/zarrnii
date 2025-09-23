"""Tests for atlas module functionality."""

import tempfile
from pathlib import Path

import nibabel as nib

import numpy as np

import pandas as pd

import pytest

from zarrnii import (
    AffineTransform,
    Atlas,
    ZarrNii,
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

    def test_get_region_volume(self, sample_atlas):
        """Test calculating region volumes."""
        atlas = sample_atlas

        # Region 1 should be 500 voxels * 1 mm³/voxel = 500 mm³
        volume = atlas.get_region_volume(1)
        assert volume == 500.0

        # Region 2 should be 250 voxels * 1 mm³/voxel = 250 mm³
        volume = atlas.get_region_volume(2)
        assert volume == 250.0

        # Test invalid region
        with pytest.raises(ValueError, match="Label 999 not found"):
            atlas.get_region_volume(999)

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
        bg_row = result[result["index"] == 0].iloc[0]
        # Background might be NaN if no background voxels have values
        assert bg_row["mean_value"] == 0.0 or pd.isna(bg_row["mean_value"])

        left_row = result[result["index"] == 1].iloc[0]
        assert left_row["mean_value"] == 10.0
        assert left_row["volume_mm3"] == 500.0

        # Test different aggregation functions
        sum_result = atlas.aggregate_image_by_regions(
            test_image, aggregation_func="sum"
        )
        assert "sum_value" in sum_result.columns

        # Test with specific regions only
        subset_result = atlas.aggregate_image_by_regions(
            test_image, aggregation_func="mean", regions=[1, 2]
        )
        assert len(subset_result) == 2

    def test_aggregate_incompatible_images(self, sample_atlas):
        """Test error handling for incompatible images."""
        atlas = sample_atlas

        # Create image with different shape
        wrong_shape_data = np.zeros((5, 5, 5), dtype=np.float64)
        wrong_image = ZarrNii.from_darr(
            wrong_shape_data, affine=AffineTransform.from_array(np.eye(4))
        )

        with pytest.raises(ValueError, match="doesn't match atlas shape"):
            atlas.aggregate_image_by_regions(wrong_image)

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

    def test_get_summary_statistics(self, sample_atlas):
        """Test getting summary statistics for all regions."""
        atlas = sample_atlas

        summary = atlas.get_summary_statistics()

        assert len(summary) == 4  # 4 regions
        assert "volume_mm3" in summary.columns
        assert "voxel_count" in summary.columns
        assert "name" in summary.columns

        # Check specific values
        left_region = summary[summary["index"] == 1].iloc[0]
        assert left_region["volume_mm3"] == 500.0
        assert left_region["voxel_count"] == 500
        assert left_region["name"] == "Left Region"

    def test_atlas_repr(self, sample_atlas):
        """Test string representation of Atlas."""
        atlas = sample_atlas
        repr_str = repr(atlas)

        assert "Atlas(" in repr_str
        assert "n_regions=4" in repr_str
        assert "dseg_shape=(10, 10, 10)" in repr_str


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
            assert atlas.labels_df["name"].tolist() == [
                "Background",
                "Region1",
                "Region2",
                "Region3",
            ]

    def test_from_files_missing_files(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError, match="Segmentation file not found"):
            Atlas.from_files("nonexistent_dseg.nii.gz", "nonexistent_labels.tsv")


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
            assert result_df["name"].tolist() == [
                "Left Region",
                "Right Top",
                "Right Bottom",
            ]

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
            assert result_df["name"].tolist() == [
                "Left Region",
                "Right Top",
                "Right Bottom",
            ]
            assert result_df["abbreviation"].tolist() == ["R001", "R002", "R003"]


class TestAtlasEdgeCases:
    """Test suite for Atlas edge cases and error conditions."""

    @pytest.fixture
    def sample_atlas_data(self):
        """Create sample atlas data for testing."""
        # Create a simple 3D atlas with 3 regions plus background
        shape = (5, 5, 5)
        dseg_data = np.zeros(shape, dtype=np.int32)

        # Region 1: left half
        dseg_data[:, :, :3] = 1
        # Region 2: right half
        dseg_data[:, :, 3:] = 2

        # Create affine matrix (1mm isotropic)
        affine = AffineTransform.from_array(np.eye(4))

        # Create ZarrNii from the data
        dseg = ZarrNii.from_darr(dseg_data, affine=affine)

        # Create labels DataFrame
        labels_df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "name": ["Background", "Left Region", "Right Region"],
                "abbreviation": ["BG", "LR", "RR"],
            }
        )

        return dseg, labels_df

    def test_atlas_with_custom_columns(self):
        """Test Atlas with custom column names."""
        # Create simple test data
        shape = (5, 5, 5)
        dseg_data = np.random.randint(0, 3, shape, dtype=np.int32)
        dseg = ZarrNii.from_darr(
            dseg_data, affine=AffineTransform.from_array(np.eye(4))
        )

        # Create DataFrame with custom column names
        labels_df = pd.DataFrame(
            {
                "label_id": [0, 1, 2],
                "region_name": ["Background", "Region1", "Region2"],
                "short_name": ["BG", "R1", "R2"],
            }
        )

        # Create Atlas with custom column mapping
        atlas = Atlas(
            dseg=dseg,
            labels_df=labels_df,
            label_column="label_id",
            name_column="region_name",
            abbrev_column="short_name",
        )

        assert atlas.label_column == "label_id"
        assert atlas.name_column == "region_name"
        assert atlas.abbrev_column == "short_name"

        # Test that methods work with custom columns
        info = atlas.get_region_info(1)
        assert info["label_id"] == 1
        assert info["region_name"] == "Region1"

    def test_atlas_without_abbreviations(self):
        """Test Atlas without abbreviation column."""
        shape = (5, 5, 5)
        dseg_data = np.random.randint(0, 3, shape, dtype=np.int32)
        dseg = ZarrNii.from_darr(
            dseg_data, affine=AffineTransform.from_array(np.eye(4))
        )

        # DataFrame without abbreviation column
        labels_df = pd.DataFrame(
            {"index": [0, 1, 2], "name": ["Background", "Region1", "Region2"]}
        )

        atlas = Atlas(dseg=dseg, labels_df=labels_df, abbrev_column="missing_column")

        # Should return empty strings for abbreviations
        abbrevs = atlas.region_abbreviations
        assert abbrevs == ["", "", ""]

    def test_aggregation_with_invalid_function(self, sample_atlas_data):
        """Test error handling for invalid aggregation functions."""
        dseg, labels_df = sample_atlas_data
        atlas = Atlas(dseg=dseg, labels_df=labels_df)

        # Create dummy image
        img_data = np.ones(dseg.shape, dtype=np.float64)
        test_image = ZarrNii.from_darr(img_data, affine=dseg.affine)

        with pytest.raises(ValueError, match="Unknown aggregation function"):
            atlas.aggregate_image_by_regions(
                test_image, aggregation_func="invalid_func"
            )

    def test_feature_map_missing_columns(self, sample_atlas_data):
        """Test error handling for missing columns in feature DataFrame."""
        dseg, labels_df = sample_atlas_data
        atlas = Atlas(dseg=dseg, labels_df=labels_df)

        # DataFrame missing label column
        bad_df = pd.DataFrame({"feature": [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain 'index' column"):
            atlas.create_feature_map(bad_df, "feature")

        # DataFrame missing feature column
        bad_df = pd.DataFrame({"index": [1, 2, 3]})

        with pytest.raises(ValueError, match="Feature column 'missing' not found"):
            atlas.create_feature_map(bad_df, "missing")
