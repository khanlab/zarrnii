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
    ZarrNii,
    ZarrNiiAtlas,
    import_lut_csv_as_tsv,
    import_lut_itksnap_as_tsv,
)


class TestZarrNiiAtlas:
    """Test suite for ZarrNiiAtlas class."""

    @pytest.fixture
    def sample_atlas_data(self):
        """Create sample atlas data for testing."""
        # Create a simple 3D atlas with 3 regions plus background
        # Note: Adding channel dimension to match ZarrNii expectations
        shape = (1, 10, 10, 10)  # (c, z, y, x)
        dseg_data = np.zeros(shape, dtype=np.int32)

        # Region 1: left half
        dseg_data[0, :, :, :5] = 1
        # Region 2: right half top
        dseg_data[0, :5, :, 5:] = 2
        # Region 3: right half bottom
        dseg_data[0, 5:, :, 5:] = 3

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
        """Create ZarrNiiAtlas instance from sample data."""
        dseg, labels_df = sample_atlas_data
        return ZarrNiiAtlas.create_from_dseg(dseg, labels_df)

    def test_atlas_creation(self, sample_atlas_data):
        """Test basic ZarrNiiAtlas creation."""
        dseg, labels_df = sample_atlas_data
        atlas = ZarrNiiAtlas.create_from_dseg(dseg, labels_df)

        assert atlas.dseg is atlas  # dseg property returns self
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
            ZarrNiiAtlas.create_from_dseg(dseg, bad_df)

        # Test duplicate labels
        dup_df = labels_df.copy()
        dup_df.loc[len(dup_df)] = [1, "Duplicate", "DUP"]  # Duplicate index 1
        with pytest.raises(ValueError, match="Duplicate labels found"):
            ZarrNiiAtlas.create_from_dseg(dseg, dup_df)

    def test_atlas_properties(self, sample_atlas):
        """Test ZarrNiiAtlas basic properties."""
        atlas = sample_atlas

        # Test basic attributes
        assert hasattr(atlas, "dseg")
        assert hasattr(atlas, "labels_df")
        assert hasattr(atlas, "label_column")
        assert hasattr(atlas, "name_column")
        assert hasattr(atlas, "abbrev_column")

        # Test DataFrame contains expected data
        assert len(atlas.labels_df) == 4
        assert "index" in atlas.labels_df.columns
        assert "name" in atlas.labels_df.columns

    def test_get_region_info(self, sample_atlas):
        """Test getting information for specific regions."""
        atlas = sample_atlas

        # Test valid region
        info = atlas.get_region_info(1)
        assert info["index"] == 1
        assert info["name"] == "Left Region"
        assert info["abbreviation"] == "LR"

        # Test invalid region
        with pytest.raises(ValueError, match="Region with label 999 not found"):
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

        # Test invalid region - should raise ValueError from get_region_info
        with pytest.raises(ValueError, match="Region with label 999 not found"):
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

        assert len(result) == 3  # 3 regions (excluding background)
        assert "mean_value" in result.columns
        assert "volume_mm3" in result.columns

        # Check values for specific regions
        left_row = result[result["label"] == 1].iloc[0]
        assert left_row["mean_value"] == 10.0

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

    def test_get_region_bounding_box_single_region(self, sample_atlas):
        """Test getting bounding box for single region by index."""
        atlas = sample_atlas

        # Region 1 is left half: [:, :, :5]
        bbox_min, bbox_max = atlas.get_region_bounding_box(1)

        # Check that bbox is in physical coordinates (tuples of 3 floats)
        assert isinstance(bbox_min, tuple)
        assert isinstance(bbox_max, tuple)
        assert len(bbox_min) == 3
        assert len(bbox_max) == 3

        # With identity affine, physical coords should match voxel coords
        # Region 1 is [:, :, :5], so in XYZ: x=[0, 5), y=[0, 10), z=[0, 10)
        assert bbox_min == (0.0, 0.0, 0.0)
        assert bbox_max == (5.0, 10.0, 10.0)

    def test_get_region_bounding_box_by_name(self, sample_atlas):
        """Test getting bounding box by region name."""
        atlas = sample_atlas

        bbox_min, bbox_max = atlas.get_region_bounding_box("Left Region")

        # Should return same as by index
        assert bbox_min == (0.0, 0.0, 0.0)
        assert bbox_max == (5.0, 10.0, 10.0)

    def test_get_region_bounding_box_by_abbreviation(self, sample_atlas):
        """Test getting bounding box by region abbreviation."""
        atlas = sample_atlas

        bbox_min, bbox_max = atlas.get_region_bounding_box("RT")  # Right Top

        # Region 2 is [:5, :, 5:], so in XYZ: x=[5, 10), y=[0, 10), z=[0, 5)
        assert bbox_min == (5.0, 0.0, 0.0)
        assert bbox_max == (10.0, 10.0, 5.0)

    def test_get_region_bounding_box_multiple_regions(self, sample_atlas):
        """Test getting bounding box for multiple regions."""
        atlas = sample_atlas

        # Select regions 2 and 3 (both in right half)
        bbox_min, bbox_max = atlas.get_region_bounding_box([2, 3])

        # Region 2: [:5, :, 5:] (right half top)
        # Region 3: [5:, :, 5:] (right half bottom)
        # Union: [:, :, 5:], so in XYZ: x=[5, 10), y=[0, 10), z=[0, 10)
        assert bbox_min == (5.0, 0.0, 0.0)
        assert bbox_max == (10.0, 10.0, 10.0)

    def test_get_region_bounding_box_regex(self, sample_atlas):
        """Test getting bounding box using regex pattern."""
        atlas = sample_atlas

        # Match all "Right" regions (Right Top and Right Bottom)
        bbox_min, bbox_max = atlas.get_region_bounding_box(regex="Right.*")

        # Should match regions 2 and 3
        assert bbox_min == (5.0, 0.0, 0.0)
        assert bbox_max == (10.0, 10.0, 10.0)

    def test_get_region_bounding_box_regex_case_insensitive(self, sample_atlas):
        """Test that regex matching is case-insensitive."""
        atlas = sample_atlas

        bbox_min, bbox_max = atlas.get_region_bounding_box(regex="left.*")

        # Should match "Left Region"
        assert bbox_min == (0.0, 0.0, 0.0)
        assert bbox_max == (5.0, 10.0, 10.0)

    def test_get_region_bounding_box_with_crop(self, sample_atlas):
        """Test that bounding box output works with crop method."""
        atlas = sample_atlas

        # Get bounding box for region 1
        bbox_min, bbox_max = atlas.get_region_bounding_box(1)

        # Crop atlas using the bounding box
        # Note: crop expects voxel coordinates, but we're returning physical
        # For now, test that the method returns valid tuples
        assert isinstance(bbox_min, tuple) and len(bbox_min) == 3
        assert isinstance(bbox_max, tuple) and len(bbox_max) == 3

    def test_get_region_bounding_box_invalid_region(self, sample_atlas):
        """Test error handling for invalid region."""
        atlas = sample_atlas

        # Invalid region label should raise ValueError from _resolve_region_identifier
        # or from no voxels found
        with pytest.raises(ValueError):
            atlas.get_region_bounding_box(999)

    def test_get_region_bounding_box_no_match_regex(self, sample_atlas):
        """Test error when regex matches no regions."""
        atlas = sample_atlas

        with pytest.raises(ValueError, match="No regions matched regex pattern"):
            atlas.get_region_bounding_box(regex="NonexistentRegion.*")

    def test_get_region_bounding_box_invalid_params(self, sample_atlas):
        """Test error handling for invalid parameter combinations."""
        atlas = sample_atlas

        # Both region_ids and regex provided
        with pytest.raises(ValueError, match="Cannot provide both"):
            atlas.get_region_bounding_box(region_ids=1, regex="Left.*")

        # Neither region_ids nor regex provided
        with pytest.raises(ValueError, match="Must provide either"):
            atlas.get_region_bounding_box()

    def test_sample_region_patches_basic(self, sample_atlas):
        """Test basic patch sampling with physical size."""
        atlas = sample_atlas

        # Sample 5 patches of 1x1x1mm from region 1
        patches = atlas.sample_region_patches(
            n_patches=5,
            patch_size=(1.0, 1.0, 1.0),
            region_ids=1,
            physical_size=True,
            seed=42,
        )

        # Check return type and length
        assert isinstance(patches, list)
        assert len(patches) == 5

        # Check each patch is a tuple of two tuples
        for bbox_min, bbox_max in patches:
            assert isinstance(bbox_min, tuple)
            assert isinstance(bbox_max, tuple)
            assert len(bbox_min) == 3
            assert len(bbox_max) == 3

            # Check that bbox_max > bbox_min for all dimensions
            assert all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max))

            # Check patch size is approximately correct (1mm +/- floating point error)
            for i in range(3):
                patch_dim = bbox_max[i] - bbox_min[i]
                assert abs(patch_dim - 1.0) < 1e-6

    def test_sample_region_patches_by_name(self, sample_atlas):
        """Test patch sampling using region name."""
        atlas = sample_atlas

        patches = atlas.sample_region_patches(
            n_patches=3,
            patch_size=(0.5, 0.5, 0.5),
            region_ids="Left Region",
            physical_size=True,
            seed=42,
        )

        assert len(patches) == 3
        # All patches should be within region 1 bounds (x=[0, 5))
        for bbox_min, bbox_max in patches:
            # Check that patch centers are roughly in region 1
            center_x = (bbox_min[0] + bbox_max[0]) / 2
            assert 0 <= center_x < 5.0

    def test_sample_region_patches_regex(self, sample_atlas):
        """Test patch sampling using regex pattern."""
        atlas = sample_atlas

        # Sample from all "Right" regions (2 and 3)
        patches = atlas.sample_region_patches(
            n_patches=4,
            patch_size=(1.0, 1.0, 1.0),
            regex="Right.*",
            physical_size=True,
            seed=42,
        )

        assert len(patches) == 4

    def test_sample_region_patches_voxel_size(self, sample_atlas):
        """Test patch sampling with voxel size instead of physical size."""
        atlas = sample_atlas

        # Sample 2x2x2 voxel patches
        patches = atlas.sample_region_patches(
            n_patches=3,
            patch_size=(2, 2, 2),
            region_ids=1,
            physical_size=False,
            level=0,
            seed=42,
        )

        assert len(patches) == 3

        # With identity affine, 2 voxels = 2mm
        for bbox_min, bbox_max in patches:
            for i in range(3):
                patch_dim = bbox_max[i] - bbox_min[i]
                # Should be approximately 2.0mm
                assert abs(patch_dim - 2.0) < 1e-6

    def test_sample_region_patches_multiple_regions(self, sample_atlas):
        """Test patch sampling from multiple regions."""
        atlas = sample_atlas

        patches = atlas.sample_region_patches(
            n_patches=6,
            patch_size=(0.5, 0.5, 0.5),
            region_ids=[2, 3],
            physical_size=True,
            seed=42,
        )

        assert len(patches) == 6

    def test_sample_region_patches_reproducibility(self, sample_atlas):
        """Test that seed produces reproducible results."""
        atlas = sample_atlas

        patches1 = atlas.sample_region_patches(
            n_patches=5,
            patch_size=(1.0, 1.0, 1.0),
            region_ids=1,
            seed=123,
        )

        patches2 = atlas.sample_region_patches(
            n_patches=5,
            patch_size=(1.0, 1.0, 1.0),
            region_ids=1,
            seed=123,
        )

        # Should be identical
        assert patches1 == patches2

    def test_sample_region_patches_invalid_n_patches(self, sample_atlas):
        """Test error handling for invalid n_patches."""
        atlas = sample_atlas

        with pytest.raises(ValueError, match="n_patches must be at least 1"):
            atlas.sample_region_patches(
                n_patches=0,
                patch_size=(1.0, 1.0, 1.0),
                region_ids=1,
            )

    def test_sample_region_patches_invalid_region(self, sample_atlas):
        """Test error handling for invalid region."""
        atlas = sample_atlas

        with pytest.raises(ValueError):
            atlas.sample_region_patches(
                n_patches=5,
                patch_size=(1.0, 1.0, 1.0),
                region_ids=999,
            )

    def test_sample_region_patches_no_matching_regex(self, sample_atlas):
        """Test error handling when regex matches no regions."""
        atlas = sample_atlas

        with pytest.raises(ValueError, match="No regions matched regex pattern"):
            atlas.sample_region_patches(
                n_patches=5,
                patch_size=(1.0, 1.0, 1.0),
                regex="NonexistentPattern.*",
            )

    def test_sample_region_patches_invalid_params(self, sample_atlas):
        """Test error handling for invalid parameter combinations."""
        atlas = sample_atlas

        # Both region_ids and regex provided
        with pytest.raises(ValueError, match="Cannot provide both"):
            atlas.sample_region_patches(
                n_patches=5,
                patch_size=(1.0, 1.0, 1.0),
                region_ids=1,
                regex="Left.*",
            )

        # Neither region_ids nor regex provided
        with pytest.raises(ValueError, match="Must provide either"):
            atlas.sample_region_patches(
                n_patches=5,
                patch_size=(1.0, 1.0, 1.0),
            )

    def test_crop_with_patch_list(self, sample_atlas):
        """Test that crop works with list of bounding boxes from sample_region_patches."""
        atlas = sample_atlas

        # Sample some patches
        patches = atlas.sample_region_patches(
            n_patches=3,
            patch_size=(1.0, 1.0, 1.0),
            region_ids=1,
            physical_size=True,
            seed=42,
        )

        # Crop using the patch list
        cropped_list = atlas.crop(patches, physical_coords=True)

        # Check return type
        assert isinstance(cropped_list, list)
        assert len(cropped_list) == 3

        # Check each cropped result is a ZarrNii
        for cropped in cropped_list:
            assert isinstance(cropped, ZarrNii)
            # Cropped regions should be small (approximately 1mm cubed)
            # With identity affine and channel dim, shape should be (1, ~1, ~1, ~1)
            assert cropped.shape[0] == 1  # channel dimension

    def test_crop_batch_invalid_params(self, sample_atlas):
        """Test that crop rejects invalid parameter combinations for batch mode."""
        atlas = sample_atlas

        patches = [
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)),
        ]

        # Providing both list and bbox_max should raise error
        with pytest.raises(ValueError, match="bbox_max should be None"):
            atlas.crop(patches, bbox_max=(10.0, 10.0, 10.0))


class TestZarrNiiAtlasFileIO:
    """Test suite for ZarrNiiAtlas file I/O operations."""

    def test_from_files(self):
        """Test loading ZarrNiiAtlas from files."""
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

            # Load ZarrNiiAtlas
            atlas = ZarrNiiAtlas.from_files(dseg_path, labels_path)

            assert isinstance(atlas, ZarrNiiAtlas)
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

            # Create test CSV file with header
            csv_data = "index,name,abbreviation\n1,Left Region,LR\n2,Right Top,RT\n3,Right Bottom,RB\n"
            with open(csv_path, "w") as f:
                f.write(csv_data)

            # Convert to TSV
            import_lut_csv_as_tsv(csv_path, tsv_path)

            # Check result
            assert tsv_path.exists()
            result_df = pd.read_csv(tsv_path, sep="\t")

            assert len(result_df) == 3

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
        files = [
            "/path/1.nii.gz",
            "/path/2.nii.gz",
            "/path/3.nii.gz",
            "/path/4.nii.gz",
            "/path/5.nii.gz",
        ]
        error = AmbiguousTemplateFlowQueryError(
            "MNI152", "T1w", files, resolution=1, cohort="01"
        )

        assert error.template == "MNI152"
        assert error.suffix == "T1w"
        assert error.matching_files == files
        assert error.query_kwargs == {"resolution": 1, "cohort": "01"}

        # Check error message truncation for many files
        assert "..." in str(error)  # Should truncate file list
