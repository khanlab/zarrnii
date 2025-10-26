"""Tests for ZarrNii __repr__ method."""

import dask.array as da
import numpy as np

from zarrnii import ZarrNii


class TestRepr:
    """Test ZarrNii string representation."""

    def test_repr_contains_all_key_attributes(self):
        """Test that repr includes all key attributes."""
        data = np.random.rand(2, 64, 128, 256).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 32, 64, 128))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

        repr_str = repr(znimg)

        # Check that all key attributes are present
        assert "name='image'" in repr_str
        assert "shape=(2, 64, 128, 256)" in repr_str
        assert "dims=['c', 'z', 'y', 'x']" in repr_str
        assert "axes_order='ZYX'" in repr_str
        assert "xyz_orientation='RAS'" in repr_str
        assert "scale=" in repr_str
        # Check for dask array info
        assert "dtype=" in repr_str
        assert "chunksize=" in repr_str

    def test_repr_with_xyz_axes_order(self):
        """Test repr with XYZ axes order."""
        data = np.random.rand(3, 50, 100, 200).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 25, 50, 100))
        znimg = ZarrNii.from_darr(dask_data, orientation="LPI", axes_order="XYZ")

        repr_str = repr(znimg)

        assert "axes_order='XYZ'" in repr_str
        assert "xyz_orientation='LPI'" in repr_str
        assert "dims=['c', 'x', 'y', 'z']" in repr_str

    def test_repr_multiline_format(self):
        """Test that repr is multi-line for readability."""
        data = np.random.rand(1, 32, 64, 128).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 16, 32, 64))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAI", axes_order="ZYX")

        repr_str = repr(znimg)

        # Should be multi-line
        assert "\n" in repr_str
        # Should start with ZarrNii(
        assert repr_str.startswith("ZarrNii(")
        # Should end with )
        assert repr_str.strip().endswith(")")

    def test_repr_different_orientations(self):
        """Test repr with various orientation strings."""
        data = np.random.rand(1, 32, 32, 32).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

        orientations = ["RAS", "LPI", "RAI", "LPS"]
        for orient in orientations:
            znimg = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="ZYX")
            repr_str = repr(znimg)
            assert f"xyz_orientation='{orient}'" in repr_str

    def test_repr_is_consistent(self):
        """Test that repr is consistent across multiple calls."""
        data = np.random.rand(1, 16, 32, 64).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 8, 16, 32))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

        repr1 = repr(znimg)
        repr2 = repr(znimg)

        assert repr1 == repr2

    def test_repr_with_different_scales(self):
        """Test repr with different scale values."""
        data = np.random.rand(1, 32, 64, 128).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 16, 32, 64))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

        # Modify scale
        znimg.ngff_image.scale = {"z": 2.0, "y": 0.5, "x": 0.5}

        repr_str = repr(znimg)

        # Check that scale values appear in repr
        assert "scale=" in repr_str
        assert "2.0" in repr_str
        assert "0.5" in repr_str

    def test_repr_html_exists(self):
        """Test that _repr_html_ method exists for notebook display."""
        data = np.random.rand(1, 32, 64, 128).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 16, 32, 64))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

        # Check that _repr_html_ exists
        assert hasattr(znimg, "_repr_html_")
        assert callable(znimg._repr_html_)

    def test_repr_html_contains_metadata(self):
        """Test that _repr_html_ includes ZarrNii metadata."""
        data = np.random.rand(2, 64, 128, 256).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 32, 64, 128))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

        html = znimg._repr_html_()

        # Check that metadata appears in HTML
        assert "ZarrNii Object" in html
        assert "image" in html
        assert "shape" in html
        assert "axes_order" in html
        assert "RAS" in html
        assert "dtype" in html
        assert "chunksize" in html

    def test_repr_html_contains_dask_visualization(self):
        """Test that _repr_html_ includes dask array visualization."""
        data = np.random.rand(1, 32, 64, 128).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 16, 32, 64))
        znimg = ZarrNii.from_darr(dask_data, orientation="LPI", axes_order="XYZ")

        html = znimg._repr_html_()

        # Check that dask array info appears in HTML
        assert "Dask Array" in html
        # Dask's HTML representation contains table elements
        assert "<table" in html or "array" in html.lower()

    def test_repr_dask_info_accuracy(self):
        """Test that dtype and chunksize in repr are accurate."""
        data = np.random.rand(3, 50, 100, 200).astype(np.float16)
        dask_data = da.from_array(data, chunks=(1, 25, 50, 100))
        znimg = ZarrNii.from_darr(dask_data, orientation="RAI", axes_order="ZYX")

        repr_str = repr(znimg)

        # Verify dtype appears correctly
        assert "dtype=float16" in repr_str
        # Verify chunksize appears
        assert "chunksize=(1, 25, 50, 100)" in repr_str
