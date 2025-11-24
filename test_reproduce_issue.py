"""Test to reproduce IndexError in create_feature_map when labels in dseg are not in feature table."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from zarrnii import ZarrNii, ZarrNiiAtlas


def test_create_feature_map_missing_labels_in_table():
    """Test create_feature_map when dseg has labels not in feature table.
    
    This reproduces the IndexError that occurs when:
    - The dseg image contains labels (e.g., 5, 10)
    - The feature_data table only contains a subset of labels (e.g., 1, 2)
    - The max label in dseg (10) is larger than max in table (2)
    
    This can happen with downsampled dseg images or small ROIs where some
    regions disappear.
    """
    # Create a dseg with labels 0, 1, 5, 10
    shape = (1, 10, 10, 10)
    dseg_data = da.zeros(shape, dtype=np.int32)
    
    # Region 1: small region
    dseg_data[0, 0:2, 0:2, 0:2] = 1
    
    # Region 5: another region (label not sequential)
    dseg_data[0, 5:7, 5:7, 5:7] = 5
    
    # Region 10: another region (large label value)
    dseg_data[0, 8:10, 8:10, 8:10] = 10
    
    # Create ZarrNii from the data
    dseg = ZarrNii.from_darr(dseg_data)
    
    # Create labels dataframe with all regions
    labels_df = pd.DataFrame({
        "index": [0, 1, 5, 10],
        "name": ["Background", "Region1", "Region5", "Region10"],
        "abbreviation": ["BG", "R1", "R5", "R10"],
    })
    
    atlas = ZarrNiiAtlas.create_from_dseg(dseg, labels_df)
    
    # Create feature DataFrame that only has labels 1 and 5 (missing 10!)
    # This simulates the case where a label exists in dseg but not in the feature table
    feature_df = pd.DataFrame({
        "index": [1, 5],  # Missing label 10 which exists in dseg!
        "feature_value": [100.0, 500.0]
    })
    
    # This should raise IndexError because lut[10] will be out of bounds
    # The lut will be sized to max(feature_df["index"]) + 1 = 6
    # But dseg contains label 10, so lut[10] will fail
    with pytest.raises(IndexError):
        feature_map = atlas.create_feature_map(feature_df, "feature_value")
        # Force computation to trigger the error
        _ = feature_map.data.compute()


def test_create_feature_map_with_missing_labels_robust():
    """Test that create_feature_map handles missing labels gracefully.
    
    After the fix, this should work and missing labels should map to 0.0.
    """
    # Create a dseg with labels 0, 1, 5, 10
    shape = (1, 10, 10, 10)
    dseg_data = da.zeros(shape, dtype=np.int32)
    
    dseg_data[0, 0:2, 0:2, 0:2] = 1
    dseg_data[0, 5:7, 5:7, 5:7] = 5
    dseg_data[0, 8:10, 8:10, 8:10] = 10
    
    dseg = ZarrNii.from_darr(dseg_data)
    
    labels_df = pd.DataFrame({
        "index": [0, 1, 5, 10],
        "name": ["Background", "Region1", "Region5", "Region10"],
    })
    
    atlas = ZarrNiiAtlas.create_from_dseg(dseg, labels_df)
    
    # Feature table only has labels 1 and 5
    feature_df = pd.DataFrame({
        "index": [1, 5],
        "feature_value": [100.0, 500.0]
    })
    
    # After fix, this should work
    feature_map = atlas.create_feature_map(feature_df, "feature_value")
    map_data = feature_map.data.compute()
    
    # Verify the expected values
    dseg_computed = dseg_data.compute()
    
    # Label 1 should map to 100.0
    assert np.all(map_data[dseg_computed == 1] == 100.0)
    
    # Label 5 should map to 500.0
    assert np.all(map_data[dseg_computed == 5] == 500.0)
    
    # Label 10 (not in feature table) should map to 0.0
    assert np.all(map_data[dseg_computed == 10] == 0.0)
    
    # Background (label 0) should map to 0.0
    assert np.all(map_data[dseg_computed == 0] == 0.0)


if __name__ == "__main__":
    print("Testing IndexError reproduction...")
    try:
        test_create_feature_map_missing_labels_in_table()
        print("✓ IndexError reproduction test passed")
    except AssertionError:
        print("✗ IndexError was not raised - test failed")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\nTesting robust implementation (will fail until fix is applied)...")
    try:
        test_create_feature_map_with_missing_labels_robust()
        print("✓ Robust test passed")
    except Exception as e:
        print(f"✗ Robust test failed: {e}")
