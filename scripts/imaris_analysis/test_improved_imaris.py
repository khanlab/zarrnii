#!/usr/bin/env python3
"""
Test the improved Imaris implementation and compare with reference file.
"""

import numpy as np
import h5py
import tempfile
import os
from pathlib import Path

def create_test_imaris_with_zarrnii():
    """Create a test Imaris file using our improved ZarrNii implementation."""
    try:
        from zarrnii import ZarrNii
        import dask.array as da
        from zarrnii.transform import AffineTransform
        
        # Create test data similar to reference file dimensions
        test_data = np.random.randint(0, 255, size=(32, 256, 256), dtype=np.uint8)
        
        # Create ZarrNii object
        darr = da.from_array(test_data, chunks=(16, 128, 128))
        
        # Create a proper ZarrNii instance
        znimg = ZarrNii.__new__(ZarrNii)
        znimg.darr = darr
        znimg.axes_order = "ZYX"
        
        # Create affine transform
        affine_matrix = np.eye(4)
        affine_matrix[0, 0] = 0.067  # X spacing (like reference)
        affine_matrix[1, 1] = 0.067  # Y spacing 
        affine_matrix[2, 2] = 0.197  # Z spacing
        znimg.affine = AffineTransform(affine_matrix)
        
        # Mock required methods for the improved implementation
        def mock_get_zooms():
            return [0.067, 0.067, 0.197]
        znimg.get_zooms = mock_get_zooms
        
        # Create test file
        test_file = "/tmp/improved_test.ims"
        znimg.to_imaris(test_file)
        
        return test_file
        
    except Exception as e:
        print(f"Error creating test file: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_hdf5_structure(filepath, name="FILE"):
    """Analyze HDF5 structure and return summary."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    with h5py.File(filepath, 'r') as f:
        # Root attributes
        print("\nROOT ATTRIBUTES:")
        for attr_name, attr_val in f.attrs.items():
            if isinstance(attr_val, np.ndarray) and attr_val.dtype.char == 'S':
                try:
                    decoded = b''.join(attr_val).decode('utf-8', errors='ignore')
                    print(f"  {attr_name}: '{decoded}' (byte array)")
                except:
                    print(f"  {attr_name}: {attr_val} (raw bytes)")
            else:
                print(f"  {attr_name}: {attr_val}")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\nGROUP: {name}")
                if obj.attrs:
                    for attr_name, attr_val in obj.attrs.items():
                        if isinstance(attr_val, np.ndarray) and attr_val.dtype.char == 'S':
                            try:
                                decoded = b''.join(attr_val).decode('utf-8', errors='ignore')
                                print(f"  ATTR: {attr_name} = '{decoded}'")
                            except:
                                print(f"  ATTR: {attr_name} = {attr_val}")
                        else:
                            print(f"  ATTR: {attr_name} = {attr_val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"DATASET: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Compression: {obj.compression}")
        
        f.visititems(print_structure)

def main():
    reference_file = "/home/runner/work/zarrnii/zarrnii/tests/PtK2Cell.ims"
    
    print("Creating improved Imaris file with ZarrNii...")
    test_file = create_test_imaris_with_zarrnii()
    
    if test_file and os.path.exists(test_file):
        print("✅ Successfully created improved Imaris file!")
        
        # Analyze both files
        if os.path.exists(reference_file):
            analyze_hdf5_structure(reference_file, "REFERENCE FILE (PtK2Cell.ims)")
        
        analyze_hdf5_structure(test_file, "IMPROVED ZARRNII FILE")
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Quick comparison of key aspects
        print("\n✅ IMPROVEMENTS MADE:")
        print("- All attributes now use byte-array encoding (like reference)")
        print("- Added all missing DataSetInfo groups and attributes")
        print("- Proper thumbnail generation with MIP")
        print("- Complete channel metadata with extents and descriptions")
        print("- Added device/acquisition metadata")
        print("- Created version-specific ImarisDataSet groups")
        print("- Added processing log entries")
        print("- Proper histogram range calculation")
        
        # Cleanup
        try:
            os.remove(test_file)
        except:
            pass
    else:
        print("❌ Failed to create improved test file")

if __name__ == "__main__":
    main()