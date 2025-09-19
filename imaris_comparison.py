#!/usr/bin/env python3
"""
Comprehensive comparison between reference Imaris file and our generated files.
This will help identify exactly what differences exist that prevent our files from being readable.
"""

import h5py
import numpy as np
import os
import tempfile
from pathlib import Path

def analyze_hdf5_structure(filepath, prefix=""):
    """Recursively analyze HDF5 structure and return detailed information."""
    info = {}
    
    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            full_path = f"{prefix}{name}" if prefix else name
            
            if isinstance(obj, h5py.Group):
                info[full_path] = {
                    'type': 'group',
                    'attrs': dict(obj.attrs.items()),
                    'keys': list(obj.keys())
                }
            elif isinstance(obj, h5py.Dataset):
                info[full_path] = {
                    'type': 'dataset',
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'attrs': dict(obj.attrs.items()),
                    'compression': obj.compression,
                    'compression_opts': obj.compression_opts,
                    'chunks': obj.chunks,
                    'fillvalue': obj.fillvalue
                }
        
        f.visititems(visitor)
        
        # Add root attributes
        info['ROOT'] = {
            'type': 'root',
            'attrs': dict(f.attrs.items()),
            'keys': list(f.keys())
        }
    
    return info

def format_attrs(attrs):
    """Format attributes for readable comparison."""
    formatted = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            try:
                formatted[key] = f"bytes: {value.decode('utf-8')}"
            except:
                formatted[key] = f"bytes: {value!r}"
        elif isinstance(value, np.ndarray):
            formatted[key] = f"array: shape={value.shape}, dtype={value.dtype}, data={value}"
        else:
            formatted[key] = f"{type(value).__name__}: {value}"
    return formatted

def compare_structures(ref_info, gen_info, name="COMPARISON"):
    """Compare two HDF5 structures and report differences."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {name}")
    print(f"{'='*60}")
    
    all_paths = set(ref_info.keys()) | set(gen_info.keys())
    
    for path in sorted(all_paths):
        ref_item = ref_info.get(path)
        gen_item = gen_info.get(path)
        
        print(f"\nPATH: {path}")
        print("-" * 40)
        
        if ref_item is None:
            print("❌ MISSING IN REFERENCE")
            print(f"Generated: {gen_item}")
        elif gen_item is None:
            print("❌ MISSING IN GENERATED")
            print(f"Reference: {ref_item}")
        else:
            # Compare types
            if ref_item['type'] != gen_item['type']:
                print(f"❌ TYPE MISMATCH: ref={ref_item['type']}, gen={gen_item['type']}")
            
            # Compare datasets
            if ref_item['type'] == 'dataset':
                if ref_item['shape'] != gen_item['shape']:
                    print(f"❌ SHAPE MISMATCH: ref={ref_item['shape']}, gen={gen_item['shape']}")
                if ref_item['dtype'] != gen_item['dtype']:
                    print(f"❌ DTYPE MISMATCH: ref={ref_item['dtype']}, gen={gen_item['dtype']}")
                if ref_item['compression'] != gen_item['compression']:
                    print(f"⚠️  COMPRESSION DIFF: ref={ref_item['compression']}, gen={gen_item['compression']}")
            
            # Compare attributes
            ref_attrs = format_attrs(ref_item['attrs'])
            gen_attrs = format_attrs(gen_item['attrs'])
            
            all_attr_keys = set(ref_attrs.keys()) | set(gen_attrs.keys())
            
            if all_attr_keys:
                print("ATTRIBUTES:")
                for attr_key in sorted(all_attr_keys):
                    ref_val = ref_attrs.get(attr_key, "MISSING")
                    gen_val = gen_attrs.get(attr_key, "MISSING")
                    
                    if ref_val != gen_val:
                        print(f"  ❌ {attr_key}:")
                        print(f"    REF: {ref_val}")
                        print(f"    GEN: {gen_val}")
                    else:
                        print(f"  ✅ {attr_key}: {ref_val}")
            
            if ref_item['type'] == 'group' and 'keys' in ref_item:
                ref_keys = set(ref_item['keys'])
                gen_keys = set(gen_item['keys'])
                if ref_keys != gen_keys:
                    missing_in_gen = ref_keys - gen_keys
                    extra_in_gen = gen_keys - ref_keys
                    if missing_in_gen:
                        print(f"❌ MISSING KEYS IN GENERATED: {missing_in_gen}")
                    if extra_in_gen:
                        print(f"⚠️  EXTRA KEYS IN GENERATED: {extra_in_gen}")

def create_test_imaris_file():
    """Create a test Imaris file using our current implementation."""
    # We'll need to import and use our current implementation
    # Let's create a simple test case
    
    print("Creating test Imaris file with our implementation...")
    
    # Create test data
    import numpy as np
    test_data = np.random.randint(0, 65535, size=(64, 64, 32), dtype=np.uint16)
    
    # Use our current to_imaris implementation
    try:
        from zarrnii import ZarrNii
        import dask.array as da
        
        # Create a ZarrNii object with test data
        darr = da.from_array(test_data, chunks=(32, 32, 16))
        
        # Create a temporary ZarrNii instance (this is a bit hacky but needed for testing)
        znimg = ZarrNii.__new__(ZarrNii)
        znimg.darr = darr
        znimg.axes_order = "ZYX"
        
        # Mock the affine transform
        from zarrnii.transform import AffineTransform
        import numpy as np
        affine_matrix = np.eye(4)
        affine_matrix[0, 0] = 1.0  # X spacing
        affine_matrix[1, 1] = 1.0  # Y spacing  
        affine_matrix[2, 2] = 1.0  # Z spacing
        znimg.affine = AffineTransform(affine_matrix)
        
        # Create test file
        test_file = "/tmp/test_generated.ims"
        znimg.to_imaris(test_file)
        
        return test_file
        
    except Exception as e:
        print(f"Error creating test file: {e}")
        return None

def main():
    reference_file = "/home/runner/work/zarrnii/zarrnii/tests/PtK2Cell.ims"
    
    if not os.path.exists(reference_file):
        print(f"Reference file not found: {reference_file}")
        return
    
    print("Analyzing reference Imaris file...")
    ref_info = analyze_hdf5_structure(reference_file)
    
    print("Creating test file with our implementation...")
    test_file = create_test_imaris_file()
    
    if test_file and os.path.exists(test_file):
        print("Analyzing generated Imaris file...")
        gen_info = analyze_hdf5_structure(test_file)
        
        # Compare structures
        compare_structures(ref_info, gen_info, "Reference vs Generated")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass
    else:
        print("Could not create test file for comparison")
        
        # At least show the reference structure
        print("\n" + "="*60)
        print("REFERENCE FILE STRUCTURE")
        print("="*60)
        
        for path, info in sorted(ref_info.items()):
            print(f"\n{path} ({info['type']})")
            if info['attrs']:
                attrs = format_attrs(info['attrs'])
                for k, v in attrs.items():
                    print(f"  {k}: {v}")
            if info['type'] == 'dataset':
                print(f"  Shape: {info['shape']}")
                print(f"  Dtype: {info['dtype']}")
                print(f"  Compression: {info['compression']}")

if __name__ == "__main__":
    main()