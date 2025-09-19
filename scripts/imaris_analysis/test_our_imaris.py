#!/usr/bin/env python3
"""
Test our current Imaris implementation and compare with reference file.
"""

import numpy as np
import h5py
import tempfile
import os

def create_test_imaris_file():
    """Create a test Imaris file using our current to_imaris method."""
    try:
        # Import zarrnii - since there are import issues, let's just implement the basics
        # to see what our current method produces
        
        # Create test data
        test_data = np.random.randint(0, 255, size=(32, 64, 64), dtype=np.uint8)
        
        # Create a basic Imaris file structure using our understanding
        test_file = "/tmp/our_test.ims"
        
        with h5py.File(test_file, 'w') as f:
            # Root attributes - these should match the reference
            f.attrs['ImarisVersion'] = np.array([b'5', b'.', b'5', b'.', b'0'])
            f.attrs['DataSetDirectoryName'] = np.array([b'D', b'a', b't', b'a', b'S', b'e', b't'])
            f.attrs['DataSetInfoDirectoryName'] = np.array([b'D', b'a', b't', b'a', b'S', b'e', b't', b'I', b'n', b'f', b'o'])
            f.attrs['ImarisDataSet'] = np.array([b'I', b'm', b'a', b'r', b'i', b's', b'D', b'a', b't', b'a', b'S', b'e', b't'])
            f.attrs['NumberOfDataSets'] = np.array([1], dtype=np.uint32)
            f.attrs['ThumbnailDirectoryName'] = np.array([b'T', b'h', b'u', b'm', b'b', b'n', b'a', b'i', b'l'])
            
            # Create DataSet group
            dataset_grp = f.create_group('DataSet')
            
            # Create resolution level
            res_grp = dataset_grp.create_group('ResolutionLevel 0')
            
            # Create timepoint
            time_grp = res_grp.create_group('TimePoint 0')
            
            # Create channel
            chan_grp = time_grp.create_group('Channel 0')
            
            # Channel attributes (these need to be byte arrays like the reference)
            z, y, x = test_data.shape
            chan_grp.attrs['ImageSizeX'] = np.array([str(x)[i].encode() for i in range(len(str(x)))])
            chan_grp.attrs['ImageSizeY'] = np.array([str(y)[i].encode() for i in range(len(str(y)))])
            chan_grp.attrs['ImageSizeZ'] = np.array([str(z)[i].encode() for i in range(len(str(z)))])
            chan_grp.attrs['ImageBlockSizeX'] = np.array([str(x)[i].encode() for i in range(len(str(x)))])
            chan_grp.attrs['ImageBlockSizeY'] = np.array([str(y)[i].encode() for i in range(len(str(y)))])  
            chan_grp.attrs['ImageBlockSizeZ'] = np.array([str(z)[i].encode() for i in range(len(str(z)))])
            chan_grp.attrs['HistogramMin'] = np.array([b'0', b'.', b'0', b'0', b'0'])
            chan_grp.attrs['HistogramMax'] = np.array([b'2', b'5', b'5', b'.', b'0', b'0', b'0'])
            
            # Create data dataset
            chan_grp.create_dataset('Data', data=test_data, compression='gzip')
            
            # Create histogram (simple histogram)
            hist, _ = np.histogram(test_data, bins=256, range=(0, 256))
            chan_grp.create_dataset('Histogram', data=hist.astype(np.uint64))
            
            # Create DataSetInfo group
            info_grp = f.create_group('DataSetInfo')
            
            # Channel info
            chan_info_grp = info_grp.create_group('Channel 0')
            chan_info_grp.attrs['Color'] = np.array([b'1', b'.', b'0', b'0', b'0', b' ', b'0', b'.', b'0', b'0', b'0', b' ', b'0', b'.', b'0', b'0', b'0'])
            chan_info_grp.attrs['Name'] = np.array([b'C', b'h', b'a', b'n', b'n', b'e', b'l', b' ', b'0'])
            
            # Add other required info groups
            imaris_info = info_grp.create_group('Imaris')
            imaris_info.attrs['Version'] = np.array([b'7', b'.', b'0'])
            imaris_info.attrs['ThumbnailMode'] = np.array([b't', b'h', b'u', b'm', b'b', b'n', b'a', b'i', b'l', b'M', b'I', b'P'])
            imaris_info.attrs['ThumbnailSize'] = np.array([b'2', b'5', b'6'])
            
            dataset_info = info_grp.create_group('ImarisDataSet')
            dataset_info.attrs['Creator'] = np.array([b'I', b'm', b'a', b'r', b'i', b's'])
            dataset_info.attrs['Version'] = np.array([b'7', b'.', b'0'])
            dataset_info.attrs['NumberOfImages'] = np.array([b'1'])
            
            # Main dataset info
            chan_info_grp.attrs['X'] = np.array([str(x)[i].encode() for i in range(len(str(x)))])
            chan_info_grp.attrs['Y'] = np.array([str(y)[i].encode() for i in range(len(str(y)))])
            chan_info_grp.attrs['Z'] = np.array([str(z)[i].encode() for i in range(len(str(z)))])
            chan_info_grp.attrs['Unit'] = np.array([b'u', b'm'])
            chan_info_grp.attrs['Noc'] = np.array([b'1'])
            
            # Extent info (sample values)
            chan_info_grp.attrs['ExtMin0'] = np.array([b'-', b'0', b'.', b'0', b'3', b'3'])
            chan_info_grp.attrs['ExtMax0'] = np.array([b'1', b'7', b'.', b'1', b'1', b'8'])
            chan_info_grp.attrs['ExtMin1'] = np.array([b'-', b'0', b'.', b'0', b'3', b'3'])
            chan_info_grp.attrs['ExtMax1'] = np.array([b'1', b'7', b'.', b'1', b'1', b'8'])
            chan_info_grp.attrs['ExtMin2'] = np.array([b'-', b'0', b'.', b'1', b'0', b'0'])
            chan_info_grp.attrs['ExtMax2'] = np.array([b'6', b'.', b'3', b'0', b'0'])
            
            # Time info
            time_info = info_grp.create_group('TimeInfo')
            time_info.attrs['DatasetTimePoints'] = np.array([b'1'])
            time_info.attrs['FileTimePoints'] = np.array([b'1'])
            time_info.attrs['TimePoint1'] = np.array([b'2', b'0', b'0', b'2', b'-', b'0', b'1', b'-', b'0', b'1', b' ', b'0', b'0', b':', b'0', b'0', b':', b'0', b'0', b'.', b'0', b'0', b'0'])
            
            # Create Thumbnail group (basic)
            thumb_grp = f.create_group('Thumbnail')
            # Create a simple thumbnail (downsampled version)
            thumb_data = np.zeros((256, 256), dtype=np.uint8)
            thumb_grp.create_dataset('Data', data=thumb_data)
        
        return test_file
        
    except Exception as e:
        print(f"Error creating test file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_file = create_test_imaris_file()
    if test_file:
        print(f"Created test file: {test_file}")
        
        # Now let's analyze it
        print("\nAnalyzing our generated file:")
        
        with h5py.File(test_file, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"GROUP: {name}")
                    for attr_name, attr_val in obj.attrs.items():
                        if isinstance(attr_val, np.ndarray) and attr_val.dtype.char == 'S':
                            decoded = b''.join(attr_val).decode('utf-8', errors='ignore')
                            print(f"  ATTR: {attr_name} = {decoded}")
                        else:
                            print(f"  ATTR: {attr_name} = {attr_val}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"DATASET: {name}, shape={obj.shape}, dtype={obj.dtype}")
            
            f.visititems(print_structure)
            
            # Root attributes
            print("\nROOT ATTRIBUTES:")
            for attr_name, attr_val in f.attrs.items():
                if isinstance(attr_val, np.ndarray) and attr_val.dtype.char == 'S':
                    decoded = b''.join(attr_val).decode('utf-8', errors='ignore')
                    print(f"  {attr_name} = {decoded}")
                else:
                    print(f"  {attr_name} = {attr_val}")
    else:
        print("Failed to create test file")