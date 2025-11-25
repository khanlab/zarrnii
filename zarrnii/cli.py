"""Console scripts for ZarrNii format conversion.

This module provides command-line interfaces for converting between
OME-Zarr and NIfTI formats using simple wrapper scripts around the
ZarrNii library functionality.

Console Scripts:
    z2n: Convert OME-Zarr to NIfTI format
    n2z: Convert NIfTI to OME-Zarr format
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from zarrnii import ZarrNii


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated list of integers."""
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",")]


def parse_float_tuple(value: str) -> Tuple[float, float, float]:
    """Parse comma-separated tuple of 3 floats."""
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected exactly 3 comma-separated floats")
    return tuple(parts)


def parse_int_tuple(value: str) -> Tuple[int, ...]:
    """Parse comma-separated tuple of integers."""
    return tuple(int(x.strip()) for x in value.split(","))


def z2n():
    """Console script to convert OME-Zarr to NIfTI format.

    Wrapper around ZarrNii.from_ome_zarr().to_nifti().
    """
    parser = argparse.ArgumentParser(
        description="Convert OME-Zarr to NIfTI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  z2n input.ome.zarr output.nii.gz
  z2n input.ozx output.nii.gz --level 1
  z2n input.ome.zarr.zip output.nii.gz --level 1
  z2n input.ome.zarr output.nii.gz --channels 0,2 --axes-order ZYX
  
  # TIFF stack export
  z2n input.ome.zarr output_z{z:04d}.tif --tiff-channel 0
  z2n input.ome.zarr slices/brain_{z:03d}.tiff --tiff-timepoint 0 --tiff-no-compress
  z2n input.ome.zarr output_z{z:04d}.tif --tiff-dtype uint8 --tiff-no-rescale
        """,
    )

    # Positional arguments
    parser.add_argument("input", help="Input OME-Zarr path or store")
    parser.add_argument(
        "output",
        help="Output file path. Use .nii/.nii.gz for NIfTI or pattern with {z} for TIFF stack (e.g., 'output_z{z:04d}.tif')",
    )

    # from_ome_zarr parameters
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Pyramid level to load (0 = highest resolution) (default: 0)",
    )
    parser.add_argument(
        "--channels",
        type=parse_int_list,
        default=None,
        help="Comma-separated channel indices to load (e.g., '0,2,3')",
    )
    parser.add_argument(
        "--channel-labels",
        type=str,
        nargs="+",
        default=None,
        help="Channel names to load by label",
    )
    parser.add_argument(
        "--timepoints",
        type=parse_int_list,
        default=None,
        help="Comma-separated timepoint indices to load (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--axes-order",
        type=str,
        default="ZYX",
        choices=["ZYX", "XYZ"],
        help="Spatial axes order for processing (default: ZYX)",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default=None,
        help="Anatomical orientation string (default: None)",
    )
    parser.add_argument(
        "--downsample-near-isotropic",
        action="store_true",
        help="Apply near-isotropic downsampling",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="auto",
        help="Chunk specification for dask arrays (default: auto)",
    )
    parser.add_argument("--rechunk", action="store_true", help="Rechunk data arrays")

    # TIFF stack specific options
    parser.add_argument(
        "--tiff-channel",
        type=int,
        default=None,
        help="Channel index to export for TIFF stack (0-based). If not specified, saves multi-channel TIFFs",
    )
    parser.add_argument(
        "--tiff-timepoint",
        type=int,
        default=None,
        help="Timepoint index to export for TIFF stack (0-based). Required for multi-timepoint data",
    )
    parser.add_argument(
        "--tiff-no-compress",
        action="store_true",
        help="Disable LZW compression for TIFF files (default: compressed)",
    )
    parser.add_argument(
        "--tiff-dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "int16", "float32"],
        help="Output data type for TIFF files (default: uint16)",
    )
    parser.add_argument(
        "--tiff-no-rescale",
        action="store_true",
        help="Disable rescaling data to fit output dtype range (default: rescale enabled)",
    )

    args = parser.parse_args()

    # Validate input/output paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Validate output format and path
    output_format = None
    if output_path.suffix in [".nii", ".gz"] or str(output_path).endswith(".nii.gz"):
        output_format = "nifti"
    elif "{z" in str(output_path) and (output_path.suffix in [".tif", ".tiff"]):
        output_format = "tiff_stack"
    elif output_path.suffix in [".tif", ".tiff"]:
        print(
            f"Warning: TIFF output '{output_path}' should contain {{z}} format specifier for stack export (e.g., 'output_z{{z:04d}}.tif')"
        )
        output_format = "tiff_stack"
    else:
        # Default to NIfTI if uncertain
        output_format = "nifti"
        if output_path.suffix not in [".nii", ".gz"]:
            if not str(output_path).endswith(".nii.gz"):
                print(
                    f"Warning: Output '{output_path}' doesn't have .nii/.nii.gz/.tif/.tiff extension, assuming NIfTI"
                )

    try:
        # Process chunks argument
        chunks_param = args.chunks
        if chunks_param != "auto":
            try:
                chunks_param = parse_int_tuple(chunks_param)
            except ValueError:
                chunks_param = "auto"  # fallback to auto if parsing fails

        # Load from OME-Zarr
        print(f"Loading OME-Zarr from: {input_path}")
        znimg = ZarrNii.from_ome_zarr(
            store_or_path=str(input_path),
            level=args.level,
            channels=args.channels,
            channel_labels=args.channel_labels,
            timepoints=args.timepoints,
            axes_order=args.axes_order,
            orientation=args.orientation,
            downsample_near_isotropic=args.downsample_near_isotropic,
            chunks=chunks_param,
            rechunk=args.rechunk,
        )

        print(f"Loaded image with shape: {znimg.darr.shape}")

        # Save based on detected output format
        if output_format == "tiff_stack":
            print(f"Saving to TIFF stack: {output_path}")
            znimg.to_tiff_stack(
                str(output_path),
                channel=args.tiff_channel,
                timepoint=args.tiff_timepoint,
                compress=not args.tiff_no_compress,
                dtype=args.tiff_dtype,
                rescale=not args.tiff_no_rescale,
            )
        else:
            # Default to NIfTI
            print(f"Saving to NIfTI: {output_path}")
            znimg.to_nifti(str(output_path))

        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


def n2z():
    """Console script to convert NIfTI to OME-Zarr format.

    Wrapper around ZarrNii.from_nifti().to_ome_zarr().
    """
    parser = argparse.ArgumentParser(
        description="Convert NIfTI to OME-Zarr format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  n2z input.nii.gz output.ome.zarr
  n2z input.nii.gz output.ozx --max-layer 6
  n2z input.nii.gz output.ome.zarr.zip --max-layer 6
  n2z input.nii.gz output.ome.zarr --axes-order XYZ --as-ref
        """,
    )

    # Positional arguments
    parser.add_argument("input", help="Input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("output", help="Output OME-Zarr path or store")

    # from_nifti parameters
    parser.add_argument(
        "--chunks",
        type=str,
        default="auto",
        help="Chunk specification for dask arrays (default: auto)",
    )
    parser.add_argument(
        "--axes-order",
        type=str,
        default="XYZ",
        choices=["XYZ", "ZYX"],
        help="Spatial axes order for processing (default: XYZ)",
    )
    parser.add_argument("--name", type=str, default=None, help="Name for the dataset")
    parser.add_argument(
        "--as-ref", action="store_true", help="Create as reference without loading data"
    )
    parser.add_argument(
        "--zooms",
        type=parse_float_tuple,
        default=None,
        help="Custom voxel sizes as comma-separated floats (e.g., '2.0,2.0,2.0')",
    )

    # to_ome_zarr parameters
    parser.add_argument(
        "--max-layer",
        type=int,
        default=4,
        help="Maximum number of pyramid levels (default: 4)",
    )
    parser.add_argument(
        "--scale-factors",
        type=parse_int_list,
        default=None,
        help="Custom scale factors as comma-separated integers (e.g., '2,4,8')",
    )

    args = parser.parse_args()

    # Validate input/output paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Validate input extension
    if input_path.suffix not in [".nii", ".gz"] and not str(input_path).endswith(
        ".nii.gz"
    ):
        print(f"Warning: Input '{input_path}' doesn't appear to be a NIfTI file")

    try:
        # Process chunks argument
        chunks_param = args.chunks
        if chunks_param != "auto":
            try:
                chunks_param = parse_int_tuple(chunks_param)
            except ValueError:
                chunks_param = "auto"  # fallback to auto if parsing fails

        # Load from NIfTI
        print(f"Loading NIfTI from: {input_path}")
        znimg = ZarrNii.from_nifti(
            path=str(input_path),
            chunks=chunks_param,
            axes_order=args.axes_order,
            name=args.name,
            as_ref=args.as_ref,
            zooms=args.zooms,
        )

        print(f"Loaded image with shape: {znimg.darr.shape}")

        # Save to OME-Zarr
        print(f"Saving to OME-Zarr: {output_path}")
        znimg.to_ome_zarr(
            store_or_path=str(output_path),
            max_layer=args.max_layer,
            scale_factors=args.scale_factors,
        )

        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "z2n":
        sys.argv = sys.argv[1:]  # Remove the script name
        z2n()
    elif len(sys.argv) > 1 and sys.argv[1] == "n2z":
        sys.argv = sys.argv[1:]  # Remove the script name
        n2z()
    else:
        print("Usage: python -m zarrnii.cli [z2n|n2z] ...")
        sys.exit(1)
