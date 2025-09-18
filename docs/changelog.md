# Changelog

All notable changes to ZarrNii will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Near-isotropic downsampling**: New `downsample_near_isotropic` parameter in `from_ome_zarr()` automatically downsamples dimensions with higher resolution to create more isotropic voxels
- Comprehensive documentation with examples and API reference
- Multi-resolution OME-Zarr support with pyramid creation
- Enhanced transformation pipeline with composite operations
- Memory-efficient processing with Dask integration
- Support for multi-channel and time-series data

### Changed
- **Migration from Poetry to uv**: Faster dependency management and builds with modern Python packaging
- **Automated versioning**: Version now derived from git tags using setuptools-scm
- **Enhanced CI/CD**: Updated workflows with trusted publishing to PyPI
- Improved performance for large dataset operations
- Enhanced metadata preservation across format conversions
- Optimized chunk sizing for better I/O performance

### Fixed
- **NumPy 2.0 Compatibility**: Fixed deprecated np.product usage
- Documentation build issues with missing files
- Improved error handling for malformed input files
- Better memory management for large transformations

## [0.1.0] - Initial Development

### Added
- Core ZarrNii class for unified OME-Zarr and NIfTI handling
- Affine and displacement transformation support
- Basic downsampling and upsampling operations
- Format conversion between OME-Zarr and NIfTI
- Spatial coordinate system management
- Integration with nibabel and zarr libraries

### Features
- Lazy loading with Dask arrays
- Metadata preservation during transformations
- Multi-scale image pyramid support
- Flexible resampling and interpolation methods
- Comprehensive test suite

### Documentation
- Getting started guide
- API reference
- Example workflows
- Installation instructions

## Development Notes

This project is under active development. The API may change between versions as we refine the interface based on user feedback and use cases.

## Contributing

See [Contributing](contributing.md) for information about contributing to ZarrNii development.
