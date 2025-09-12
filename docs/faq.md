# FAQ: Frequently Asked Questions

This page addresses common questions and provides troubleshooting tips for using ZarrNii.

---

## General Questions

### **1. What is ZarrNii?**
ZarrNii is a Python library that bridges the gap between OME-Zarr and NIfTI formats, enabling seamless conversion, transformations, and metadata handling for multidimensional biomedical images.

---

### **2. What formats does ZarrNii support?**
ZarrNii supports:
- **OME-Zarr**: A format for storing chunked, multidimensional microscopy images.
- **NIfTI**: A format commonly used for neuroimaging data.

---

### **3. Can ZarrNii handle large datasets?**
Yes! ZarrNii uses Dask arrays to handle datasets that don't fit into memory. Most transformations are lazy, meaning computations are only performed when explicitly triggered using `.compute()`.

---

## Installation Issues

### **1. I installed ZarrNii, but I can't import it.**
Ensure that ZarrNii is installed in the correct Python environment. Use `uv tree` or `pip show zarrnii` to verify the installation.

If you're still encountering issues, try reinstalling the library:
```bash
uv sync --dev
```

---

## Troubleshooting

---

## Performance Tips

### **1. How can I speed up transformations on large datasets?**
- Use appropriate chunk sizes with `.rechunk()` for operations like downsampling or interpolation.
- Trigger computations only when necessary using `.compute()`.

---

### **2. How do I optimize multiscale processing?**
For OME-Zarr datasets with multiscale pyramids:
1. Use the appropriate `level` when loading the dataset.
```python
znimg = ZarrNii.from_ome_zarr("path/to/dataset.zarr", level=2)
```

---

## Metadata Questions

### **1. How do I access OME-Zarr metadata?**
ZarrNii provides attributes for accessing metadata:
```python
print("Axes:", znimg.axes)
print("Coordinate transformations:", znimg.coordinate_transformations)
print("Omero metadata:", znimg.omero)
```

---

### **2. Does ZarrNii preserve metadata during transformations?**
Yes, ZarrNii updates the metadata to remain consistent with transformations like cropping, downsampling, or affine transformations.

---

## Getting Help

If you encounter issues not covered here:
1. Check the [API Reference](../reference.md) for detailed information about ZarrNii methods.
2. Open an issue on the [GitHub repository](https://github.com/yourusername/zarrnii/issues).

---

## Summary

This FAQ covers common questions about ZarrNii, troubleshooting tips, and best practices for working with large datasets and metadata. For more in-depth information, explore:
- [Examples](../examples/zarr_nifti.md)
- [API Reference](../reference.md)

