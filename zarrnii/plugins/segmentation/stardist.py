"""
StarDist deep learning segmentation plugin.

This module implements StarDist segmentation using pre-trained models for
nuclei and cell instance segmentation. Uses dask_relabeling for efficient
processing of large tiled images.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from .base import SegmentationPlugin


class StarDistSegmentation(SegmentationPlugin):
    """
    StarDist deep learning segmentation plugin.

    This plugin uses pre-trained StarDist models for nuclei/cell instance segmentation.
    It supports both 2D and 3D models with optional GPU acceleration and can process
    large images efficiently using dask_relabeling for tiled processing.

    Parameters:
        model_name: Name of the pre-trained model (e.g., '2D_versatile_fluo', '3D_demo')
        model_path: Path to custom model file (optional, overrides model_name)
        n_tiles: Number of tiles for processing (auto-determined if None)
        prob_thresh: Probability threshold for object detection (default: 0.5)
        nms_thresh: Non-maximum suppression threshold (default: 0.4)
        use_gpu: Whether to use GPU acceleration (default: None for auto-detect)
        normalize: Whether to normalize input (default: True)
        use_dask_relabeling: Whether to use dask_relabeling for large images \
(default: False)
        overlap: Overlap size for dask_relabeling tiles in pixels (default: 64)
    """

    def __init__(
        self,
        model_name: str = "2D_versatile_fluo",
        model_path: Optional[str] = None,
        n_tiles: Optional[Union[int, tuple]] = None,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        use_gpu: Optional[bool] = None,
        normalize: bool = True,
        # Default to False since dask-relabel not on PyPI
        use_dask_relabeling: bool = False,
        overlap: int = 64,
        **kwargs,
    ):
        """
        Initialize StarDist segmentation plugin.

        Args:
            model_name: Name of the pre-trained model
            model_path: Path to custom model file (optional)
            n_tiles: Number of tiles for processing
            prob_thresh: Probability threshold for object detection
            nms_thresh: Non-maximum suppression threshold
            use_gpu: Whether to use GPU acceleration
            normalize: Whether to normalize input
            use_dask_relabeling: Whether to use dask_relabeling for large images
            overlap: Overlap size for dask_relabeling tiles in pixels
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            n_tiles=n_tiles,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            use_gpu=use_gpu,
            normalize=normalize,
            use_dask_relabeling=use_dask_relabeling,
            overlap=overlap,
            **kwargs,
        )

        self.model_name = model_name
        self.model_path = model_path
        self.n_tiles = n_tiles
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.use_dask_relabeling = use_dask_relabeling
        self.overlap = overlap

        self._model = None
        self._model_loaded = False
        self._dask_relabeling_available = None  # Cache availability check

    def _is_dask_relabeling_available(self) -> bool:
        """Check if dask_relabeling is available."""
        if self._dask_relabeling_available is None:
            try:
                import dask.array as da  # noqa: F401
                from relabel import image2labels  # noqa: F401

                self._dask_relabeling_available = True
            except ImportError:
                self._dask_relabeling_available = False
        return self._dask_relabeling_available

    def _load_model(self):
        """Load the StarDist model on first use."""
        if self._model_loaded:
            return

        try:
            from stardist.models import StarDist2D, StarDist3D
        except ImportError:
            raise ImportError(
                "StarDist is not installed. Install with: "
                "pip install 'zarrnii[stardist]' or pip install stardist"
            )

        # Determine if we need 2D or 3D model based on model name
        is_3d = "3D" in self.model_name.upper() or (
            self.model_path and "3D" in self.model_path.upper()
        )

        try:
            if self.model_path:
                # Load custom model
                if is_3d:
                    self._model = StarDist3D(None, name=self.model_path)
                else:
                    self._model = StarDist2D(None, name=self.model_path)
            else:
                # Load pre-trained model
                if is_3d:
                    self._model = StarDist3D.from_pretrained(self.model_name)
                else:
                    self._model = StarDist2D.from_pretrained(self.model_name)

            self._model_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to load StarDist model: {e}")

    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Segment image using StarDist.

        Args:
            image: Input image as numpy array (2D or 3D)
            metadata: Optional metadata dictionary containing image information

        Returns:
            Labeled segmentation mask with unique integer labels for each instance.
            Background pixels have value 0.

        Raises:
            ValueError: If input image has invalid dimensions
            ImportError: If StarDist dependencies are not installed
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")

        # Load model on first use
        self._load_model()

        # Handle multi-channel images - StarDist typically expects single channel
        work_image = image
        if work_image.ndim > 3:
            # For 4D+ images, take the first channel/timepoint
            while work_image.ndim > 3:
                work_image = work_image[0]

        if work_image.ndim == 3:
            # Check if this is a multi-channel 2D image (C,H,W) or a 3D image (D,H,W)
            model_is_3d = "3D" in self.model_name.upper() or (
                self.model_path and "3D" in self.model_path.upper()
            )

            if not model_is_3d and work_image.shape[0] <= 4:
                # Likely channels-first format for 2D, take first channel
                work_image = work_image[0]

        # Use dask_relabeling for large images if enabled and available
        if (
            self.use_dask_relabeling
            and self._should_use_dask_relabeling(work_image)
            and self._is_dask_relabeling_available()
        ):
            return self._segment_with_dask_relabeling(work_image)
        else:
            return self._segment_direct(work_image)

    def _should_use_dask_relabeling(self, image: np.ndarray) -> bool:
        """
        Determine if dask_relabeling should be used based on image size.

        Args:
            image: Input image array

        Returns:
            True if image is large enough to benefit from dask_relabeling
        """
        # Use dask_relabeling for images larger than 2048x2048 (2D) or 512x512x512 (3D)
        if image.ndim == 2:
            return image.shape[0] > 2048 or image.shape[1] > 2048
        elif image.ndim == 3:
            return image.shape[0] > 512 or image.shape[1] > 512 or image.shape[2] > 512
        return False

    def _segment_with_dask_relabeling(self, image: np.ndarray) -> np.ndarray:
        """
        Segment using dask_relabeling for large images.

        Args:
            image: Input image array

        Returns:
            Labeled segmentation mask
        """
        try:
            import dask.array as da
            from relabel import image2labels
        except ImportError:
            warnings.warn(
                "dask_relabeling not available, falling back to direct segmentation. "
                "For tiled processing of large images, install dask-relabel from: "
                "https://github.com/TheJacksonLaboratory/dask_relabeling",
                stacklevel=2,
            )
            return self._segment_direct(image)

        # Convert to dask array if needed
        if not isinstance(image, da.Array):
            # Choose appropriate chunk size based on image dimensions
            if image.ndim == 2:
                chunk_size = (min(1024, image.shape[0]), min(1024, image.shape[1]))
            else:  # 3D
                chunk_size = (
                    min(128, image.shape[0]),
                    min(512, image.shape[1]),
                    min(512, image.shape[2]),
                )
            image_da = da.from_array(image, chunks=chunk_size)
        else:
            image_da = image

        # Create segmentation function for dask_relabeling
        def segment_chunk(chunk):
            """Segment a single chunk using StarDist."""
            return self._segment_direct(chunk)

        # Use dask_relabeling to process with overlap handling
        spatial_dims = 2 if image.ndim == 2 else 3
        labels_da = image2labels(
            image_da,
            seg_fn=segment_chunk,
            overlaps=self.overlap,
            spatial_dims=spatial_dims,
        )

        # Compute and return result
        return labels_da.compute()

    def _segment_direct(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image directly using StarDist model.

        Args:
            image: Input image array

        Returns:
            Labeled segmentation mask
        """
        # Determine n_tiles if not specified
        n_tiles = self.n_tiles
        if n_tiles is None:
            # Auto-determine tiles based on image size and available memory
            if image.ndim == 2:
                # For 2D images, use tiles if image is large
                if image.shape[0] > 1024 or image.shape[1] > 1024:
                    n_tiles = (2, 2)
                else:
                    n_tiles = (1, 1)
            else:  # 3D
                if image.shape[0] > 64 or image.shape[1] > 512 or image.shape[2] > 512:
                    n_tiles = (1, 2, 2)
                else:
                    n_tiles = (1, 1, 1)

        # Predict using StarDist model
        try:
            labels, _ = self._model.predict_instances(
                image,
                n_tiles=n_tiles,
                prob_thresh=self.prob_thresh,
                nms_thresh=self.nms_thresh,
                normalize=self.normalize,
            )

            # Ensure output is uint32 for large numbers of objects
            return labels.astype(np.uint32)

        except Exception as e:
            # If StarDist fails, return empty segmentation with same shape
            warnings.warn(f"StarDist segmentation failed: {e}", stacklevel=2)
            return np.zeros(image.shape, dtype=np.uint32)

    @property
    def name(self) -> str:
        """Return the name of the segmentation algorithm."""
        return f"StarDist ({self.model_name})"

    @property
    def description(self) -> str:
        """Return a description of the segmentation algorithm."""
        return (
            f"StarDist deep learning instance segmentation using "
            f"model '{self.model_name}'. "
            "Provides accurate cell and nuclei segmentation with pre-trained models. "
            f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}, "
            f"Dask relabeling: "
            f"{'enabled' if self.use_dask_relabeling else 'disabled'}."
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            self._load_model()

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_3d": hasattr(self._model, "config") and self._model.config.n_dim == 3,
            "prob_thresh": self.prob_thresh,
            "nms_thresh": self.nms_thresh,
            "use_gpu": self.use_gpu,
        }
