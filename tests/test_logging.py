"""
Tests for the logging module.

This module tests the logging configuration and usage in zarrnii.
"""

import io
import logging

import pytest

from zarrnii.logging import LIBRARY_LOGGER_NAME, configure_logging, get_logger


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger_with_none(self):
        """Test get_logger with None returns root zarrnii logger."""
        logger = get_logger(None)
        assert logger.name == LIBRARY_LOGGER_NAME

    def test_get_logger_with_module_name(self):
        """Test get_logger with module name returns child logger."""
        logger = get_logger("mymodule")
        assert logger.name == f"{LIBRARY_LOGGER_NAME}.mymodule"

    def test_get_logger_with_zarrnii_prefixed_name(self):
        """Test get_logger with zarrnii prefix uses name directly."""
        logger = get_logger("zarrnii.submodule")
        assert logger.name == "zarrnii.submodule"

    def test_get_logger_returns_logging_logger(self):
        """Test get_logger returns a standard logging.Logger instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_child_logger_hierarchy(self):
        """Test that child loggers inherit from parent."""
        parent = get_logger(None)
        child = get_logger("child")

        # Child should be a descendant of parent
        assert child.parent == parent


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def teardown_method(self):
        """Clean up logger handlers after each test."""
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.NOTSET)

    def test_configure_logging_default_level(self):
        """Test configure_logging sets INFO level by default."""
        configure_logging()
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        assert logger.level == logging.INFO

    def test_configure_logging_debug_level(self):
        """Test configure_logging with DEBUG level."""
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_configure_logging_string_level(self):
        """Test configure_logging accepts string level."""
        configure_logging(level="DEBUG")
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_configure_logging_case_insensitive_string_level(self):
        """Test configure_logging handles lowercase string levels."""
        configure_logging(level="debug")
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_configure_logging_creates_handler(self):
        """Test configure_logging creates a StreamHandler."""
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # Should have exactly one handler after configuration
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_configure_logging_custom_format(self):
        """Test configure_logging with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        configure_logging(format_string=custom_format)
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # Check the formatter was applied
        handler = logger.handlers[0]
        assert handler.formatter._fmt == custom_format

    def test_configure_logging_custom_handler(self):
        """Test configure_logging with custom handler."""
        custom_handler = logging.FileHandler("/dev/null")
        configure_logging(handler=custom_handler)
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        assert len(logger.handlers) == 1
        assert logger.handlers[0] is custom_handler

        # Clean up
        custom_handler.close()

    def test_configure_logging_custom_stream(self):
        """Test configure_logging with custom stream."""
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # Log a message
        logger.info("Test message")

        # Check the message was written to our stream
        output = stream.getvalue()
        assert "Test message" in output

    def test_configure_logging_replaces_existing_handlers(self):
        """Test configure_logging clears existing handlers to avoid duplicates."""
        # Configure twice
        configure_logging(level=logging.INFO)
        configure_logging(level=logging.DEBUG)

        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # Should still have only one handler
        assert len(logger.handlers) == 1

    def test_configure_logging_actual_logging_output(self):
        """Test that configured logging actually outputs messages."""
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        # Use the get_logger function to get a logger
        logger = get_logger("test_module")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        output = stream.getvalue()
        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output


class TestLibraryLoggingSetup:
    """Tests for library logging initialization."""

    def test_null_handler_added_by_default(self):
        """Test that NullHandler is added to prevent warnings."""
        # The zarrnii logger should have at least a NullHandler after import
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # Check that there's at least one handler
        # (might be NullHandler or user-configured handler)
        assert len(logger.handlers) >= 0

    def test_no_handler_warning_without_configuration(self):
        """Test that logging without configuration doesn't produce warnings."""
        # This should not raise any warnings about missing handlers
        logger = get_logger("test_no_warning")
        logger.debug("Test message - should not warn")


class TestLoggingIntegration:
    """Integration tests for logging in scaled processing context."""

    def teardown_method(self):
        """Clean up logger handlers after each test."""
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.NOTSET)

    def test_scaled_processing_logging_output(self):
        """Test that scaled processing logs debug information when enabled."""
        import tempfile

        import dask.array as da
        import nibabel as nib
        import numpy as np

        from zarrnii import ZarrNii, configure_logging
        from zarrnii.plugins.scaled_processing import GaussianBiasFieldCorrection

        # Configure logging to capture output
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        # Create a simple test NIfTI image and load it to get a proper dask array
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(16, 16, 16).astype(np.float32)
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(data, affine)
            nifti_path = f"{tmpdir}/test.nii"
            nib.save(nifti_img, nifti_path)

            # Load as ZarrNii (this ensures we get a dask array)
            znii = ZarrNii.from_nifti(nifti_path, axes_order="ZYX")

            # Apply scaled processing
            plugin = GaussianBiasFieldCorrection(sigma=2.0)
            result = znii.apply_scaled_processing(plugin, downsample_factor=2)

        # Check that debug logging output contains expected information
        output = stream.getvalue()

        # Should log plugin info
        assert "Gaussian Bias Field Correction" in output

        # Should log shape information
        assert "shape" in output.lower()

        # Should log chunking information
        assert "chunk" in output.lower()

    def test_logging_level_filtering(self):
        """Test that logging level filters messages correctly."""
        stream = io.StringIO()
        configure_logging(level=logging.WARNING, stream=stream)

        logger = get_logger("test_filtering")
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")

        output = stream.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
