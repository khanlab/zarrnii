"""
Logging configuration for ZarrNii library.

This module provides logging utilities for ZarrNii that can be easily controlled
by calling scripts. It follows the standard Python library logging pattern where:

1. A NullHandler is added by default to prevent "No handlers found" warnings
2. Calling scripts can configure logging to their preferences
3. Debug-level logging is available for diagnosing issues (e.g., memory)

Example usage in calling scripts (e.g., Snakemake):
    >>> import logging
    >>> from zarrnii.logging import configure_logging
    >>>
    >>> # Simple setup with DEBUG level
    >>> configure_logging(level=logging.DEBUG)
    >>>
    >>> # Or configure your own handler
    >>> import logging
    >>> logging.getLogger("zarrnii").setLevel(logging.DEBUG)
    >>> logging.getLogger("zarrnii").addHandler(logging.StreamHandler())

Example usage for library-level logging:
    >>> from zarrnii.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.debug("This is debug info")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Union

# The main library logger
LIBRARY_LOGGER_NAME = "zarrnii"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for use within the zarrnii library.

    This function returns a logger that is a child of the main zarrnii logger,
    allowing for hierarchical logging control. Using this function ensures
    consistent logger naming across the library.

    Args:
        name: The logger name, typically `__name__` from the calling module.
              If None, returns the root zarrnii logger.

    Returns:
        A logger instance for the specified name

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing started")
        >>> logger.info("File loaded successfully")
    """
    if name is None:
        return logging.getLogger(LIBRARY_LOGGER_NAME)

    # If name already starts with 'zarrnii', use it directly
    if name.startswith(LIBRARY_LOGGER_NAME):
        return logging.getLogger(name)

    # Otherwise, make it a child of the zarrnii logger
    return logging.getLogger(f"{LIBRARY_LOGGER_NAME}.{name}")


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
    stream: Optional[object] = None,
) -> None:
    """
    Configure logging for the zarrnii library.

    This is a convenience function for calling scripts (like Snakemake workflows)
    to easily set up logging for zarrnii. It configures the root zarrnii logger
    with the specified settings.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, "DEBUG", "INFO")
        format_string: Custom format string for log messages.
                      Defaults to "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler: Custom handler to use. If None, a StreamHandler is created.
        stream: Stream for the default StreamHandler (default: sys.stderr).
                Only used if handler is None.

    Examples:
        >>> import logging
        >>> from zarrnii.logging import configure_logging
        >>>
        >>> # Enable DEBUG logging
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Use custom format
        >>> configure_logging(
        ...     level="DEBUG",
        ...     format_string="%(levelname)s: %(message)s"
        ... )
        >>>
        >>> # Use file handler
        >>> file_handler = logging.FileHandler("zarrnii.log")
        >>> configure_logging(level=logging.DEBUG, handler=file_handler)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Get the root zarrnii logger
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)

    # Set the level
    logger.setLevel(level)

    # Create or use provided handler
    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)

    # Set formatter if the handler doesn't have one
    if handler.formatter is None:
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

    # Set handler level to match logger level
    handler.setLevel(level)

    # Remove existing handlers to avoid duplicates if called multiple times
    logger.handlers.clear()

    # Add the handler
    logger.addHandler(handler)


def _setup_library_logging() -> None:
    """
    Set up library-level logging with NullHandler.

    This function is called when the module is imported to ensure that
    the zarrnii logger has a NullHandler, following Python logging best
    practices for libraries.

    This prevents the "No handlers could be found for logger 'zarrnii'"
    warning when the calling application doesn't configure logging.
    """
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    # Only add NullHandler if no handlers are configured
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


# Set up library logging when module is imported
_setup_library_logging()
