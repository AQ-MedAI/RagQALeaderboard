"""
PRGB Core Module

This module contains the core functionality for PRGB evaluation.
"""

__version__ = "1.0.0"
__author__ = "PRGB Team"

from .eval_main import SUPPORTED_DATASET, get_eval

__all__ = ["get_eval", "SUPPORTED_DATASET"]
