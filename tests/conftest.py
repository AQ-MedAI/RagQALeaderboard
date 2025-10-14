"""
Pytest configuration file for the RAGQA-Leaderboard project
"""

import os
import sys
from pathlib import Path

import pytest

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Configure pytest
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark integration tests
        if "integration" in item.name.lower() or "pipeline" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
