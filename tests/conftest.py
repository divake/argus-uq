"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_cell_id() -> str:
    """The cell used in Listing 1."""
    return "PTxOSG"
