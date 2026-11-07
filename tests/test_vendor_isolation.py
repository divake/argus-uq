"""Vendor isolation contract.

Each vendor module under ``argus_uq.models.closed_source.{openai,
anthropic, google}`` MUST NOT import from any other vendor module.
This rule prevents subtle cross-vendor coupling (e.g. a prompt tweak
in one vendor silently propagating to another).

This test parses the AST of each vendor module and asserts no import
references another vendor.
"""

from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path

import pytest


VENDOR_NAMES = ("openai", "anthropic", "google")


def _vendor_module_path(name: str) -> Path:
    pkg = resources.files("argus_uq.models.closed_source")
    return Path(str(pkg.joinpath(f"{name}.py")))


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_vendor_does_not_import_other_vendors(vendor):
    """Vendor module ``X`` may not import from vendor modules ``Y != X``."""

    path = _vendor_module_path(vendor)
    if not path.is_file():
        pytest.skip(f"vendor module {vendor} not present")
    src = path.read_text()
    tree = ast.parse(src)
    other_vendors = {v for v in VENDOR_NAMES if v != vendor}

    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for other in other_vendors:
                if module.endswith(f"closed_source.{other}") or module == f"argus_uq.models.closed_source.{other}":
                    bad.append(f"{path.name}: from {module} ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for other in other_vendors:
                    if alias.name == f"argus_uq.models.closed_source.{other}":
                        bad.append(f"{path.name}: import {alias.name}")

    assert not bad, "Cross-vendor imports detected:\n" + "\n".join(bad)


def test_all_three_vendors_present():
    """All three vendor modules exist and parse."""

    for vendor in VENDOR_NAMES:
        path = _vendor_module_path(vendor)
        assert path.is_file(), f"missing vendor module: {vendor}.py"
        ast.parse(path.read_text())
