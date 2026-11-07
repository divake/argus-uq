"""Smoke tests for the four CLIs (no real model load)."""

from __future__ import annotations

import subprocess

import pytest

from argus_uq.cli import normalize, reproduce, score


pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _maybe_skip_if_no_data():
    try:
        import argus_uq

        argus_uq.load("PTxOSG")
    except FileNotFoundError:
        pytest.skip("data_release not populated")


def test_argus_score_runs():
    _maybe_skip_if_no_data()
    rc = score.main(["--cell", "PTxOSG", "--method", "saplma", "--seeds", "5"])
    assert rc == 0


def test_argus_score_unknown_method_returns_error():
    _maybe_skip_if_no_data()
    rc = score.main(["--cell", "PTxOSG", "--method", "not_a_method", "--seeds", "1"])
    assert rc == 1


def test_argus_score_unknown_cell_returns_error():
    rc = score.main(["--cell", "NotACell", "--method", "saplma", "--seeds", "1"])
    assert rc == 1


def test_argus_score_json_emit():
    _maybe_skip_if_no_data()
    rc = score.main(["--cell", "PTxOSG", "--method", "msp", "--seeds", "3", "--json"])
    assert rc == 0


def test_argus_reproduce_lists_tables():
    rc = reproduce.main(["--list"])
    assert rc == 0


def test_argus_reproduce_cells_table():
    rc = reproduce.main(["--table", "cells"])
    assert rc == 0


def test_argus_reproduce_known_table():
    _maybe_skip_if_no_data()
    rc = reproduce.main(["--table", "leaderboard"])
    assert rc == 0


def test_argus_reproduce_unknown_table():
    rc = reproduce.main(["--table", "not_a_table"])
    assert rc == 1


def test_argus_normalize():
    _maybe_skip_if_no_data()
    rc = normalize.main(["--cell", "PTxOSG", "--method", "saplma"])
    assert rc == 0


def test_argus_normalize_unknown_method():
    _maybe_skip_if_no_data()
    rc = normalize.main(["--cell", "PTxOSG", "--method", "not_a_method"])
    assert rc == 1


def test_console_scripts_installed():
    """Verify argus_score / argus_reproduce binaries are wired.

    Skipped if argus-uq isn't installed in editable mode (unusual on CI).
    """

    import shutil

    if shutil.which("argus_score") is None:
        pytest.skip("argus_score not in PATH (re-run pip install -e .)")
    out = subprocess.run(
        ["argus_score", "--help"], capture_output=True, text=True, timeout=10
    )
    assert out.returncode == 0
    assert "--cell" in out.stdout
