"""Shared pytest fixtures for the BRE test suite.

Rules implemented here:

* Isolation: every test runs with ``BRE_DB_URL`` pointed at a SQLite file under pytest's
  ``tmp_path`` (autouse), so ``db.resolve_db_url()`` never falls back to ``sqlite:///bre.db`` and
  no test can read or write a real database. A test that needs another URL sets it itself with
  ``monkeypatch`` (the two patches are undone in LIFO order).
* Determinism (CLAUDE.md rule 5): the ``rng`` fixture is ``numpy.random.default_rng(TEST_SEED)``
  with ``TEST_SEED = 0``; tests draw all randomness from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TEST_SEED = 0
"""Seed of the ``rng`` fixture. Fixed so every test run is reproducible."""

ENV_DB_URL = "BRE_DB_URL"
"""Environment variable read by ``db.session.resolve_db_url`` (kept literal to avoid importing db)."""


@pytest.fixture(autouse=True)
def bre_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point ``BRE_DB_URL`` at a throwaway SQLite file for the duration of one test.

    Returns the URL (``sqlite:////abs/path/bre_test.db``) so tests that want the same database
    the code under test resolves can request the fixture explicitly.
    """
    url = f"sqlite:///{tmp_path / 'bre_test.db'}"
    monkeypatch.setenv(ENV_DB_URL, url)
    return url


@pytest.fixture()
def rng() -> np.random.Generator:
    """A fresh ``numpy.random.default_rng(TEST_SEED)`` per test."""
    return np.random.default_rng(TEST_SEED)
