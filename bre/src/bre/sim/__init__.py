"""Synthetic-data generators of the Phase 2 recovery study (PLAN.md section 5).

Every generator is a module ``bre.sim.<name>`` that exposes

* ``NAME`` — the short generator id (``"gq"``, ``"gc"``, ``"gf"``), and
* ``generate(n_subjects, seed, population=None, design=None, mixed_population=True)
  -> (DataFrame, dict)`` — a DecisionEvent frame in the ``bre.schema`` layout with
  ``is_synthetic == True`` on every row, plus a *truth* dict holding the population parameters and
  every per-subject latent variable that produced the responses.

``bre.sim.common`` holds the pieces the generators share (subject ids and covariates, the design
items, the assembly of an item's two rows in the schema, the parquet + truth writer);
``bre.sim.cli`` runs a grid of generators x N x seeds into ``data/synthetic/`` (``make sim``).

Rule (CLAUDE.md rule 3): everything produced here is synthetic. Every population default is a
labelled synthetic design choice (its ``SOURCE`` string says so), never a number fitted to or
quoted from data, and every table carries ``is_synthetic = True``.
"""

from __future__ import annotations

GENERATOR_NAMES: tuple[str, ...] = ("gq", "gc", "gf")
"""Generator ids of PLAN.md section 5 in the order the recovery grid runs them; each one lives in
``bre.sim.<name>`` when it has been written (``bre.sim.cli`` imports them lazily and reports the
missing ones)."""

__all__ = ["GENERATOR_NAMES"]
