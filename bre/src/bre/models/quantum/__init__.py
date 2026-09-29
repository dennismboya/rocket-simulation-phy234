"""Quantum-probability models (PLAN.md section 4): state vectors or density matrices in C^2,
non-commuting projectors for questions, unitaries for contexts, Born-rule probabilities. All of it
runs on a classical CPU; there is no quantum hardware anywhere in this project.

``bre.models.quantum.core`` holds the Hilbert-space primitives shared by Q1-Q5; the concrete
models live in their own modules next to it.
"""

from bre.models.quantum import core  # noqa: F401  (re-export; also enables x64 via bre.models)

__all__ = ["core"]
