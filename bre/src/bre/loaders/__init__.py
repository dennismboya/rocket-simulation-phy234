"""Dataset loaders of the Behavioral Risk Engine (Phase 1, PLAN.md section 7).

One module per catalogued dataset (``data/CATALOG.md`` is the contract; its "Schema mapping" lines
are the spec of each loader). Every module exposes

* ``load_<name>(raw_dir=RAW_DIR, **opts) -> pd.DataFrame`` -- the DecisionEvent frame
  (:data:`bre.schema.COLUMNS` order, canonical dtypes, ``is_synthetic == False``),
* ``load_<name>_with_meta(raw_dir=RAW_DIR, **opts) -> (frame, meta)`` -- the same frame plus a
  dict with counts (rows, subjects, dropped rows and why, parser statistics), and
* ``write_<name>(out_dir=PROCESSED_DIR, raw_dir=RAW_DIR, **opts)`` -- calls
  :func:`bre.schema.write_events`, which validates the frame and writes the parquet file and its
  ``<name>.validation.md`` report.

``python -m bre.data_cli`` runs every writer and generates ``data/processed/README.md`` from the
actual run (:mod:`bre.data_cli`).
"""

from __future__ import annotations

from bre.loaders._common import PROCESSED_DIR, RAW_DIR

__all__ = ["PROCESSED_DIR", "RAW_DIR"]
