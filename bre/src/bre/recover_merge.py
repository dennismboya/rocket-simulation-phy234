"""Merge several ``bre.recover`` output directories into one recovery report.

``python -m bre.recover_merge --inputs reports/recovery_n50 reports/recovery_n100 reports/recovery_n200 --out reports/recovery``

Why: the grid had to be run in memory-safe pieces (one N per invocation) after container restarts.
Each input ``results.json`` carries the per-fit rows (``fits``) and the per-cell parameter-recovery
records (``recovery``); this script unions them (a later input overrides an earlier one for the same
(generator, N, seed, model) cell), recomputes the selection table, the confusion matrices and the
N_target rule with the functions of :mod:`bre.recover`, copies the scatter figures, and writes
``results.json``, ``summary.md``, ``confusion_N<n>.md`` and ``N_target.json`` to ``--out``. Every
number is recomputed from the input rows; nothing is typed in.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bre.recover import aggregate, confusion_markdown, n_target, summary_markdown
from bre.sim.common import jsonable


def load(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text(encoding="utf-8"))


def merge(inputs: list[Path], out: Path, items_per_subject: int = 170) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    payloads = [load(p) for p in inputs]
    fits: dict[tuple, dict[str, Any]] = {}
    recovery: dict[tuple, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    figures: list[str] = []
    models: list[str] = []
    for src, p in zip(inputs, payloads):
        for r in p["fits"]:
            fits[(r["gen"], int(r["n"]), int(r["seed"]), r["model"])] = r
        for rec in p.get("recovery", []):
            key = (rec["gen"], int(rec["n"]), int(rec["seed"]))
            # union per cell: a later input adds or replaces the model blocks it carries and never
            # erases blocks (e.g. a B1/B4-only completion batch must not drop the Q2/Q4 statistics)
            merged = dict(recovery.get(key, {}))
            merged.update({k: v for k, v in rec.items() if v not in (None, {}, [])})
            recovery[key] = merged
        for m in p["models"]:
            if m not in models:
                models.append(m)
        for f in p.get("failed", []):
            failed.append({**f, "source": str(src)})
        for fig in p.get("figures", []):
            fp = Path(fig)
            if fp.exists():
                dest = out / fp.name
                if fp.resolve() != dest.resolve():
                    shutil.copyfile(fp, dest)
                figures.append(str(dest))
    # a failure is stale when a later input fitted that cell
    failed = [f for f in failed if (f["gen"], int(f["n"]), int(f["seed"]), f["model"]) not in fits]
    results = [fits[k] for k in sorted(fits)]
    ns = sorted({int(r["n"]) for r in results})
    table = aggregate(results, models)
    nt = n_target(table, items_per_subject)
    out.mkdir(parents=True, exist_ok=True)
    for n in ns:
        (out / f"confusion_N{n}.md").write_text(confusion_markdown(table, n, models, results), encoding="utf-8")
    (out / "N_target.json").write_text(json.dumps(jsonable(nt), indent=1) + "\n", encoding="utf-8")
    base = payloads[0]
    payload = {
        "is_synthetic": True,
        "merged_from": [str(p) for p in inputs],
        "started_at": min(p["started_at"] for p in payloads),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": float(sum(p.get("wall_time_s", 0.0) for p in payloads)),
        "workers": [p.get("workers") for p in payloads],
        "generators": sorted({g for p in payloads for g in p["generators"]}),
        "ns": ns,
        "seeds": sorted({int(s) for p in payloads for s in p["seeds"]}),
        "models": models,
        "model_settings": base.get("model_settings"),
        "selection_frac": base.get("selection_frac"),
        "n_boot": base.get("n_boot"),
        "n_jobs": len(results) + len(failed),
        "n_failed": len(failed),
        "failed": failed,
        "log": [p.get("log") for p in payloads],
        "artifact_dir": [p.get("artifact_dir") for p in payloads],
        "table": table.to_dict(orient="records"),
        "n_target": nt,
        "recovery": [recovery[k] for k in sorted(recovery)],
        "fits": results,
        "figures": figures,
    }
    (out / "results.json").write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(jsonable(payload)), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items-per-subject", type=int, default=170)
    a = ap.parse_args(argv)
    payload = merge([Path(p) for p in a.inputs], Path(a.out), a.items_per_subject)
    print(f"[recover_merge] {len(payload['fits'])} fits from {len(a.inputs)} input(s); N_target={payload['n_target']['N_target']}; outputs in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
