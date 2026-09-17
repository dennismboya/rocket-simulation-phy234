"""Command line of the synthetic generators (``make sim``, PLAN.md section 5).

    python -m bre.sim.cli --generators gq,gc,gf --n 200 --seeds 5 --out data/synthetic

runs every named generator (``bre.sim.<name>``, imported lazily so that a generator that has not
been written yet is reported as missing instead of breaking the others) for every ``N`` and
every seed, writes ``<name>_n<N>_seed<seed>.parquet`` plus ``.truth.json`` through
:func:`bre.sim.common.write_synthetic` (which validates the frame and enforces the
``data/synthetic`` location and ``is_synthetic == True``) and finally rewrites
``<out>/manifest.json`` listing every truth file in the output directory (generator, N, seed,
rows, mixed_population, file names), which the recovery study reads.

Options: ``--n`` takes one integer or a comma list (``100,200,400,800`` for the full grid);
``--seeds`` an integer count (seeds ``0 .. k-1``) or a comma list of seeds; ``--no-mixed`` sets
``mixed_population=False`` (G_Q: every ``gamma_i = 0``; the other generators receive the flag
and may ignore it); ``--dry-run`` lists the jobs without generating. Exit status 0 when at least
one generator ran and every job succeeded, 1 when no generator was available or a job failed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from bre import schema as S
from bre.sim import GENERATOR_NAMES, common

MANIFEST_NAME = "manifest.json"


def _parse_int_list(text: str, what: str) -> list[int]:
    try:
        values = [int(t) for t in str(text).split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{what} must be an integer or a comma list of integers; got {text!r}") from exc
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"{what} must be non-negative; got {text!r}")
    return values


def parse_seeds(text: str) -> list[int]:
    """``"5"`` -> ``[0, 1, 2, 3, 4]``; ``"0,3,7"`` -> ``[0, 3, 7]``."""
    if "," in str(text):
        return _parse_int_list(text, "--seeds")
    k = _parse_int_list(text, "--seeds")[0]
    return list(range(k))


def load_generator(name: str) -> ModuleType | None:
    """Import ``bre.sim.<name>``; None when that module does not exist (any other import error
    propagates). The module must expose ``NAME`` and ``generate``."""
    modname = f"bre.sim.{name}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        if exc.name == modname:
            return None
        raise
    if not hasattr(mod, "generate") or not hasattr(mod, "NAME"):
        raise AttributeError(f"{modname} does not expose NAME and generate()")
    return mod


def file_stem(name: str, n: int, seed: int) -> str:
    return f"{name}_n{int(n)}_seed{int(seed)}"


def write_manifest(out_dir: Path) -> Path:
    """Rewrite ``manifest.json`` from every ``*.truth.json`` in ``out_dir`` (sorted by name)."""
    entries: list[dict[str, Any]] = []
    for t_path in sorted(out_dir.glob("*.truth.json")):
        truth = S.json_loads_strict(t_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "generator": truth.get("generator"),
                "n_subjects": truth.get("n_subjects"),
                "seed": truth.get("seed"),
                "n_rows": truth.get("n_rows"),
                "mixed_population": truth.get("mixed_population"),
                "is_synthetic": True,
                "file": truth.get("file", t_path.name.replace(".truth.json", ".parquet")),
                "truth_file": t_path.name,
            }
        )
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps({"is_synthetic": True, "entries": entries}, indent=1) + "\n", encoding="utf-8")
    return path


def run(
    generators: list[str],
    ns: list[int],
    seeds: list[int],
    out_dir: Path,
    mixed_population: bool = True,
    dry_run: bool = False,
    log=print,
) -> dict[str, Any]:
    """Run the grid; returns ``{"written": [...], "missing": [...], "failed": [...]}``."""
    out_dir = Path(out_dir)
    written: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[dict[str, Any]] = []
    for name in generators:
        mod = load_generator(name)
        if mod is None:
            missing.append(name)
            log(f"[sim] generator {name!r} missing (no module bre.sim.{name}); skipped")
            continue
        for n in ns:
            for seed in seeds:
                stem = file_stem(mod.NAME, n, seed)
                if dry_run:
                    log(f"[sim] would generate {stem} (mixed_population={mixed_population})")
                    continue
                t0 = time.time()
                try:
                    df, truth = mod.generate(n, seed, mixed_population=mixed_population)
                    pq_path, t_path = common.write_synthetic(df, truth, stem, out_dir)
                except Exception as exc:  # noqa: BLE001 - report and continue with the grid
                    failed.append({"generator": name, "n": n, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                    log(f"[sim] {stem} FAILED: {type(exc).__name__}: {exc}")
                    continue
                dt = time.time() - t0
                written.append({"generator": mod.NAME, "n": n, "seed": seed, "rows": int(len(df)), "file": str(pq_path), "seconds": dt})
                log(f"[sim] {stem}: {len(df)} rows -> {pq_path.name} (+ {t_path.name}) in {dt:.1f}s")
    if not dry_run and written:
        m = write_manifest(out_dir)
        log(f"[sim] manifest: {m}")
    return {"written": written, "missing": missing, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bre.sim.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("--generators", default=",".join(GENERATOR_NAMES), help="comma list of generator ids (default: %(default)s)")
    parser.add_argument("--n", default="200", help="subjects per table: an integer or a comma list (default: %(default)s)")
    parser.add_argument("--seeds", default="5", help="seed count (0..k-1) or a comma list of seeds (default: %(default)s)")
    parser.add_argument("--out", default=str(S.SYNTHETIC_DIR), help="output directory, must be a data/synthetic directory (default: %(default)s)")
    parser.add_argument("--no-mixed", action="store_true", help="mixed_population=False (G_Q: gamma_i = 0 for every subject)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs without generating")
    args = parser.parse_args(argv)

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    if not generators:
        parser.error("--generators is empty")
    try:
        ns = _parse_int_list(args.n, "--n")
        seeds = parse_seeds(args.seeds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    out_dir = Path(args.out)
    if not S.is_synthetic_path(out_dir / "x.parquet"):
        parser.error(f"--out must be a data/synthetic directory (CLAUDE.md rule 3); got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run(generators, ns, seeds, out_dir, mixed_population=not args.no_mixed, dry_run=args.dry_run)
    n_ok = len(result["written"])
    print(
        f"[sim] done: {n_ok} table(s) written, {len(result['failed'])} failed, "
        f"missing generators: {result['missing'] or 'none'}"
    )
    available = [g for g in generators if g not in result["missing"]]
    if not available:
        print("[sim] no generator available", file=sys.stderr)
        return 1
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
