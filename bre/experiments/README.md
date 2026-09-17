# experiments/

Configuration-driven experiment definitions for the research pipeline: the Phase 2 recovery-study
grid (PLAN.md section 5) and the Phase 3/4 real-data fits and evaluation splits (PLAN.md sections 4
and 6). The directory is empty until Phase 2 starts; STATUS.md says which experiments exist.

## What belongs here

* One folder per experiment, named `<phase>-<short-name>` (for example `p2-recovery-n200`), holding
  a config file (`config.json` or `config.yaml`) and, if the experiment is not a plain call into
  `bre.recover` / `bre.fit` / `bre.eval`, a `run.py` that reads that config and calls the library.
* Configs state every knob that affects a number: generator, N, seeds, models fitted, restarts,
  early-stopping fold, split type, bootstrap draws. A config with a value missing is a bug, not a
  default.
* Configs reference data only by catalogued name (data/CATALOG.md) or by the synthetic file the
  generator writes under data/synthetic/.

## What does not belong here

* Outputs. Every run writes to `runs/<experiment>/<UTC timestamp>-seed<seed>/` (a copy of the
  config, `metrics.json`, the log, figures). `runs/` is gitignored except `.gitkeep`.
* Data. Raw copies live in data/raw, processed tables in data/processed, synthetic tables in
  data/synthetic (CLAUDE.md rule 3: every synthetic table carries `is_synthetic = True` and no
  real-labelled table or screen may contain a synthetic result).
* Library code. Anything reusable goes into src/bre and is imported here, never copied.

## Conventions

* Seeds are explicit and fixed (CLAUDE.md rule 5). The recovery grid uses five seeds per cell.
* A run longer than ten minutes goes to the background with its log under runs/ and a progress line
  in STATUS.md at least every fifteen minutes; anything over two hours wall-clock is a GATE
  (CLAUDE.md rule 4).
* Negative results are results: an experiment's summary reports what did not work with the same
  detail as what did (CLAUDE.md rule 6).
