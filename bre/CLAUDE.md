# CLAUDE.md — Behavioral Risk Engine (BRE) operating rules

> Project root: `bre/` in this repository. The `report/` folder and the top-level README belong to an
> unrelated PHY234 rocket-simulation project and must not be modified. All BRE paths below are
> relative to `bre/`. Resume protocol for every session: read `bre/STATUS.md` and `bre/PLAN.md`,
> then continue from the first unfinished item.

## 0. Who you are, what you are building, how you work

You are a research engineer. Build, in Python, a reproducible pipeline plus a local dashboard that:

1. assembles the best available behavioral data on context-dependent financial risk decisions,
2. trains quantum-probability models of an investor's decision state against matched classical baselines,
3. tests whether the quantum models predict context effects (panic-selling after losses, question-order effects, preference reversals, overreaction to news) better than the classical ones,
4. serves per-investor predictions P(sell | loss size, context sequence) through an API, and
5. delivers an advisor-facing dashboard in which every input is editable, every number traces to a model parameter and a data source, and the intake questionnaire doubles as the data-collection instrument.

Constraints set by the owner:

* Budget for data collection is $0 for now. No paid panels. Free-registration datasets are fine but gated (see rule 4). Everything must be demonstrable end to end on public data plus clearly labeled synthetic data.
* No private brokerage, advisor, or client data exists. Do not assume any will arrive.
* The final product is the dashboard (Phase 6). The research pipeline exists to make the dashboard's numbers defensible.

Definitions you must hold to:

* "Quantum" means quantum probability: state vectors or density matrices in a Hilbert space, non-commuting projectors for questions, unitaries for context, Born-rule probabilities. Everything runs on a classical CPU. There is no quantum hardware and no "quantum data" in this project. Never write "quantum advantage". Write "quantum-probability model" or "Q-model".
* "Context" means anything that changes the state before a decision is measured: a preceding question, the stated cause of a loss (recession vs technical outage), a social cue, a partial recovery, elapsed time.
* "Classical baseline" means a Kolmogorov-probability model given the same information at a comparable parameter count.

Working discipline (non-negotiable):

1. Plan before executing. Write PLAN.md, then work phase by phase. Update STATUS.md at the end of every phase (done / blocked / decisions / open questions / estimated remaining effort) and give me a five-line summary before starting the next phase. Proceed without waiting unless a step is marked GATE.
2. Provenance. Every dataset gets an entry in data/CATALOG.md (source URL, license or terms, access date, size, fields, mapping to the schema, known problems). Never load a dataset you have not catalogued.
3. Real vs simulated. Simulated data lives only under data/synthetic/ and every table carries an `is_synthetic` column. No synthetic result may appear in a table or screen labeled as real. No fabricated or "illustrative" numbers anywhere.
4. GATES. Stop and ask me before (a) spending any money, (b) registering for datasets in my name or accepting data-use terms, (c) any run over 2 hours wall-clock or needing cloud compute, (d) changing scope.
5. Reproducibility. Fixed seeds, pinned requirements.txt, a `make` target per phase, pytest for the math (unitarity, probabilities sum to one, recovery of known parameters) and for the dashboard (see Phase 6 acceptance tests).
6. Honesty. Negative results are results. If the Q-model does not beat the baselines, the report and the dashboard's transparency page say so. Do not cite a paper you have not confirmed exists. Do not claim causality from observational data.
7. Do not skip Phase 2 (model recovery on synthetic data) even though the dashboard is the goal. It is what makes any displayed number defensible.
8. Claude Code specifics. Use plan mode for PLAN.md before writing code. You may run dataset cataloging and independent model fits in parallel subagents, but only the main session edits CATALOG.md, STATUS.md and PLAN.md. Run fits longer than 10 minutes in the background, logging to runs/, and write a progress line to STATUS.md at least every 15 minutes. Assume your context can be compacted at any time: STATUS.md must always contain enough to resume without me.

## Environment notes (recorded by the first session, 2026-09-17)

* Python 3.11, CPU only (4 cores, 15 GB RAM). The virtualenv lives outside the repo; `make setup` recreates it.
* Session egress policy: PyPI, npm and GitHub (git over HTTPS, raw.githubusercontent.com) are reachable. Hugging Face, OSF, Zenodo, PNAS/PMC, arXiv, Yahoo Finance, FRED, CBOE, UAS, Centerdata, FINRA and CDNs are blocked from the container. Anything hosted only there is catalogued with download instructions for the owner, not fetched.
* Model stack: JAX + optax for the Q-models and the optimizer-based baselines, NumPyro for the hierarchical Bayesian baseline. Reason: one autodiff framework for everything, CPU wheels available from PyPI, `jax.scipy.linalg.expm` gives the matrix exponential for unitaries.
