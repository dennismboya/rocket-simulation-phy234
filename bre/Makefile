# Makefile — Behavioral Risk Engine (bre/). One target per phase (PLAN.md section 7).
#
# Definition of done from a clean clone:
#     make setup && make sim && make recover && make fit && make eval && make dashboard
#
# The virtualenv defaults to ./.venv (gitignored). Point BRE_VENV at an existing one to reuse it,
# e.g. `BRE_VENV=/home/user/bre-venv make test`. Every target that runs Python first checks that
# the interpreter exists and otherwise prints a one-line hint naming BRE_VENV. The instrument
# targets need Node (`NODE=/opt/node22/bin/node make instrument` in the build container). A phase
# whose code does not exist yet fails with "phase not built yet: see STATUS.md" instead of silently
# succeeding.

SHELL := /bin/bash

VENV ?= $(or $(BRE_VENV),.venv)
PY := $(VENV)/bin/python
SYSTEM_PYTHON ?= python3.11
NODE ?= node
API_HOST ?= 127.0.0.1
BRE_API_PORT ?= 8000
API_PORT ?= $(BRE_API_PORT)
PYTEST_ARGS ?=

.PHONY: help setup check-venv test test-instrument data sim recover fit eval api dashboard retrain instrument report clean
.DEFAULT_GOAL := help

# $(call require,<path>): the file that implements a phase must exist, otherwise fail on one line.
define require
@test -e "$(1)" || { echo "phase not built yet: see STATUS.md (missing $(1))"; exit 1; }
endef

help:
	@echo "BRE make targets (PLAN.md section 7; current state in STATUS.md):"
	@echo "  setup           create $(VENV) if missing, install requirements.txt and this package (editable)"
	@echo "  test            pytest -q"
	@echo "  test-instrument Phase 7 checks, read-only: validate_battery.py and tests/test_balance.js (needs NODE)"
	@echo "  data            Phase 1: load catalogued datasets into data/processed/*.parquet"
	@echo "  sim             Phase 2: synthetic generators G_Q, G_C, G_F (N=200, seeds 0-4) into data/synthetic/"
	@echo "  recover         Phase 2: model-recovery study, confusion matrices, N target"
	@echo "  fit             Phase 3: fit B1-B6 and Q1-Q5"
	@echo "  eval            Phase 4: pre-registered evaluation and verdict"
	@echo "  api             Phase 5: uvicorn api.main:app on $(API_HOST):$(API_PORT)"
	@echo "  dashboard       Phase 6: streamlit run app/Home.py (starts the API in the background unless $(API_HOST):$(API_PORT) is already listening)"
	@echo "  retrain         Phase 6: retrain on consented responses, promote only if held-out NLL does not degrade (RETRAIN_ARGS=...)"
	@echo "  instrument      Phase 7: validate battery.json, regenerate static/battery.data.js, run the balance test"
	@echo "  report          Phase 8: REPORT.md, MODEL_CARD.md"
	@echo "  clean           remove caches and build artifacts (keeps data/, runs/ and the venv)"
	@echo "Python: $(PY)   Node: $(NODE)"

# Fails on one line, naming BRE_VENV, when the interpreter is missing (instead of bash's raw error).
check-venv:
	@test -x "$(PY)" || { echo "python not found at $(PY): run 'make setup' to create $(VENV), or point BRE_VENV at an existing virtualenv (e.g. BRE_VENV=/home/user/bre-venv make $(MAKECMDGOALS))"; exit 2; }

setup:
	@test -x "$(PY)" || { echo "creating virtualenv $(VENV) with $(SYSTEM_PYTHON)"; $(SYSTEM_PYTHON) -m venv "$(VENV)"; }
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test: check-venv
	$(PY) -m pytest -q $(PYTEST_ARGS)

# Read-only Phase 7 checks: the validator without --write-static, then the balance rule in Node.
test-instrument: check-venv
	$(call require,instrument/validate_battery.py)
	$(PY) instrument/validate_battery.py && $(NODE) instrument/tests/test_balance.js

data: check-venv
	$(call require,src/bre/data_cli.py)
	$(PY) -m bre.data_cli

# Phase 2 generators: N=200, seeds 0-4 for every available generator (missing ones are reported).
sim: check-venv
	$(call require,src/bre/sim/cli.py)
	$(PY) -m bre.sim.cli --generators gq,gc,gf --n 200 --seeds 5 --out data/synthetic

# Phase 2 first pass: N=200, seeds 0-2, generators gq/gc/gf, models B1 B2 B4 Q2 Q4 (~30-40 min on 4 cores;
# outputs under reports/recovery/, log under runs/recover/). Options: python -m bre.recover --help.
recover: check-venv
	$(call require,src/bre/recover.py)
	$(PY) -m bre.recover --n 200 --seeds 0,1,2

# Phase 3: every experiment listed in experiments/fit_all.yaml (artifacts under runs/fits/, metrics under reports/fits/).
fit: check-venv
	$(call require,src/bre/fit.py)
	$(PY) -m bre.fit experiments/fit_all.yaml

# Phase 4: every evaluation listed in experiments/eval_all.yaml (metrics + markdown under reports/eval/).
eval: check-venv
	$(call require,src/bre/eval.py)
	$(PY) -m bre.eval experiments/eval_all.yaml

# Phase 5: when the demo database is missing (and BRE_DB_URL does not point elsewhere), seed it
# first (synthetic book + interventions + demo model, about a minute); then regenerate
# api/openapi.json from the app and serve on BRE_API_PORT (default 8000). See api/README.md.
api: check-venv
	$(call require,api/main.py)
	@if [ -z "$$BRE_DB_URL" ] && [ ! -e data/synthetic/demo.db ]; then \
	  echo "seeding the demo database data/synthetic/demo.db (python -m bre.demo --seed-db, about a minute)"; \
	  PYTHONPATH=src:. $(PY) -m bre.demo --seed-db || exit 1; \
	fi
	PYTHONPATH=src:. $(PY) api/export_openapi.py
	PYTHONPATH=src:. $(PY) -m uvicorn api.main:app --host $(API_HOST) --port $(BRE_API_PORT)

# Phase 6: the API is started in the background only when nothing listens on $(API_HOST):$(API_PORT)
# (a service you started with `make api` is reused and left running); one started here logs to
# runs/api.log and is stopped when streamlit exits (or on Ctrl-C). The app reads BRE_API_URL.
# The first start on the default database seeds the demo book and fits the demo model (about a
# minute); the pages wait for the service (BRE_API_WAIT_S, default 90 s). BRE_ADMIN=1 in the
# environment enables the model-activation control on the transparency page (api/phase6.py).
dashboard: check-venv
	$(call require,api/main.py)
	$(call require,app/Home.py)
	@mkdir -p runs
	@if $(PY) -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$(API_HOST)', $(API_PORT))) == 0 else 1)"; then \
	  echo "API already listening on $(API_HOST):$(API_PORT); reusing it"; \
	  BRE_API_URL=http://$(API_HOST):$(API_PORT) $(PY) -m streamlit run app/Home.py; \
	else \
	  echo "starting the API on $(API_HOST):$(API_PORT) in the background (log: runs/api.log)"; \
	  PYTHONPATH=src:. $(PY) -m uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --log-level warning > runs/api.log 2>&1 & echo $$! > runs/.api.pid; \
	  trap 'kill $$(cat runs/.api.pid) 2>/dev/null; rm -f runs/.api.pid' EXIT INT TERM; \
	  BRE_API_URL=http://$(API_HOST):$(API_PORT) BRE_API_WAIT_S=$${BRE_API_WAIT_S:-180} $(PY) -m streamlit run app/Home.py; \
	fi

# Phase 6: pull the consented responses from the database (BRE_DB_URL, default the demo database),
# refit the active model type on 80% of the subjects, evaluate the held-out 20%, promote the refit
# only if its held-out NLL is not worse than the active artifact's on the same rows; artifact under
# runs/models/<version>/, log and result under runs/retrain/<stamp>/, audit_log row. Options via
# RETRAIN_ARGS (e.g. RETRAIN_ARGS="--model Q2 --steps 500"); the dashboard's Retrain button calls
# POST /retrain, the same code path. A running API picks the promoted model up on its next start
# (the button reloads it in place).
retrain: check-venv
	$(call require,src/bre/retrain.py)
	PYTHONPATH=src:. $(PY) -m bre.retrain $(RETRAIN_ARGS)

# Phase 7: check battery.json, rewrite static/battery.data.js from it, run the balance test.
instrument: check-venv
	$(call require,instrument/validate_battery.py)
	$(PY) instrument/validate_battery.py --write-static && $(NODE) instrument/tests/test_balance.js

report: check-venv
	$(call require,src/bre/report.py)
	$(PY) -m bre.report

clean:
	rm -rf .pytest_cache build src/bre.egg-info
	find src db api app tests instrument experiments -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find src db api app tests instrument experiments -name '*.pyc' -delete 2>/dev/null || true
