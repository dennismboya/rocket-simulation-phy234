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
API_PORT ?= 8000
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
	@echo "  sim             Phase 2: synthetic generators G_Q, G_C, G_F into data/synthetic/"
	@echo "  recover         Phase 2: model-recovery study, confusion matrices, N target"
	@echo "  fit             Phase 3: fit B1-B6 and Q1-Q5"
	@echo "  eval            Phase 4: pre-registered evaluation and verdict"
	@echo "  api             Phase 5: uvicorn api.main:app on $(API_HOST):$(API_PORT)"
	@echo "  dashboard       Phase 6: API in the background + streamlit run app/Home.py"
	@echo "  retrain         Phase 6: retrain on consented intake responses"
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
	$(call require,src/bre/loaders/__main__.py)
	$(PY) -m bre.loaders

sim: check-venv
	$(call require,src/bre/sim/__main__.py)
	$(PY) -m bre.sim

recover: check-venv
	$(call require,src/bre/recover.py)
	$(PY) -m bre.recover

fit: check-venv
	$(call require,src/bre/fit.py)
	$(PY) -m bre.fit

eval: check-venv
	$(call require,src/bre/eval.py)
	$(PY) -m bre.eval

api: check-venv
	$(call require,api/main.py)
	$(PY) -m uvicorn api.main:app --host $(API_HOST) --port $(API_PORT)

# The API is started in the background and stopped when streamlit exits (or on Ctrl-C).
dashboard: check-venv
	$(call require,api/main.py)
	$(call require,app/Home.py)
	@mkdir -p runs
	$(PY) -m uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --log-level warning & echo $$! > runs/.api.pid; \
	trap 'kill $$(cat runs/.api.pid) 2>/dev/null; rm -f runs/.api.pid' EXIT INT TERM; \
	$(PY) -m streamlit run app/Home.py

retrain: check-venv
	$(call require,src/bre/retrain.py)
	$(PY) -m bre.retrain

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
