# Python UV makefile

PY := python3
UV := uv

USER_NAME := $(shell whoami)

BASE := /goinfre/$(USER_NAME)/call_me_maybe_cache
ARGS ?=

VENV := .venv
CACHE := $(BASE)/uv_cache
TMP := $(BASE)/tmp
HF := $(BASE)/hf_cache

ENV := \
	HF_HOME=$(HF) \
	TRANSFORMERS_CACHE=$(HF) \
	HF_DATASETS_CACHE=$(HF) \
	UV_CACHE_DIR=$(CACHE) \
	TMPDIR=$(TMP)

install:
	mkdir -p $(BASE) $(CACHE) $(TMP) $(HF)

	$(ENV) $(UV) venv $(VENV)
	$(ENV) $(UV) sync

update:
	$(ENV) $(UV) sync --reinstall

run:
	$(ENV) $(UV) run --project . $(PY) -m src $(ARGS)

test:
	$(ENV) $(UV) run --project . $(PY) -m src \
	--functions_definition data/input/break_functions.json \
	--input data/input/break_inputs.json \
	--output data/output/result.json

debug:
	$(ENV) $(UV) run --project . $(PY) -m pdb -m src

lint:
	$(ENV) $(UV) run --project . flake8 src
	$(ENV) $(UV) run --project . mypy src

lint-strict:
	$(ENV) $(UV) run --project . flake8 src
	$(ENV) $(UV) run --project . mypy src --strict


clean:
	rm -rf $(BASE)
	rm -rf $(VENV)
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

re: clean install

# If you want to use "uv sync" and "uv run", you will need to set by hand
# the next env variables:

# HF_HOME=/goinfre/cdonaire/call_me_maybe_cache/hf_cache
# TRANSFORMERS_CACHE=/goinfre/cdonaire/call_me_maybe_cache/hf_cache
# HF_DATASETS_CACHE=/goinfre/cdonaire/call_me_maybe_cache/hf_cache
# UV_CACHE_DIR=/goinfre/cdonaire/call_me_maybe_cache/uv_cache
# TMPDIR=/goinfre/cdonaire/call_me_maybe_cache/tmp