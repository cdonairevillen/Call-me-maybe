# ========================
# CONFIG
# ========================

PYTHON      := python
VENV        := .venv
VENV_BIN    := $(VENV)/Scripts
PIP         := $(VENV_BIN)/pip
PY          := $(VENV_BIN)/python

REQ         := requirements.txt

# ========================
# INSTALL
# ========================

install:
	@echo "🔧 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)

	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install pydantic numpy

# ========================
# RUN
# ========================

run:
	@echo "🚀 Running project..."
	$(PY) -m src

# ========================
# DEBUG
# ========================

debug:
	$(PY) -m pdb -m src

# ========================
# LINT
# ========================

lint:
	$(PIP) install flake8 mypy
	$(VENV_BIN)/flake8 .
	$(VENV_BIN)/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# ========================
# CLEAN
# ========================

clean:
	@echo "🧹 Cleaning..."
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf $(VENV)