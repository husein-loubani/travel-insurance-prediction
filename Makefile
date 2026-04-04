.PHONY: help install data lint notebook clean

# ── Settings ──────────────────────────────────────────────────────────────────
PYTHON   = venv/bin/python3
PIP      = venv/bin/pip
NOTEBOOK = notebooks/travel_insurance_prediction.ipynb
MODULE   = travel_insurance

## help     : Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/^## //'

## install  : Create venv and install all dependencies
install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Environment ready. Activate with: source venv/bin/activate"

## data     : Verify raw CSV is in place
data:
	@test -f data/raw/TravelInsurancePrediction.csv && \
		echo "data/raw/TravelInsurancePrediction.csv found." || \
		echo "WARNING: data/raw/TravelInsurancePrediction.csv missing."

## lint     : Run ruff linter on the source module
lint:
	$(PYTHON) -m ruff check $(MODULE)/

## notebook : Launch Jupyter
notebook:
	$(PYTHON) -m jupyter lab $(NOTEBOOK)

## clean    : Remove compiled Python files and Jupyter checkpoints
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
