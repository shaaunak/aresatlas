.PHONY: setup data-audit smoke-test train evaluate research test api frontend reproduce clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

setup:
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision numpy pandas fastparquet pillow fastapi uvicorn pydantic requests scipy pytest python-multipart httpx

data-audit:
	PYTHONPATH=. $(PYTHON) -m src.aresatlas.data.audit

smoke-test:
	PYTHONPATH=. $(PYTEST) tests/test_v2_research.py -v

train:
	PYTHONPATH=. $(PYTHON) scripts/train_model.py

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_benchmarks.py

research:
	PYTHONPATH=. $(PYTHON) scripts/run_route_regret_study.py
	PYTHONPATH=. $(PYTHON) scripts/run_robustness_study.py

test:
	PYTHONPATH=. $(PYTEST) tests backend/tests -v

api:
	PYTHONPATH=. $(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload

frontend:
	cd frontend && python3 server.py

reproduce: data-audit evaluate research test
	@echo "[+] Full AresAtlas v2.0 reproduction pipeline completed successfully."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
