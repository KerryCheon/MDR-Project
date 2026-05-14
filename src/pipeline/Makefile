PY=python3
VENV=.venv

.PHONY: help venv install run test build build-pipeline docker-run run-station

help:
	@echo "Makefile targets: venv, install, run, run-station, test, build, build-pipeline, docker-run"

venv:
	$(PY) -m venv $(VENV)

install: venv
	@echo "Activating venv and installing requirements..."
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	@echo "Running main pipeline (all stations)"
	PYTHONPATH=$(PWD)/.. $(PY) main.py

run-station:
ifndef STATION
	$(error STATION is undefined. Use `make run-station STATION=spokane_17_ssw`)
endif
	@echo "Running pipeline for station $(STATION)"
	PYTHONPATH=$(PWD)/.. $(PY) main.py --station "$(STATION)"

test:
	pytest -q

build:
	@echo "Building Docker image from Temporal root (default)"
	docker build -t temporal-pipeline -f ../Dockerfile ..

build-pipeline:
	@echo "Building pipeline-specific Docker image from Temporal root"
	docker build -t temporal-pipeline -f Dockerfile ..

docker-run:
	@echo "Running docker image with data volume mounted"
	docker run --rm --workdir /app/Temporal/Pipeline -v $(PWD)/data:/app/Temporal/Pipeline/data temporal-pipeline
