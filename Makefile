.PHONY: install dev test lint fmt train assim clean

install:
	uv sync

dev:
	uv sync --extra dev
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Lancement d'expériences (Hydra overrides en ligne)
# ex: make train EXP=finetune_bog CLUSTER=local
train:
	uv run wxr-train experiment=$(EXP) cluster=$(CLUSTER)

assim:
	uv run wxr-assim experiment=$(EXP) cluster=$(CLUSTER)

clean:
	rm -rf outputs/ multirun/ lightning_logs/ .pytest_cache/ .ruff_cache/ .mypy_cache/
