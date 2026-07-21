"""Shared runtime options for saved experiment runners."""

from __future__ import annotations

import argparse


DEFAULT_DEVICE = "cuda"
DEFAULT_WORKERS = 4


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common, explicit runtime controls used by every runner."""
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=DEFAULT_DEVICE,
        help="XGBoost device. Default: cuda.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Independent fit workers; each XGBoost fit remains n_jobs=1. Default: 4.",
    )


def validate_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    return workers
