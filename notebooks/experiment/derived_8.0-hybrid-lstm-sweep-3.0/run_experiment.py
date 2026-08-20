#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from core import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare historical LSTM-family encoders with the fixed derived-8.0 two-regime model."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two candidates, two epochs, and 20-tree regime models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run_experiment(smoke=arguments.smoke)
    print(json.dumps(result, indent=2))
