"""Run one or all resumable global feature-selection development stages."""

from __future__ import annotations

import argparse
import json

from fs21.global_pipeline import GLOBAL_STAGE_FUNCTIONS, build_context, run_stage


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", *GLOBAL_STAGE_FUNCTIONS),
        default="all",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    context = build_context(
        device=args.device,
        workers=args.workers,
        smoke=args.smoke,
    )
    stages = (
        list(GLOBAL_STAGE_FUNCTIONS)
        if args.stage == "all"
        else [args.stage]
    )
    outputs = [str(run_stage(context, name)) for name in stages]
    print(json.dumps({"status": "complete", "stages": outputs}, indent=2))


if __name__ == "__main__":
    main()

