"""Single reviewer-oriented command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agew_heabert.config import load_config
from agew_heabert.stages import STAGE_SPECS, execute_stage

STAGES = tuple(STAGE_SPECS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agew-heabert",
        description="Reviewer-facing interface for the AGEW-HeaBERT workflow.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage, spec in STAGE_SPECS.items():
        command = subparsers.add_parser(stage, help=spec.description)
        command.add_argument("--config", required=True, help="JSON or YAML stage configuration")
        command.add_argument(
            "--check-only",
            action="store_true",
            help="Validate configuration and external artifacts without executing the stage",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = execute_stage(args.stage, config, check_only=args.check_only)
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
