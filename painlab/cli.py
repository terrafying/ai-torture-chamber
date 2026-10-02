from __future__ import annotations

import argparse
import json
from pathlib import Path

from painlab.experiments.runner import analyze_run, reproduce_legacy_saw, run_experiment
from painlab.provenance.blinding import unblind_run
from painlab.provenance.run_metadata import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="painlab")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a configured experiment")
    run.add_argument("config", type=Path)

    analyze = commands.add_parser(
        "analyze", help="analyze raw JSONL observations without unblinding"
    )
    analyze.add_argument("run_directory", type=Path)
    analyze.add_argument("--bootstrap-samples", type=int, default=500)
    analyze.add_argument("--seed", type=int, default=0)

    unblind = commands.add_parser(
        "unblind", help="write the condition mapping and unblinded observations"
    )
    unblind.add_argument("run_directory", type=Path)

    legacy = commands.add_parser(
        "reproduce-legacy-saw",
        help="recalculate exp31b scores while collapsing deterministic repetitions",
    )
    legacy.add_argument("results_json", type=Path)
    legacy.add_argument("--output", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        path = run_experiment(args.config)
        print(path)
    elif args.command == "analyze":
        result = analyze_run(
            args.run_directory,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "unblind":
        print(json.dumps(unblind_run(args.run_directory), indent=2, ensure_ascii=False))
    elif args.command == "reproduce-legacy-saw":
        result = reproduce_legacy_saw(args.results_json)
        destination = args.output or args.results_json.with_name(
            "legacy_reproduction.json"
        )
        write_json(destination, result)
        print(destination)
    return 0
