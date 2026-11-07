"""Command line entry point: `python -m deeptrace <command>` or `uv run deeptrace <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deeptrace.utils import load_yaml, setup_logging


def _cmd_prepare(args: argparse.Namespace) -> int:
    from deeptrace.data.prepare import PrepareConfig, run_prepare

    raw = load_yaml(args.config)
    if args.raw_root:
        raw["raw_root"] = args.raw_root
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.device:
        raw["device"] = args.device
    cfg = PrepareConfig.from_dict(raw)
    setup_logging(log_file=Path(cfg.output_dir) / "prepare.log")
    report = run_prepare(cfg, limit=args.limit)
    summary = {k: report[k] for k in ("crops", "per_split", "skipped", "dropped")}
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeptrace", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="detect faces, write crops + manifest.csv")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--raw-root", help="override raw_root from the config")
    p.add_argument("--output-dir", help="override output_dir from the config")
    p.add_argument("--device", help="cpu, cuda or auto")
    p.add_argument("--limit", type=int, help="process at most N new source files this run")
    p.set_defaults(func=_cmd_prepare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
