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


def _cmd_train(args: argparse.Namespace) -> int:
    from deeptrace.train.config import TrainConfig
    from deeptrace.train.loop import Trainer

    raw = load_yaml(args.config)
    if args.data_root:
        raw.setdefault("data", {})["root"] = args.data_root
    if args.output_root:
        raw["output_root"] = args.output_root
    if args.num_workers is not None:
        raw.setdefault("data", {})["num_workers"] = args.num_workers
    for key in ("batch_size", "device", "time_budget_hours", "epochs"):
        value = getattr(args, key)
        if value is not None:
            raw.setdefault("train", {})[key] = value
    cfg = TrainConfig.from_dict(raw)
    run_dir = Path(cfg.output_root) / cfg.run_name
    setup_logging(log_file=run_dir / "train.log")
    summary = Trainer(cfg, resume=args.resume).fit()
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

    t = sub.add_parser("train", help="fine-tune a model from a YAML config")
    t.add_argument("--config", required=True, type=Path)
    t.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    t.add_argument("--data-root", help="folder containing manifest.csv")
    t.add_argument("--output-root", help="where run folders are written")
    t.add_argument("--batch-size", type=int)
    t.add_argument("--num-workers", type=int, help="DataLoader workers (keep <= 2 locally)")
    t.add_argument("--epochs", type=int)
    t.add_argument("--device", help="cpu, cuda or auto")
    t.add_argument("--time-budget-hours", type=float, help="stop cleanly before this limit")
    t.set_defaults(func=_cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
