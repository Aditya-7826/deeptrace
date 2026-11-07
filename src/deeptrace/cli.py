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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from deeptrace.eval.evaluate import EvalOptions, run_evaluate

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else Path(args.results) / run_dir.name
    setup_logging(log_file=out / "evaluate.log")
    opts = EvalOptions(
        run_dir=run_dir,
        data_roots=[Path(args.data_root), *[Path(p) for p in args.extra_data_root]],
        out_dir=out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        threads=args.threads,
        max_test=args.max_test or None,
        robustness_max=args.robustness_max or None,
        bootstrap=args.bootstrap,
        band=(args.band[0], args.band[1]),
    )
    report = run_evaluate(opts)
    brief = {
        "in_dataset_auc": {k: v["auc"] for k, v in report["in_dataset"].items()},
        "cross_dataset_auc": {k: v["auc"] for k, v in report["cross_dataset"].items()},
        "temperature": report["calibration"]["temperature"],
        "threshold": report["calibration"]["threshold"],
        "metrics": str(out / "metrics.json"),
    }
    print(json.dumps(brief, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from deeptrace.eval.compare import run_compare

    for key, path in run_compare(Path(args.results), Path(args.groups)).items():
        print(f"{key}: {path}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from deeptrace.explain.run import explain_image, run_explain
    from deeptrace.utils import resolve_device, set_cpu_threads

    setup_logging()
    run_dir = Path(args.run)
    if args.image:
        set_cpu_threads(args.threads)
        metrics = Path(args.results) / run_dir.name / "metrics.json"
        temp = 1.0
        if metrics.exists():
            temp = json.loads(metrics.read_text(encoding="utf-8"))["calibration"]["temperature"]
        out = Path(args.out or "explain.png")
        faces = explain_image(run_dir, Path(args.image), out, resolve_device(args.device), temp)
        print(json.dumps({"faces": faces, "heatmap": str(out)}, indent=2))
        return 0
    if not args.data_root:
        raise SystemExit("--data-root is required unless --image is given")
    gallery = None if args.no_gallery else Path(args.gallery)
    report = run_explain(
        run_dir,
        Path(args.data_root),
        Path(args.results),
        args.n,
        gallery,
        args.device,
        args.threads,
    )
    summary = {"randomization": report["randomization_test"], "energy": report["energy"]}
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

    e = sub.add_parser("evaluate", help="test metrics, calibration, robustness -> metrics.json")
    e.add_argument("--run", required=True, help="run folder with best.pt (artifacts/runs/<run>)")
    e.add_argument("--data-root", required=True, help="training manifest folder (val + test)")
    e.add_argument(
        "--extra-data-root",
        action="append",
        default=[],
        help="more manifests whose test rows are cross-dataset sets (repeatable)",
    )
    e.add_argument("--results", default="results", help="results root (default: results)")
    e.add_argument("--out", help="output folder (default: <results>/<run name>)")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--num-workers", type=int, default=2)
    e.add_argument("--device", default="auto")
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--max-test", type=int, default=0, help="cap per test set; 0 = all")
    e.add_argument("--robustness-max", type=int, default=2000, help="0 = whole test set")
    e.add_argument("--bootstrap", type=int, default=1000)
    e.add_argument("--band", type=float, nargs=2, default=(0.4, 0.6), metavar=("LO", "HI"))
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("compare", help="architecture + ablation tables from metrics.json files")
    c.add_argument("--results", default="results")
    c.add_argument("--groups", default="configs/compare.yaml")
    c.set_defaults(func=_cmd_compare)

    x = sub.add_parser("explain", help="Grad-CAM sanity checks + gallery, or one image")
    x.add_argument("--run", required=True)
    x.add_argument("--data-root", help="manifest folder (sanity checks + gallery)")
    x.add_argument("--results", default="results")
    x.add_argument("--n", type=int, default=200, help="test images for the sanity checks")
    x.add_argument("--gallery", default="docs/gradcam_gallery.png")
    x.add_argument("--no-gallery", action="store_true")
    x.add_argument("--image", help="explain a single image instead")
    x.add_argument("--out", help="output PNG for --image")
    x.add_argument("--device", default="auto")
    x.add_argument("--threads", type=int, default=4)
    x.set_defaults(func=_cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
