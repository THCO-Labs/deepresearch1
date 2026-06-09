from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HarnessConfig, load_harness_config, merge_cli_overrides
from .engine import EvaluationHarness


def _parse_args():
    parser = argparse.ArgumentParser(description="Run DeepResearch modular harness")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to harness config")
    common.add_argument("--run-id")
    common.add_argument("--limit", type=int)
    common.add_argument("--only-zh", action="store_true")
    common.add_argument("--only-en", action="store_true")
    common.add_argument("--force", action="store_true")
    common.add_argument("--max-workers", type=int)

    p_run = sub.add_parser("run", parents=[common], help="Generate + scoring")
    p_run.add_argument("--no-race", action="store_true")
    p_run.add_argument("--no-fact", action="store_true")
    p_run.add_argument("--skip-cleaning", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")

    p_gen = sub.add_parser("generate", parents=[common], help="Generate agent outputs only")
    p_gen.add_argument("--dry-run", action="store_true")

    p_race = sub.add_parser("race", parents=[common], help="Run RACE scoring only")

    p_fact = sub.add_parser("fact", parents=[common], help="Run FACT only")

    p_sum = sub.add_parser("summarize", help="Print run summary")
    p_sum.add_argument("run_dir")

    return parser.parse_args()


def _configure(cfg: HarnessConfig, args):
    cfg = merge_cli_overrides(cfg, args)
    if hasattr(args, "skip_cleaning") and getattr(args, "skip_cleaning", False):
        cfg.features.skip_cleaning = True
    if hasattr(args, "dry_run") and getattr(args, "dry_run", False):
        cfg.features.dry_run = True
    return cfg


def _cmd_run(cfg: HarnessConfig, args):
    if getattr(args, "no_race", False):
        cfg.features.run_race = False
    if getattr(args, "no_fact", False):
        cfg.features.run_fact = False

    engine = EvaluationHarness(cfg)
    if args.command == "generate":
        paths = engine.prepare()
        generated = engine.generate(paths)
        return {"generated": len(generated), "run_dir": str(paths["run_dir"]) }
    if args.command == "race":
        paths = engine.prepare()
        # Reuse existing generated artifacts for this run if present.
        generated = engine.generate(paths)
        engine.run_race(paths, len(generated))
        return {"race": "done", "run_dir": str(paths["run_dir"]) }
    if args.command == "fact":
        paths = engine.prepare()
        generated = engine.generate(paths)
        engine.run_fact(paths, len(generated))
        return {"fact": "done", "run_dir": str(paths["run_dir"]) }
    return engine.run()


def main():
    args = _parse_args()
    cfg = load_harness_config(args.config)
    cfg = _configure(cfg, args)

    if args.command == "summarize":
        manifest_file = Path(args.run_dir) / "manifest.json"
        if not manifest_file.exists():
            print(f"No manifest found at {manifest_file}")
            return
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        print(json.dumps(data.get("summary", {}), indent=2))
        return

    summary = _cmd_run(cfg, args)
    print(summary)


if __name__ == "__main__":
    main()