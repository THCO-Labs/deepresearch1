from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import AgentAdapter, AgentAdapterError, make_adapter
from .config import HarnessConfig
from .datasets import QueryTask, filter_tasks, load_tasks
from .judges import apply_judge_env
from .utils import append_jsonl, read_jsonl, safe_dump_text, write_jsonl


class EvaluationHarness:
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.root = Path(cfg.run.output_root)

    def _run_id(self) -> str:
        if self.cfg.run.run_id:
            return self.cfg.run.run_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}"

    def prepare(self) -> Dict[str, Path]:
        run_id = self._run_id()
        run_dir = self.root / run_id
        bench_ws = run_dir / "bench"
        raw_dir = run_dir / "raw"
        cleaned_dir = run_dir / "cleaned"
        race_dir = run_dir / "race"
        fact_dir = run_dir / "fact"

        for p in [run_dir, raw_dir, cleaned_dir, race_dir, fact_dir, bench_ws]:
            p.mkdir(parents=True, exist_ok=True)

        paths = {
            "run_dir": run_dir,
            "run_id": run_id,
            "bench_ws": bench_ws,
            "raw_dir": raw_dir,
            "cleaned_dir": cleaned_dir,
            "race_dir": race_dir,
            "fact_dir": fact_dir,
            "log_dir": run_dir / "logs",
        }

        run_dir.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "config": json.loads(json.dumps(asdict(self.cfg))),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._copy_benchmark_workspace(paths["bench_ws"])
        return paths

    def _copy_benchmark_workspace(self, target: Path) -> None:
        src = Path(self.cfg.data.source_root)
        if not src.exists():
            raise RuntimeError(f"Benchmark source not found: {src}")
        target.mkdir(parents=True, exist_ok=True)

        # Copy scripts
        for item in [
            "deepresearch_bench_race.py",
            "requirements.txt",
            "README.md",
            "LICENSE",
        ]:
            path = src / item
            if path.exists() and path.is_file():
                shutil.copy2(path, target / item)

        # Copy required directories
        for item in ["utils", "prompt", "data"]:
            source_dir = src / item
            if not source_dir.exists():
                continue
            target_dir = target / item
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

        # Ensure generated output directory exists for RACE
        (target / "data/test_data/raw_data").mkdir(parents=True, exist_ok=True)

    def _load_adapter(self) -> AgentAdapter:
        return make_adapter(self.cfg.agent)

    def _load_tasks(self) -> List[QueryTask]:
        source_query = Path(self.cfg.data.source_root) / self.cfg.data.query_file
        tasks = load_tasks(source_query)
        return filter_tasks(
            tasks,
            limit=self.cfg.run.limit,
            only_zh=self.cfg.run.only_zh,
            only_en=self.cfg.run.only_en,
            task_ids=self.cfg.run.task_ids,
        )

    def _write_raw_file(self, paths: Dict[str, Path], rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        run_file = paths["raw_dir"] / f"{self.cfg.agent.name}.jsonl"
        bench_file = paths["bench_ws"] / "data/test_data/raw_data" / f"{self.cfg.agent.name}.jsonl"
        write_jsonl(bench_file, rows)
        write_jsonl(run_file, rows)

    def generate(self, paths: Dict[str, Path]) -> List[Dict[str, Any]]:
        if self.cfg.features.dry_run:
            return []

        adapter = self._load_adapter()
        tasks = self._load_tasks()

        bench_ws = paths["bench_ws"]
        raw_data_file = bench_ws / "data/test_data/raw_data" / f"{self.cfg.agent.name}.jsonl"
        existing_results: Dict[str, Dict[str, Any]] = {}

        if raw_data_file.exists() and not self.cfg.run.force:
            for row in read_jsonl(raw_data_file):
                existing_results[str(row.get("id"))] = row

        generated: List[Dict[str, Any]] = list(existing_results.values())
        completed = {str(r.get("id")) for r in generated}

        for task in tasks:
            task_id = str(task.id)
            if task_id in completed:
                continue

            task_out = paths["run_dir"] / "tasks" / task_id
            task_out.mkdir(parents=True, exist_ok=True)

            try:
                article = adapter.run(
                    {
                        "id": task.id,
                        "prompt": task.prompt,
                        "language": task.language,
                        "topic": task.topic,
                    },
                    task_output_dir=task_out,
                )
            except AgentAdapterError as e:
                article = f"ERROR: {e}"

            row = {
                "id": task.id,
                "prompt": task.prompt,
                "article": article,
                "language": task.language,
                "topic": task.topic,
                "agent": self.cfg.agent.name,
            }
            generated.append(row)

        # Keep only rows for the tasks selected in this run.
        selected_task_ids = {task.id for task in tasks}
        selected_rows = [r for r in generated if int(r.get("id")) in selected_task_ids]
        self._write_raw_file(paths, selected_rows)

        # Use append to keep resume-safe behavior and append only new rows only when needed.
        new_rows = [r for r in selected_rows if str(r.get("id")) not in existing_results]
        if new_rows:
            append_jsonl(raw_data_file, new_rows)
        return selected_rows

    def _run_command(self, cwd: Path, command: List[str], env: Optional[Dict[str, str]] = None) -> None:
        merged_env = dict(os.environ)
        if env:
            merged_env.update({k: str(v) for k, v in env.items()})

        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=merged_env,
            capture_output=True,
            text=True,
            check=False,
        )

        log_file = cwd.parent / "logs" / "pipeline.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"CMD: {' '.join(command)}\n\n"
            f"RC: {proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed. See {log_file}")

    def run_race(self, paths: Dict[str, Path], generated_count: int) -> None:
        if not self.cfg.features.run_race:
            return
        if generated_count == 0:
            return

        bench_ws = paths["bench_ws"]
        race_output = paths["race_dir"] / self.cfg.agent.name

        args: List[str] = [
            sys.executable,
            "-u",
            "deepresearch_bench_race.py",
            self.cfg.agent.name,
            "--raw_data_dir",
            "data/test_data/raw_data",
            "--cleaned_data_dir",
            "data/test_data/cleaned_data",
            "--max_workers",
            str(self.cfg.run.max_workers),
            "--query_file",
            "data/prompt_data/query.jsonl",
            "--output_dir",
            str(race_output),
        ]

        if self.cfg.run.limit:
            args.extend(["--limit", str(self.cfg.run.limit)])
        if self.cfg.run.only_zh:
            args.append("--only_zh")
        if self.cfg.run.only_en:
            args.append("--only_en")
        if self.cfg.run.force:
            args.append("--force")
        if self.cfg.features.skip_cleaning:
            args.append("--skip_cleaning")

        judge_env = apply_judge_env({}, self.cfg.judge_race, self.cfg.judge_fact)
        self._run_command(bench_ws, args, judge_env)

    def run_fact(self, paths: Dict[str, Path], generated_count: int) -> None:
        if not self.cfg.features.run_fact:
            return
        if generated_count == 0:
            return

        bench_ws = paths["bench_ws"]
        fact_output = paths["fact_dir"]
        raw_file = f"data/test_data/raw_data/{self.cfg.agent.name}.jsonl"

        judge_env = apply_judge_env({}, self.cfg.judge_race, self.cfg.judge_fact)
        nproc = self.cfg.run.max_workers

        extract_cmd = [
            sys.executable,
            "-u",
            "-m",
            "utils.extract",
            "--raw_data_path",
            raw_file,
            "--output_path",
            str(fact_output / "extracted.jsonl"),
            "--query_data_path",
            "data/prompt_data/query.jsonl",
            "--n_total_process",
            str(nproc),
        ]
        self._run_command(bench_ws, extract_cmd, judge_env)

        dedup_cmd = [
            sys.executable,
            "-u",
            "-m",
            "utils.deduplicate",
            "--raw_data_path",
            str(fact_output / "extracted.jsonl"),
            "--output_path",
            str(fact_output / "deduplicated.jsonl"),
            "--query_data_path",
            "data/prompt_data/query.jsonl",
            "--n_total_process",
            str(nproc),
        ]
        self._run_command(bench_ws, dedup_cmd, judge_env)

        scrape_cmd = [
            sys.executable,
            "-u",
            "-m",
            "utils.scrape",
            "--raw_data_path",
            str(fact_output / "deduplicated.jsonl"),
            "--output_path",
            str(fact_output / "scraped.jsonl"),
            "--n_total_process",
            str(nproc),
        ]
        self._run_command(bench_ws, scrape_cmd, judge_env)

        validate_cmd = [
            sys.executable,
            "-u",
            "-m",
            "utils.validate",
            "--raw_data_path",
            str(fact_output / "scraped.jsonl"),
            "--output_path",
            str(fact_output / "validated.jsonl"),
            "--query_data_path",
            "data/prompt_data/query.jsonl",
            "--n_total_process",
            str(nproc),
        ]
        self._run_command(bench_ws, validate_cmd, judge_env)

        stat_cmd = [
            sys.executable,
            "-u",
            "-m",
            "utils.stat",
            "--input_path",
            str(fact_output / "validated.jsonl"),
            "--output_path",
            str(fact_output / "fact_result.txt"),
        ]
        self._run_command(bench_ws, stat_cmd, judge_env)

    def summarize(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        summary = {
            "run_id": paths["run_id"],
            "agent": self.cfg.agent.name,
            "generated": 0,
            "race": None,
            "fact": None,
            "artifacts": {},
        }

        raw_out = paths["raw_dir"] / f"{self.cfg.agent.name}.jsonl"
        if raw_out.exists():
            items = read_jsonl(raw_out)
            summary["generated"] = len(items)

        race_summary = paths["race_dir"] / self.cfg.agent.name / "race_result.txt"
        if race_summary.exists():
            summary["race"] = race_summary.read_text(encoding="utf-8")

        fact_summary = paths["fact_dir"] / "fact_result.txt"
        if fact_summary.exists():
            summary["fact"] = fact_summary.read_text(encoding="utf-8")

        out_file = paths["run_dir"] / "summary.txt"
        lines = [
            f"run_id={summary['run_id']}",
            f"agent={summary['agent']}",
            f"generated={summary['generated']}",
            "",
        ]
        if summary["race"]:
            lines.append("[RACE]")
            lines.append(summary["race"].strip())
            lines.append("")
        if summary["fact"]:
            lines.append("[FACT]")
            lines.append(summary["fact"].strip())

        safe_dump_text(out_file, "\n".join(lines))

        summary["artifacts"] = {
            "run_dir": str(paths["run_dir"]),
            "summary": str(out_file),
            "race_dir": str(paths["race_dir"]),
            "fact_dir": str(paths["fact_dir"]),
            "raw_data": str(paths["raw_dir"] / f"{self.cfg.agent.name}.jsonl"),
        }

        manifest = json.loads((paths["run_dir"] / "manifest.json").read_text(encoding="utf-8"))
        manifest["summary"] = summary
        manifest["artifacts"] = summary["artifacts"]
        safe_dump_text(paths["run_dir"] / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        return summary

    def run(self) -> Dict[str, Any]:
        paths = self.prepare()

        generated = self.generate(paths)
        self.run_race(paths, len(generated))
        self.run_fact(paths, len(generated))
        return self.summarize(paths)