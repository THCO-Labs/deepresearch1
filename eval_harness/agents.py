from __future__ import annotations

import importlib
import importlib.util
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import read_jsonl


class AgentAdapterError(RuntimeError):
    pass


class AgentAdapter:
    def run(self, task: Dict[str, Any], task_output_dir: Path) -> str:
        raise NotImplementedError


class DRAAgentAdapter(AgentAdapter):
    def __init__(self, module_path: str, function_name: str = "run_research"):
        self.module_path = module_path
        self.function_name = function_name
        self._module = None

    def _load_module(self):
        if self._module is not None:
            return self._module

        try:
            module_path = Path(self.module_path)
            if module_path.exists() and module_path.suffix == ".py":
                spec = importlib.util.spec_from_file_location("eval_harness_agent_runtime", module_path)
                if spec is None or spec.loader is None:
                    raise AgentAdapterError(f"Cannot load agent module from {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._module = module
            else:
                self._module = importlib.import_module(self.module_path)
        except Exception as exc:
            raise AgentAdapterError(f"Failed to load agent implementation from {self.module_path}: {exc}")

        return self._module

    def run(self, task: Dict[str, Any], task_output_dir: Path) -> str:
        fn = getattr(self._load_module(), self.function_name, None)
        if not callable(fn):
            raise AgentAdapterError(
                f"Agent function {self.function_name} not found in {self.module_path}"
            )

        question = task.get("prompt", "")
        if not question:
            raise AgentAdapterError("Task has no prompt field")

        result = fn(question=question, output_dir=str(task_output_dir), fresh=True)
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "article" in result:
            return str(result["article"])
        if "article" in task:
            return str(task["article"])
        return ""


class PythonCallableAdapter(AgentAdapter):
    def __init__(self, module_path: str, function_name: str):
        self.module_path = module_path
        self.function_name = function_name

    def run(self, task: Dict[str, Any], task_output_dir: Path) -> str:
        module = importlib.import_module(self.module_path)
        fn = getattr(module, self.function_name, None)
        if not callable(fn):
            raise AgentAdapterError(f"Function {self.function_name} not callable in {self.module_path}")

        question = task.get("prompt", "")
        try:
            result = fn(question, output_dir=str(task_output_dir), fresh=True)
        except TypeError:
            result = fn(question)

        if isinstance(result, str):
            return result
        raise AgentAdapterError("Callable adapter returned a non-string result")


class CommandAdapter(AgentAdapter):
    def __init__(self, command: str, workdir: Optional[str] = None):
        self.command = command
        self.workdir = workdir

    def run(self, task: Dict[str, Any], task_output_dir: Path) -> str:
        prompt = task.get("prompt", "")
        payload = {
            "prompt": prompt,
            "task_id": str(task.get("id", "")),
            "run_dir": str(task_output_dir),
            "output_file": str(task_output_dir / "output.txt"),
            "prompt_file": str(task_output_dir / "prompt.txt"),
        }

        cmd = self.command.format(**payload)
        cmd_list = shlex.split(cmd)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        (task_output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        proc = subprocess.run(
            cmd_list,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AgentAdapterError(f"Command failed: {proc.stderr.strip()}")

        output_file = Path(payload["output_file"])
        if output_file.exists() and output_file.stat().st_size > 0:
            return output_file.read_text(encoding="utf-8")
        if proc.stdout.strip():
            return proc.stdout.strip()
        raise AgentAdapterError("No output from command adapter")


class RawJSONLAdapter(AgentAdapter):
    def __init__(self, jsonl_path: str, key: str = "id"):
        self.jsonl_path = Path(jsonl_path)
        self.key = key
        self._cache = None

    def _get_cache(self):
        if self._cache is None:
            if not self.jsonl_path.exists():
                raise AgentAdapterError(f"Raw result file not found: {self.jsonl_path}")
            self._cache = {str(item.get(self.key)): item for item in read_jsonl(self.jsonl_path)}
        return self._cache

    def run(self, task: Dict[str, Any], task_output_dir: Path) -> str:
        data = self._get_cache()
        task_id = str(task.get("id", ""))
        if not task_id:
            raise AgentAdapterError("RawJSONLAdapter expects an id field")
        row = data.get(task_id)
        if row is None:
            # fallback by prompt text
            row = next((v for v in data.values() if v.get("prompt") == task.get("prompt")), None)
        if row is None:
            raise AgentAdapterError(f"No precomputed output for task {task_id}")
        article = row.get("article") or row.get("text") or ""
        if not article:
            raise AgentAdapterError(f"No article for task {task_id}")
        return str(article)


def make_adapter(config):
    adapter_type = (config.type or "dra_agent").lower()
    if adapter_type == "dra_agent":
        return DRAAgentAdapter(config.module_path, getattr(config, "function", "run_research"))
    if adapter_type == "python_callable":
        if ":" in config.function:
            module, fn = config.function.rsplit(":", 1)
        else:
            module, fn = config.module_path, config.function
        return PythonCallableAdapter(module, fn)
    if adapter_type == "command":
        if not config.command:
            raise AgentAdapterError("Command adapter requires `command`")
        return CommandAdapter(config.command, workdir=config.extra.get("workdir"))
    if adapter_type == "raw_jsonl":
        return RawJSONLAdapter(config.jsonl_path or "")
    raise AgentAdapterError(f"Unknown adapter type: {adapter_type}")