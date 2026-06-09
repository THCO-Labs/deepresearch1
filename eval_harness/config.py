"""Harness configuration schema and helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    source_root: str = "..\\deep_research_bench"
    query_file: str = "data/prompt_data/query.jsonl"
    criteria_file: str = "data/criteria_data/criteria.jsonl"
    reference_file: str = "data/test_data/cleaned_data/reference.jsonl"


@dataclass
class AgentConfig:
    type: str = "dra_agent"
    name: str = "dra-agent"
    module_path: str = "agent.py"
    function: str = "run_research"
    command: Optional[str] = None
    jsonl_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    provider: str = "openai_compatible"
    model: str = "openai/gpt-5.5"
    base_url: Optional[str] = None
    api_key_env: str = "OPENROUTER_API_KEY"
    fact_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    backend: Optional[str] = None


@dataclass
class RunConfig:
    output_root: str = "eval_runs"
    run_id: Optional[str] = None
    limit: Optional[int] = None
    only_zh: bool = False
    only_en: bool = False
    force: bool = False
    max_workers: int = 5
    task_ids: Optional[List[int]] = None


@dataclass
class FeaturesConfig:
    run_race: bool = True
    run_fact: bool = False
    skip_cleaning: bool = False
    dry_run: bool = False


@dataclass
class HarnessConfig:
    data: DataConfig
    agent: AgentConfig
    judge_race: JudgeConfig
    judge_fact: JudgeConfig
    run: RunConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


def _as_dict(raw: Dict[str, Any], *path: str) -> Dict[str, Any]:
    obj = raw
    for p in path:
        if not isinstance(obj, dict):
            return {}
        obj = obj.get(p, {})
    return obj if isinstance(obj, dict) else {}


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        expanded = Path(value).expanduser()
        as_posix = expanded.as_posix()
        # Avoid mutating URL-like values. Using Path(...).as_posix() would turn
        # "https://..." into "https:/..." which breaks downstream judge URLs.
        if "://" in str(value):
            return str(value)
        return as_posix
    if isinstance(value, list):
        return [_expand_env_values(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_values(v) for k, v in value.items()}
    return value


def _dict_to_dataclass(cls, payload: Dict[str, Any]):
    return cls(**payload)


def load_harness_config(path: str) -> HarnessConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = _as_dict(raw, "data")
    agent = _as_dict(raw, "agent")
    judge = _as_dict(raw, "judge")
    run = _as_dict(raw, "run")
    features = _as_dict(raw, "features")

    data_cfg = DataConfig(**_expand_env_values(data))
    agent_cfg = AgentConfig(**_expand_env_values(agent))

    judge_race_raw = _expand_env_values(judge.get("race", {}))
    judge_fact_raw = _expand_env_values(judge.get("fact", {}))
    judge_race_cfg = _dict_to_dataclass(JudgeConfig, judge_race_raw)
    judge_fact_cfg = _dict_to_dataclass(JudgeConfig, judge_fact_raw)

    run_cfg = RunConfig(**_expand_env_values(run))
    features_cfg = FeaturesConfig(**_expand_env_values(features))

    return HarnessConfig(
        data=data_cfg,
        agent=agent_cfg,
        judge_race=judge_race_cfg,
        judge_fact=judge_fact_cfg,
        run=run_cfg,
        features=features_cfg,
    )


def merge_cli_overrides(cfg: HarnessConfig, cli_args):
    run_cfg = cfg.run
    features = cfg.features

    if getattr(cli_args, "run_id", None):
        run_cfg.run_id = cli_args.run_id
    if getattr(cli_args, "limit", None) is not None:
        run_cfg.limit = cli_args.limit
    if getattr(cli_args, "only_zh", False):
        run_cfg.only_zh = True
        run_cfg.only_en = False
    if getattr(cli_args, "only_en", False):
        run_cfg.only_en = True
        run_cfg.only_zh = False
    if getattr(cli_args, "force", False):
        run_cfg.force = True
    if getattr(cli_args, "max_workers", None):
        run_cfg.max_workers = cli_args.max_workers

    if getattr(cli_args, "no_race", False):
        features.run_race = False
    if getattr(cli_args, "no_fact", False):
        features.run_fact = False
    if getattr(cli_args, "skip_cleaning", False):
        features.skip_cleaning = True
    if getattr(cli_args, "dry_run", False):
        features.dry_run = True

    return cfg
