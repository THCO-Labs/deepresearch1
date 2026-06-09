from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class JudgeEnv:
    env: Dict[str, str]
    backend_key: str
    model: str


def _detect_backend(judge_cfg) -> str:
    if judge_cfg.backend:
        return judge_cfg.backend
    base = (judge_cfg.base_url or "").lower()
    if "openrouter" in base:
        return "openrouter"
    return "openai"


def build_env_from_judge(judge_cfg) -> Dict[str, str]:
    backend = _detect_backend(judge_cfg)
    key_env = judge_cfg.api_key_env or "OPENROUTER_API_KEY"
    key_value = os.environ.get(key_env)
    if not key_value:
        raise RuntimeError(f"Missing judge key from environment: {key_env}")

    env = {
        "LLM_BACKEND": backend,
        key_env: key_value,
        "RACE_MODEL": getattr(judge_cfg, "model", "openai/gpt-5.5") if backend == "openrouter" else getattr(judge_cfg, "model", "gpt-5.5"),
    }

    base_url = judge_cfg.base_url
    if base_url:
        if backend == "openrouter":
            env["OPENROUTER_BASE_URL"] = base_url
        else:
            env["OPENAI_BASE_URL"] = base_url

    # Keep these names for downstream scripts that still use FACT_MODEL.
    if env["LLM_BACKEND"] == "openrouter":
        env["FACT_MODEL"] = getattr(judge_cfg, "fact_model", "openai/gpt-5.4-mini")
    else:
        env["FACT_MODEL"] = getattr(judge_cfg, "fact_model", "gpt-5.4-mini")

    return env


def apply_judge_env(base_env: Dict[str, str], race_cfg, fact_cfg, override_fact_model: str | None = None) -> Dict[str, str]:
    env = dict(base_env)
    race_env = build_env_from_judge(race_cfg)
    env.update(race_env)
    if override_fact_model:
        env["FACT_MODEL"] = override_fact_model
    else:
        env["FACT_MODEL"] = getattr(fact_cfg, "model", env.get("FACT_MODEL", "gpt-5.4-mini"))
    return env