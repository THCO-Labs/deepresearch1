from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class QueryTask:
    id: int
    prompt: str
    language: str
    topic: str = ""


def load_tasks(path: str | Path) -> List[QueryTask]:
    tasks: List[QueryTask] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(
                QueryTask(
                    id=int(item.get("id")),
                    prompt=str(item.get("prompt", "")),
                    language=str(item.get("language", "")).lower(),
                    topic=str(item.get("topic", "")),
                )
            )
    return tasks


def filter_tasks(
    tasks: List[QueryTask],
    limit: Optional[int] = None,
    only_zh: bool = False,
    only_en: bool = False,
    task_ids: Optional[List[int]] = None,
) -> List[QueryTask]:
    filtered = [t for t in tasks if t.prompt and t.id is not None]
    if task_ids:
        wanted = set(task_ids)
        filtered = [t for t in filtered if t.id in wanted]
    if only_zh and not only_en:
        filtered = [t for t in filtered if t.language == "zh"]
    if only_en and not only_zh:
        filtered = [t for t in filtered if t.language == "en"]
    if limit and limit > 0:
        filtered = filtered[:limit]
    return filtered