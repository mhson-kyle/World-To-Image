import json
from pathlib import Path
from typing import Iterable, Dict, Any


def read_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue