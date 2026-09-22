"""Append-only JSONL event log. One line per event, flushed immediately, so a crashed
run still leaves a complete record up to the crash and runs can be replayed/analysed."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._f = open(self.path, "a", buffering=1) if self.path else None

    def log(self, kind: str, **data) -> None:
        if self._f:
            self._f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **data}, default=str) + "\n")

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def read_trace(path: Path) -> list[dict]:
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # torn last line after a crash
            break
    return events
