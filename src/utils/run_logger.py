"""Timing + JSONL experiment logging.

Every pipeline run is appended as ONE JSON line, so a crashed experiment never
loses finished runs and the file can be loaded straight into pandas:

    import pandas as pd
    df = pd.read_json("results/runs.jsonl", lines=True)
"""

import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Timer:
    """Accumulates named wall-clock spans (milliseconds).

        t = Timer()
        with t.span("retrieval"):
            ...
        t.ms("retrieval")
    """

    def __init__(self):
        self._start = time.perf_counter()
        self._spans = {}

    @contextmanager
    def span(self, name):
        begin = time.perf_counter()
        try:
            yield
        finally:
            self._spans[name] = self._spans.get(name, 0.0) + (
                time.perf_counter() - begin
            )

    def ms(self, name):
        return round(self._spans.get(name, 0.0) * 1000, 1)

    def total_ms(self):
        return round((time.perf_counter() - self._start) * 1000, 1)


class RunLogger:
    """Append-only JSONL logger with resume support."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record):
        record = {
            "run_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **record,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record

    def read(self):
        return read_jsonl(self.path)

    def completed(self, key_fields=("qid", "mode", "model")):
        """Keys of runs that finished WITHOUT an error (used to resume)."""
        done = set()
        for rec in self.read():
            if rec.get("error"):
                continue
            done.add(tuple(rec.get(k) for k in key_fields))
        return done


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
