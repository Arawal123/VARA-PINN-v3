"""CSV/JSON experiment logging."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import save_json


def make_run_id(benchmark: str, mode: str, seed: int) -> str:
    """Create a readable run id."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{benchmark}_{mode}_seed{seed}_{stamp}"


class CSVLogger:
    """Append dictionaries to a CSV file with lazy header creation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                self.fieldnames = list(csv.DictReader(f).fieldnames or [])

    def log(self, row: dict[str, Any]) -> None:
        row = {k: _scalar(v) for k, v in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            write_header = not self.path.exists() or self.path.stat().st_size == 0
        else:
            extra = [k for k in row if k not in self.fieldnames]
            if extra:
                self._extend_schema(extra)
            write_header = False
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _extend_schema(self, extra: list[str]) -> None:
        """Rewrite prior rows when a later record introduces new columns."""
        old_fieldnames = list(self.fieldnames or [])
        rows: list[dict[str, Any]] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, fieldnames=old_fieldnames))
            if rows and all(rows[0].get(name) == name for name in old_fieldnames):
                rows = rows[1:]
        self.fieldnames = [*old_fieldnames, *extra]
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class JSONListLogger:
    """Collect records and flush them as a JSON list."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []

    def log(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        save_json(self.records, self.path)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
