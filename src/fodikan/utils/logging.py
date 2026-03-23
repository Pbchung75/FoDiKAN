"""Logging helpers for FoDiKAN."""

from __future__ import annotations

import os
from typing import Any, List, Sequence


class TxtLogger:
    def __init__(self, path: str, also_print: bool = True) -> None:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.path = path
        self.also_print = bool(also_print)
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self._fh.write(f"{text}\n")
        self._fh.flush()
        if self.also_print:
            print(text)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return " | ".join(headers)

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def _format_row(row: Sequence[Any]) -> str:
        return "  ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers)))

    out = [_format_row(headers), "  ".join("-" * width for width in widths)]
    out.extend(_format_row(row) for row in rows)
    return "\n".join(out)
