"""
progress.py — Pipeline progress display.

Prints a clean stage-by-stage progress indicator with timing.
"""

import time


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Pipeline:
    """Track and display pipeline stages with timing and status."""

    def __init__(self, domain: str, stages: list[str]):
        self.domain = domain
        self.stages = stages
        self.results: list[tuple[str, float, bool]] = []
        self.current = 0
        self._start = time.time()

    def stage(self, label: str = None):
        return _Stage(self, label or self.stages[self.current])

    def summary(self):
        total = time.time() - self._start
        print(f"\n\033[1mAutoforge — {self.domain}\033[0m  ({_fmt_time(total)})")
        for label, elapsed, ok in self.results:
            icon = "\033[32m✔\033[0m" if ok else "\033[31m✗\033[0m"
            print(f"  {icon} {label:<44} {_fmt_time(elapsed):>6}")
        remaining = self.stages[len(self.results):]
        for label in remaining:
            print(f"  ○ {label}")
        all_ok = all(ok for _, _, ok in self.results)
        if all_ok and not remaining:
            print(f"\n  Done in {_fmt_time(total)}.")
        elif not all_ok:
            failed = [label for label, _, ok in self.results if not ok]
            print(f"\n  Failed: {', '.join(failed)}")


class _Stage:
    def __init__(self, pipeline: Pipeline, label: str):
        self.pipeline = pipeline
        self.label = label
        self.start = 0.0

    def __enter__(self):
        n = self.pipeline.current + 1
        total = len(self.pipeline.stages)
        print(f"\n\033[33m●\033[0m [{n}/{total}] {self.label}")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        ok = exc_type is None
        self.pipeline.results.append((self.label, elapsed, ok))
        self.pipeline.current += 1
        icon = "\033[32m✔\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"{icon} {self.label}  ({_fmt_time(elapsed)})")
        return False
