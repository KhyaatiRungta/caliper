"""Run store: runs are JSON files on disk under ``.caliper/runs/``.

Why files and not a database: a run is an immutable artifact produced by CI.
Files can be diffed, attached to a pull request, committed when you want a
baseline pinned, copied between machines with scp, and read by anything that
can parse JSON. A database would buy indexed queries over a dataset that is
almost always under a few thousand rows, at the cost of a service to run and a
schema to migrate. The moment that tradeoff flips, this module is the only
thing that has to change.
"""

from __future__ import annotations

import json
import os
import random
import string
import time
from datetime import datetime, timezone
from pathlib import Path

from caliper.types import SuiteRun

DEFAULT_ROOT = Path(".caliper")


def new_run_id(suite: str = "", agent: str = "") -> str:
    """Sortable, human-readable, collision-resistant enough for one machine."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    salt = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    slug = "".join(c for c in (agent or suite) if c.isalnum() or c in "-_")[:20]
    return f"{stamp}-{slug}-{salt}" if slug else f"{stamp}-{salt}"


class RunStore:
    """List, load, save and prune :class:`SuiteRun` files."""

    def __init__(self, root: Path | str | None = None):
        env = os.environ.get("CALIPER_HOME")
        self.root = Path(root or env or DEFAULT_ROOT)
        self.runs_dir = self.root / "runs"

    def path_for(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def save(self, run: SuiteRun) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(run.run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)  # atomic: a half-written run file is never readable
        return path

    def load(self, run_id: str) -> SuiteRun:
        path = self.path_for(run_id)
        if not path.exists():
            resolved = self.resolve(run_id)
            if resolved is None:
                raise FileNotFoundError(
                    f"no run {run_id!r} under {self.runs_dir}. "
                    f"Run 'caliper list' to see available run ids."
                )
            path = self.path_for(resolved)
        return SuiteRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, prefix: str) -> str | None:
        """Accept a unique run-id prefix, or ``latest`` / ``latest~N``."""
        ids = self.list_ids()
        if not ids:
            return None
        if prefix in ("latest", "last"):
            return ids[0]
        if prefix.startswith("latest~"):
            try:
                offset = int(prefix.split("~", 1)[1])
            except ValueError:
                return None
            return ids[offset] if offset < len(ids) else None
        matches = [i for i in ids if i.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def list_ids(self) -> list[str]:
        """Run ids, newest first."""
        if not self.runs_dir.exists():
            return []
        paths = sorted(
            self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [p.stem for p in paths]

    def list_runs(self, limit: int | None = None) -> list[SuiteRun]:
        out: list[SuiteRun] = []
        for rid in self.list_ids()[: limit or None]:
            try:
                out.append(self.load(rid))
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # a corrupt run file must not break `caliper list`
        return out

    def prune(self, keep: int = 50, older_than_days: float | None = None) -> list[str]:
        """Delete old runs. Returns the ids removed."""
        removed: list[str] = []
        ids = self.list_ids()
        cutoff = time.time() - (older_than_days * 86400) if older_than_days else None
        for i, rid in enumerate(ids):
            path = self.path_for(rid)
            too_many = i >= keep
            too_old = cutoff is not None and path.stat().st_mtime < cutoff
            if too_many or too_old:
                path.unlink(missing_ok=True)
                removed.append(rid)
        return removed
