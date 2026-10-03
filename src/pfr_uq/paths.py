"""Repository-relative paths.

All paths are resolved relative to the repository root, which is located by
searching upward from the current working directory (or this file) for
``pyproject.toml``.  No absolute, machine-specific paths are used anywhere.
"""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root (the first ancestor containing pyproject.toml)."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent)
    for base in candidates:
        for parent in (base, *base.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "src" / "pfr_uq").is_dir():
                return parent
    raise FileNotFoundError("Could not locate the repository root (pyproject.toml).")


ROOT = find_repo_root()
RESULTS = ROOT / "results"
DATA = RESULTS / "data"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "manuscript" / "figures"
GENERATED_TEX = ROOT / "manuscript" / "generated"

for _d in (DATA, TABLES, FIGURES, GENERATED_TEX):
    _d.mkdir(parents=True, exist_ok=True)
