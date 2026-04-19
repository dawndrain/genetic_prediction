"""Resolve paths to shipped resources, downloaded data, and external tools.

Three layers:
  - Shipped resources (genepred/resources/): small reference tables that
    install with the package and never need rebuilding.
  - Data dir ($GENEPRED_DATA, default ./data): large downloads — PGS
    weight files, 1KG VCFs, Beagle reference panels.
  - Tools dir ($GENEPRED_TOOLS, default ./tools): vendored binaries
    (beagle.jar, plink2, bcftools) when not on $PATH / not via conda.
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path
from typing import IO

PKG_DIR = Path(__file__).parent
RESOURCES = PKG_DIR / "resources"


def data_dir() -> Path:
    p = Path(os.environ.get("GENEPRED_DATA", "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def tools_dir() -> Path:
    return Path(os.environ.get("GENEPRED_TOOLS", "tools"))


def resource(name: str) -> Path:
    """Path to a shipped resource. Transparently resolves a .gz sibling."""
    p = RESOURCES / name
    if p.exists():
        return p
    if p.with_suffix(p.suffix + ".gz").exists():
        return p.with_suffix(p.suffix + ".gz")
    raise FileNotFoundError(
        f"Shipped resource '{name}' not found in {RESOURCES}. "
        f"Run `genepred fetch-resources` or rebuild via reference/."
    )


def find_tool(name: str, *candidates: str) -> str:
    """Locate a binary: $PATH first, then tools_dir() with given relative
    candidates. Returns the path as a string for subprocess use."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    td = tools_dir()
    for cand in (name, *candidates):
        p = td / cand
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"'{name}' not found on $PATH or in {td}. "
        f"Install via conda (`environment.yml`) or run tools/fetch_tools.sh."
    )


def open_maybe_gz(path, mode: str = "rt") -> IO[str]:
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)  # type: ignore[return-value]
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)


# Convenience accessors for the data layout under data_dir().
def kg_dir() -> Path:
    return data_dir() / "1kg"


def pgs_weights_dir() -> Path:
    p = data_dir() / "pgs_scoring_files"
    p.mkdir(parents=True, exist_ok=True)
    return p
