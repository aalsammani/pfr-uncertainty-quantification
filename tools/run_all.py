"""Reproduce the complete study from a clean environment.

Steps (each fails loudly on error):
  1. run the unit tests;
  2. execute notebooks 01 -> 04 in order from a fresh kernel (outputs written in place);
  3. compile the clean manuscript (and, if latexdiff-so is available, the marked one).

Usage:
    python tools/run_all.py               # everything (about 25 minutes)
    python tools/run_all.py --skip-latex  # computations and figures only
    python tools/run_all.py --only 01 02  # selected notebooks

Notebook 03 dominates the run time (about 20 minutes on a laptop).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_model_verification", "02_numerical_validation", "03_uncertainty_analysis", "04_publication_figures"]


def run(cmd, cwd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def execute_notebook(name: str) -> None:
    path = ROOT / "notebooks" / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path.relative_to(ROOT)} in {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-latex", action="store_true")
    ap.add_argument("--only", nargs="*", help="notebook prefixes to run, e.g. 01 04")
    args = ap.parse_args()

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], ROOT)
    for name in NOTEBOOKS:
        if args.only and name[:2] not in args.only:
            continue
        execute_notebook(name)
    if not args.skip_latex:
        ms = ROOT / "manuscript"
        if shutil.which("latexmk") is None:
            print("latexmk not found: skipping manuscript compilation")
            return 0
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main_revised_clean.tex"], ms)
        differ = shutil.which("latexdiff-so") or shutil.which("latexdiff")
        if differ:
            marked = ms / "main_revised_marked.tex"
            with marked.open("w", encoding="utf-8") as fh:
                subprocess.run([differ, "--flatten", "--math-markup=whole",
                                "--config=PICTUREENV=(?:picture|DIFnomarkup|tabular|algorithmic)[\\w\\d*@]*",
                                "../original/main.tex", "main_revised_clean.tex"], cwd=ms, check=True, stdout=fh)
            run([sys.executable, str(ROOT / "tools" / "fix_latexdiff.py"), str(marked)], ROOT)
            run(["latexmk", "-pdf", "-interaction=nonstopmode", "main_revised_marked.tex"], ms)
        else:
            print("latexdiff not found: the committed marked manuscript is left unchanged")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
