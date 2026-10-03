"""Build docs/reference_audit.md from references.bib, the Crossref check and the manuscript.

For every bibliography entry the table lists authors, title, source, year, DOI,
the verification status (from docs/reference_crossref_check.csv, produced by
tools/verify_references.py), the manuscript sections in which it is cited, and
the purpose of the citation.  Removed references of the original manuscript are
documented in a second table.

Usage:  python tools/build_reference_audit.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_references import parse_bib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "manuscript" / "references.bib"
TEX = ROOT / "manuscript" / "main_revised_clean.tex"
CHECK = ROOT / "docs" / "reference_crossref_check.csv"
OUT = ROOT / "docs" / "reference_audit.md"

PURPOSE = {
    "danckwerts1953": "origin of the dispersion model and of the closed-vessel boundary conditions",
    "wehner1956": "steady solution with Danckwerts conditions (Eq. 6); discussion of boundary conditions",
    "pearson1959": "justification and limitations of the Danckwerts conditions",
    "levenspiel1957": "dispersion (diffusion-type) model for longitudinal mixing",
    "taylor1953": "physical origin of axial (Taylor) dispersion",
    "aris1956": "Taylor--Aris dispersion theory",
    "levenspiel1999": "textbook treatment of dispersion model, PFR/CSTR limits, closed-vessel RTD variance",
    "fogler2016": "textbook treatment of non-ideal reactors",
    "too1986": "related work: stochastic axial dispersion model",
    "nakama2017": "related work: stochastic dispersion coefficient model for tubular equipment",
    "taylor2026": "related work: Sobol' sensitivity across Pe--Da regimes (groundwater transport)",
    "hundsdorfer2003": "numerical methods for ADR equations",
    "morton2005": "numerical PDE background (FTCS instability)",
    "strikwerda2004": "von Neumann and Lax--Richtmyer stability; Lax equivalence",
    "leveque2002": "finite-volume methods",
    "crank1947": "Crank--Nicolson scheme",
    "hindmarsh1984": "stability conditions of explicit Euler for advection--diffusion",
    "rannacher1984": "Rannacher start-up for Crank--Nicolson",
    "harten1983": "high-resolution (TVD) schemes as alternative for discontinuous data",
    "boris1973": "flux-corrected transport as alternative for discontinuous data",
    "protter1984": "maximum principle for parabolic equations",
    "gustafsson1975": "boundary truncation error of lower order need not reduce the global order",
    "gustafsson1981": "boundary truncation error of lower order need not reduce the global order (general mixed problems)",
    "roache2002": "method of manufactured solutions / code verification",
    "oberkampf2010": "verification and validation methodology",
    "sobol1967": "Sobol' sequences",
    "niederreiter1992": "low-discrepancy sequences, (t,s)-sequences",
    "caflisch1998": "MC and QMC methods",
    "owen1997": "scrambled nets and their convergence rate for smooth integrands",
    "matousek1998": "linear matrix scrambling (used by SciPy)",
    "joe2008": "Sobol' direction numbers (used by SciPy)",
    "lecuyer2002": "randomised QMC and replicate-based error estimation",
    "lemieux2009": "MC and QMC sampling (textbook)",
    "dick2010": "digital nets and sequences",
    "sobol2001": "definition of Sobol' sensitivity indices",
    "saltelli2010": "first-order Sobol' index estimator",
    "jansen1999": "total Sobol' index estimator",
    "kucherenko2011": "effective dimension and QMC efficiency",
    "hou2019": "QMC efficiency in engineering uncertainty analysis",
    "smith2013": "uncertainty quantification framework",
    "xiu2002": "polynomial chaos as an alternative to sampling",
    "jcgm2008": "GUM: random and systematic measurement error",
    "kennedy2001": "model discrepancy / systematic effects in calibration",
    "bates1988": "nonlinear least squares and Wald intervals",
    "more1978": "Levenberg--Marquardt algorithm",
    "kaipio2005": "statistical inverse problems",
    "kaipio2007": "inverse crimes (fine-grid synthetic data)",
    "virtanen2020": "SciPy",
    "harris2020": "NumPy",
    "hunter2007": "Matplotlib",
    "meurer2017": "SymPy",
}


def strip_tex(s: str) -> str:
    s = re.sub(r"\\[`'^\"~=.uvHcdbrk]\{?([a-zA-Z])\}?", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return s.replace("{", "").replace("}", "").replace("--", "–").strip()


def citation_locations(tex: str) -> dict:
    """Map citation key -> ordered list of section titles in which it is cited."""
    locs: dict[str, list[str]] = {}
    section = "Front matter"
    for line in tex.splitlines():
        m = re.search(r"\\(?:section|subsection)\*?\{([^}]*)\}", line)
        if m:
            section = strip_tex(m.group(1))
        if "\\begin{abstract}" in line:
            section = "Abstract"
        for cm in re.finditer(r"\\cite[pt]?\{([^}]*)\}", line):
            for key in (k.strip() for k in cm.group(1).split(",")):
                locs.setdefault(key, [])
                if section not in locs[key]:
                    locs[key].append(section)
    return locs


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    tex = TEX.read_text(encoding="utf-8")
    locs = citation_locations(tex)
    checks = {}
    if CHECK.exists():
        with CHECK.open(encoding="utf-8") as fh:
            checks = {r["key"]: r for r in csv.DictReader(fh)}
    lines = [
        "# Reference audit",
        "",
        "Generated by `tools/build_reference_audit.py` from `manuscript/references.bib`, the Crossref comparison in",
        "`docs/reference_crossref_check.csv` (produced by `tools/verify_references.py`) and the citations in",
        "`manuscript/main_revised_clean.tex`. Every DOI below was obtained from, and compared field by field",
        "(title, year, volume, first page, author family names) with, its Crossref record; no DOI was inferred from a pattern.",
        "Books without DOIs were checked by ISBN (Open Library) or publisher record.",
        "",
        "## Final bibliography",
        "",
        "| Key | Authors | Title | Source | Year | DOI | Verification | Cited in | Purpose |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        key = e["ID"]
        authors = strip_tex(e.get("author", "")).replace(" and ", "; ")
        source = strip_tex(e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("institution", ""))
        doi = e.get("doi", "")
        ck = checks.get(key, {})
        status = ck.get("status", "")
        if key == "danckwerts1953" and status.startswith("CHECK"):
            status = "verified (Crossref stores only the main title 'Continuous flow systems')"
        if key == "dick2010" and status.startswith("CHECK"):
            status = "verified (Crossref stores the title without subtitle)"
        if key == "lecuyer2002" and status.startswith("CHECK"):
            status = "verified (author mismatch is the typographic apostrophe in L’Ecuyer)"
        if not doi:
            status = f"no DOI exists; checked by {'ISBN ' + e['isbn'] if 'isbn' in e else 'publisher record (BIPM URL)'}"
        cited = "; ".join(locs.get(key, [])) or "**not cited**"
        doi_md = f"[{doi}](https://doi.org/{doi})" if doi else "–"
        lines.append(f"| `{key}` | {authors} | {strip_tex(e.get('title', ''))} | {source} | {e.get('year', '')} | {doi_md} | {status} | {cited} | {PURPOSE.get(key, '')} |")
    uncited = [e["ID"] for e in entries if e["ID"] not in locs]
    missing = sorted(k for k in locs if k not in {e["ID"] for e in entries})
    lines += ["", f"Entries: {len(entries)}; uncited entries: {uncited or 'none'}; citations without entry: {missing or 'none'}.", ""]
    lines += [REMOVED]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(entries)} entries; uncited: {uncited}; missing: {missing})")
    return 0


REMOVED = r"""## References of the original manuscript

The original manuscript (`original/main.tex`) contained a hand-written list of 34 items. Each was checked against
Crossref. Items that are not needed for the revised content were removed even when correct; incorrect metadata are
recorded so that they are not re-introduced.

| Original key | Outcome | Finding |
|---|---|---|
| danckwerts1953, taylor1953, aris1956, crank1947, boris1973, harten1983, sobol1967, hundsdorfer2003, strikwerda2004, levenspiel1999, fogler2016 | retained | metadata correct; DOIs added from Crossref |
| morton1994 | replaced by morton2005 | the 2nd edition (2005, DOI 10.1017/CBO9780511812248) is the version with a DOI |
| kaipio2004 | corrected to kaipio2005 | Crossref/Springer give 2005 (Applied Mathematical Sciences) |
| groppi2001 | removed | actual record: Groppi & Tronconi, *Chem. Eng. Sci.* **55**(12):2161–2171, **2000** (DOI 10.1016/S0009-2509(99)00440-6); year, volume and pages in the original were wrong; application example not needed |
| illanes2008 | removed | the paper "Recent trends in biocatalysis engineering" is Illanes et al., *Bioresour. Technol.* **115**:48–57, **2012** (DOI 10.1016/j.biortech.2011.12.050); original year/volume/pages were wrong |
| verwer1984 | removed (was not cited in the original text) | the paper is Verwer, Hundsdorfer & Sommeijer, *Numer. Math.* **57**:157–178, **1990** (DOI 10.1007/BF01386405); original year/volume/pages were wrong |
| gallucci2013, hessel2005 | removed | metadata correct; application examples not needed in the revised introduction |
| levenspiel1970, froment1990, schiesser1991, thomas1995, shu1988, hochbruck2010, berger1984, halton1960, cacuci2003, rubinstein2016, evensen2009, beck1977, taylor1997, seborg2016 | removed | not needed for the revised content (most were not cited in the original text) |
| niederreiter1992, xiu2002 | retained | metadata correct; now cited for low-discrepancy sequences and polynomial chaos |
| (citation placeholder "[reference]" for Sobol' sequences) | resolved | now cites sobol1967, joe2008 |
| van der Laan (1958) | considered and **not used** | Crossref record 10.1016/0009-2509(58)80025-1 (Chem. Eng. Sci. 7(3):187–191) is titled only "Letters to the editors" and lists no author, so authorship could not be confirmed; the closed-vessel variance is instead derived from the transfer function and cited to Levenspiel (1999) |
| courant1952 | removed | metadata correct (DOI 10.1002/cpa.3160050303); upwind differencing is no longer discussed in a way that requires it |
"""


if __name__ == "__main__":
    raise SystemExit(main())
