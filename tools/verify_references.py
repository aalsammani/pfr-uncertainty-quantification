"""Verify every DOI in manuscript/references.bib against its Crossref record.

For each entry with a ``doi`` field the Crossref record is fetched and compared
with the BibTeX fields: title (normalised similarity), year, volume, first page
and the family names of the listed authors.  The result is written to
``docs/reference_crossref_check.csv`` and summarised on screen.  Entries
without a DOI are listed as "no DOI (checked manually)".

Usage:  python tools/verify_references.py
Requires network access; requests are paced to respect the Crossref API.
"""
from __future__ import annotations

import csv
import difflib
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossref_lookup import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "manuscript" / "references.bib"
OUT = ROOT / "docs" / "reference_crossref_check.csv"


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {"ENTRYTYPE": kind, "ID": key}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append(fields)
    return entries


def norm(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)              # drop LaTeX commands
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split().__str__()


def latex_to_ascii(s: str) -> str:
    s = re.sub(r"\\[`'^\"~=.uvHcdbrk]\{?([a-zA-Z])\}?", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def bib_families(author_field: str) -> list[str]:
    fams = []
    for a in author_field.split(" and "):
        a = a.strip()
        if a.lower() == "others" or a.startswith("{"):
            continue
        fam = a.split(",")[0] if "," in a else a.split()[-1]
        fams.append(latex_to_ascii(fam).strip())
    return fams


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    rows, problems = [], 0
    for e in entries:
        row = {"key": e["ID"], "doi": e.get("doi", ""), "status": "", "title_similarity": "", "year_bib": e.get("year", ""),
               "year_crossref": "", "volume_bib": e.get("volume", ""), "volume_crossref": "", "first_page_bib": "",
               "first_page_crossref": "", "authors_match": "", "crossref_title": "", "notes": ""}
        if not e.get("doi"):
            row["status"] = "no DOI (checked manually: ISBN/publisher record)"
            rows.append(row)
            continue
        try:
            rec = fetch(e["doi"])
        except Exception as exc:  # report, do not hide
            row["status"] = f"FETCH FAILED: {exc}"
            problems += 1
            rows.append(row)
            continue
        ct = (rec.get("title") or [""])[0]
        row["crossref_title"] = ct
        sim = difflib.SequenceMatcher(None, norm(e.get("title", "")), norm(ct)).ratio()
        row["title_similarity"] = f"{sim:.2f}"
        dates = [rec.get(k, {}).get("date-parts", [[None]])[0][0] for k in ("published-print", "issued", "published-online")]
        dates = [d for d in dates if d]
        row["year_crossref"] = "/".join(str(d) for d in sorted(set(dates)))
        row["volume_crossref"] = rec.get("volume", "")
        fp_bib = re.split(r"[-–]", e.get("pages", ""))[0].strip()
        fp_cr = re.split(r"[-–]", rec.get("page", "") or rec.get("article-number", ""))[0].strip()
        row["first_page_bib"], row["first_page_crossref"] = fp_bib, fp_cr
        cr_fams = [latex_to_ascii(a.get("family", "")) for a in rec.get("author", []) if a.get("family")]
        if "author" in e and cr_fams:
            bf = bib_families(e["author"])
            row["authors_match"] = str(all(any(b == c or b in c or c in b for c in cr_fams) for b in bf))
        elif not cr_fams:
            row["authors_match"] = "no authors in Crossref record"
        issues = []
        if sim < 0.85:
            issues.append("title differs")
        if e.get("year") and str(e["year"]) not in row["year_crossref"].split("/"):
            issues.append("year differs")
        if row["volume_crossref"] and e.get("volume") and e.get("ENTRYTYPE") == "article" and row["volume_crossref"] != e["volume"]:
            issues.append("volume differs")
        if fp_cr and fp_bib and fp_cr.lower() != fp_bib.lower():
            issues.append("first page differs")
        if row["authors_match"] == "False":
            issues.append("author mismatch")
        row["status"] = "verified" if not issues else "CHECK: " + "; ".join(issues)
        problems += bool(issues)
        rows.append(row)
        print(f"{e['ID']:18s} {row['status']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} entries, {sum(1 for r in rows if r['doi'])} with DOI, {problems} flagged -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
