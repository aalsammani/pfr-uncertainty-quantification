"""Look up candidate references in the Crossref REST API (bibliographic search).

Usage: python tools/crossref_lookup.py "free-text citation" [...]
Prints the top matches (DOI, title, first author, year, container, volume, issue, pages)
so that a human can confirm the correct record.  Nothing is written to the .bib file.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "pfr_uq-reference-audit/1.0 (mailto:none)"


def _get(url, attempts=6):
    """GET with polite pacing and exponential back-off on HTTP 429/5xx."""
    delay = 2.0
    for i in range(attempts):
        time.sleep(1.2)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["message"]
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and i < attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def search(query, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query, "rows": rows})
    return _get(url)["items"]


def fetch(doi):
    return _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))


def describe(it):
    au = it.get("author", [{}])
    first = (au[0].get("family", "") + ", " + au[0].get("given", "")) if au else ""
    year = (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
    return (f"  DOI {it.get('DOI')} | {(it.get('title') or [''])[0][:110]} | {first} et al.({len(au)}) | {year} | "
            f"{(it.get('container-title') or [''])[0][:60]} | v{it.get('volume','')} i{it.get('issue','')} p{it.get('page','')}"
            f" | type {it.get('type')}")


if __name__ == "__main__":
    for q in sys.argv[1:]:
        try:
            if q.startswith("doi:"):
                print("DOI RECORD:", q[4:])
                print(describe(fetch(q[4:])))
            else:
                print("QUERY:", q)
                for it in search(q):
                    print(describe(it))
        except Exception as exc:  # network errors are reported, not hidden
            print("  lookup failed:", exc)
