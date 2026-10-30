#!/usr/bin/env python3
"""Compare a doc's markdown tables against the tables a regen script generated (2026-10-30).

Written after two numbers checks of the tune1 comparison doc found hand-copied cells that had
drifted from the data (a re-rounded peak force, a stale tail time; L199). The regen writes every
doc table in the doc's own format; this script then checks the doc cell by cell, so a table
pasted into a doc cannot silently disagree with the logs.

  SECTION=FILE   compare the tables in FILE with the doc's tables in section SECTION (the doc
                 heading "## SECTION. ..."; "Summary" for "## Summary").

Pairing and matching:
  - a generated table is compared with the doc table in that section with the same header row
    (case and ** ignored); a generated table with no such doc table is listed, not an error;
  - rows are matched by their leading label cells: the fewest leading cells that are unique in
    the generated table (e.g. "Grid | baseline"). A doc label may add a bracket ("rung 1
    (−0.2416/−0.2400)" matches "rung 1");
  - every other cell must be identical, except that the doc may show only the part of a
    generated cell before ", " ("0.0760(6)" for "0.0760(6), 0.0804 (2013)").

Exit status 1 if any doc cell differs from its generated cell or a paired doc row has no generated
row. Generated rows the doc leaves out, and doc tables with no generated counterpart (prose-backed
tables, run lists), are printed as UNCHECKED so the coverage is visible.

Usage:
  check_doc_tables.py <doc.md> SECTION=FILE [SECTION=FILE ...]
"""
import re
import sys


def norm(c):
    return re.sub(r"\s+", " ", c.replace("**", "").strip()).lower()


def tables(lines):
    """[(header cells, [(line number, cells), ...])] for every markdown table in `lines`."""
    out, cur = [], None
    for n, line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if cur is None:
                cur = (cells, [])
                out.append(cur)
            else:
                cur[1].append((n, cells))
        else:
            cur = None
    return out


def sections(path):
    """{section id: [(line number, text)]}; the id is the number before '.' or the heading."""
    out, sid = {}, None
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        m = re.match(r"## (?:(\w+)\.\s|(.+))", line)
        if m:
            sid = m.group(1) or m.group(2).strip()
            out.setdefault(sid, [])
            continue
        if sid is not None:
            out[sid].append((n, line))
    return out


def key_width(rows):
    for k in range(1, 4):
        keys = [tuple(norm(c) for c in cells[:k]) for _, cells in rows]
        if len(set(keys)) == len(keys):
            return k
    return 3


def label_match(doc_cells, gen_cells):
    for d, g in zip(doc_cells, gen_cells):
        d, g = norm(d), norm(g)
        if d != g and not d.startswith(g + " ("):
            return False
    return True


def cell_match(d, g):
    d, g = d.replace("**", "").strip(), g.replace("**", "").strip()
    return d == g or g.startswith(d + ", ")


def main(argv):
    doc, pairs = argv[0], [a.split("=", 1) for a in argv[1:]]
    secs = sections(doc)
    dtabs_of = {sid: tables(lines) for sid, lines in secs.items()}
    bad, unchecked, ok = 0, [], 0
    paired_doc = set()
    for sid, fname in pairs:
        if sid not in secs:
            print("ERROR  section %s not found in %s" % (sid, doc))
            bad += 1
            continue
        dtabs = dtabs_of[sid]
        gtabs = tables(enumerate(open(fname, encoding="utf-8"), 1))
        for ghead, grows in gtabs:
            dt = [i for i, t in enumerate(dtabs)
                  if [norm(c) for c in t[0]] == [norm(c) for c in ghead]]
            if not dt:
                unchecked.append("§%s generated table not in the doc: %s (%s)" % (
                    sid, " | ".join(ghead), fname))
                continue
            dhead, drows = dtabs[dt[0]]
            paired_doc.add((sid, dt[0]))
            k = key_width(grows)
            used = set()
            for _, gcells in grows:
                hit = [(n, c) for n, c in drows if label_match(c[:k], gcells[:k])]
                if not hit:
                    unchecked.append("§%s generated row not in the doc: %s" % (
                        sid, " | ".join(gcells[:k])))
                    continue
                n, dcells = hit[0]
                used.add(n)
                diffs = [(j, d, g) for j, (d, g) in enumerate(zip(dcells[k:], gcells[k:]), k)
                         if not cell_match(d, g)]
                if len(dcells) != len(gcells):
                    diffs.append((-1, "%d cells" % len(dcells), "%d cells" % len(gcells)))
                if diffs:
                    bad += 1
                    for j, d, g in diffs:
                        col = dhead[j] if 0 <= j < len(dhead) else "row length"
                        print("MISMATCH  %s:%d  §%s  %s / %s:  doc %r  generated %r" % (
                            doc, n, sid, " | ".join(gcells[:k]), col, d, g))
                else:
                    ok += 1
            for n, dcells in drows:
                if n not in used:
                    bad += 1
                    print("MISSING   %s:%d  §%s  doc row with no generated row: %s" % (
                        doc, n, sid, " | ".join(dcells[:k])))
    for sid, dtabs in dtabs_of.items():
        for i, (head, rows) in enumerate(dtabs):
            if (sid, i) not in paired_doc:
                unchecked.append("§%s doc table with no generated table (line %d): %s" % (
                    sid, rows[0][0] - 2 if rows else 0, " | ".join(head)))
    print("check_doc_tables: %d rows identical, %d problems" % (ok, bad))
    for u in unchecked:
        print("UNCHECKED  " + u)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
