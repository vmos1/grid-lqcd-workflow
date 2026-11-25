#!/usr/bin/env python3
"""Keep only the hunks of a `git diff -U0` (stdin) whose text matches a regex.

    git -C <repo> diff -U0 -- <files> | python3 select_hunks.py '<regex>' > mine.patch
    git -C <repo> apply --cached --unidiff-zero mine.patch

For committing this session's hunks of a file that also holds another session's
uncommitted hunks. Prints kept/dropped counts per file on stderr.
"""
import re
import sys

pattern = re.compile(sys.argv[1])
out = []
header = []
hunk = []
kept = dropped = 0
fname = None
file_has_kept = False


def flush_hunk():
    global hunk, kept, dropped, file_has_kept
    if not hunk:
        return
    if pattern.search("".join(hunk)):
        if not file_has_kept:
            out.extend(header)
            file_has_kept = True
        out.extend(hunk)
        kept += 1
    else:
        dropped += 1
    hunk = []


def flush_file():
    global header, kept, dropped, file_has_kept
    flush_hunk()
    if fname is not None:
        sys.stderr.write("%s: kept %d, dropped %d hunks\n" % (fname, kept, dropped))
    header = []
    kept = dropped = 0
    file_has_kept = False


in_header = False
for line in sys.stdin:
    if line.startswith("diff --git "):
        flush_file()
        fname = line.split(" b/", 1)[1].strip()
        header = [line]
        in_header = True
    elif line.startswith("@@"):
        flush_hunk()
        in_header = False
        hunk = [line]
    elif in_header:
        header.append(line)
    else:
        hunk.append(line)
flush_file()
sys.stdout.write("".join(out))
