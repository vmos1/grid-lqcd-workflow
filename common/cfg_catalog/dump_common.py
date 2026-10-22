"""Shared reader for the catalog dumps written by fetch_dump.sh."""
import json


def load_dump(path):
    """Return the records of a MetaCat query dump.

    The query endpoint separates records with characters that only
    str.splitlines() treats as line breaks, so a plain `for line in file`
    loop finds nothing.
    """
    recs = []
    for line in open(path).read().splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs
