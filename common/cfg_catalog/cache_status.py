"""Disk-cache status and tape volume of many HPSS files, in one hsi session.

usage: python3 cache_status.py <list file> <out tsv> [cut]

<list file> has `stream trajectory size path` per line (noxml_paths.py).
Runs `hsi ls -X` on every path (metadata only, nothing is recalled), writes
`stream traj label status tape path` to <out tsv>, and prints counts per
stream, size label (A below cut, B above; default cut 6115370000) and status.
DISK = in the HPSS disk cache, quick to read; TAPE = a read stages the whole
file from tape first.
"""
import collections, os, subprocess, sys, tempfile

lst, out = sys.argv[1], sys.argv[2]
cut = int(sys.argv[3]) if len(sys.argv) > 3 else 6115370000
info = {}
for line in open(lst):
    s, t, size, p = line.split()
    info[p.rsplit("/", 1)[1]] = (s, int(t), "A" if int(size) < cut else "B", p)

with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False) as fh:
    for v in info.values():
        fh.write(f"ls -X {v[3]}\n")
    cmd = fh.name
res = subprocess.run(["hsi", "-q", f"in {cmd}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     universal_newlines=True)
os.unlink(cmd)

rows, cur, level = {}, None, ""
for line in res.stdout.splitlines():
    p = line.split()
    if line.rstrip().endswith(".lime") and len(p) >= 9 and p[-1] in info:
        cur = p[-1]
        rows[cur] = [p[6], "-"]
    elif "(tape)" in line or "(disk)" in line:
        level = "tape" if "(tape)" in line else "disk"
    elif "PV List:" in line and cur and level == "tape" and rows[cur][1] == "-":
        rows[cur][1] = line.split("PV List:")[1].split()[0]

cnt = collections.Counter()
with open(out, "w") as fh:
    for name, (s, t, lab, path) in sorted(info.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        status, tape = rows.get(name, ["?", "-"])
        cnt[(s, lab, status)] += 1
        fh.write(f"{s}\t{t}\t{lab}\t{status}\t{tape}\t{path}\n")
for k in sorted(cnt):
    print(f"{k[0]:8s} {k[1]} {k[2]:5s} {cnt[k]}")
print(f"wrote {out}")
