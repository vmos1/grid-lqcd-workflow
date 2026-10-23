"""List the LIME records at the start of a config and save its XML records.

usage: python3 lime_head.py <local file | hpss:/path> <out dir> [nbytes]

Reads at most nbytes (default 8 MB). For hpss: it streams `hsi get -` and
stops, so only the header crosses the network; but a file that is only on
tape is still staged whole (6 GB at 48^3x96) before the first byte arrives.
Check `hsi -q ls -X <path>` first: DISK means cached, TAPE means a full stage.

Chroma's input XML is the `scidac-record-xml` record (record 3 in the
cl21_48_96 configs). It is byte-identical to the stream's restart XML for the
same trajectory when one exists.
"""
import os, struct, subprocess, sys, tempfile

src, outdir = sys.argv[1], sys.argv[2]
nbytes = int(sys.argv[3]) if len(sys.argv) > 3 else 8 << 20
if src.startswith("hpss:"):
    with tempfile.TemporaryFile() as errf:
        p = subprocess.Popen(["hsi", "-q", "get - : " + src[5:]], stdout=subprocess.PIPE, stderr=errf)
        buf = b""
        while len(buf) < nbytes:
            chunk = p.stdout.read(min(1 << 20, nbytes - len(buf)))
            if not chunk:
                break
            buf += chunk
        p.kill()
        p.wait()
        if not buf:
            # e.g. "*** hpss_Open: I/O error [-5: HPSS_EIO]" when a tape recall fails
            errf.seek(0)
            for line in errf.read().decode(errors="replace").splitlines():
                if line.strip():
                    print("hsi:", line.strip())
else:
    with open(src, "rb") as fh:
        buf = fh.read(nbytes)

os.makedirs(outdir, exist_ok=True)
pos, i = 0, 0
while pos + 144 <= len(buf):
    magic, ver, flags, length = struct.unpack(">IHHQ", buf[pos:pos + 16])
    if magic != 0x456789AB:
        print(f"bad magic at {pos}")
        break
    rtype = buf[pos + 16:pos + 144].split(b"\0")[0].decode()
    start = pos + 144
    end = start + length
    complete = end <= len(buf)
    print(f"record {i}: type={rtype} bytes={length}{'' if complete else ' (beyond read window)'}")
    if complete and length < (4 << 20) and "binary" not in rtype:
        name = f"{i:02d}_{rtype.replace('/', '_')}.xml"
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(buf[start:end].rstrip(b"\0"))
    if not complete:
        break
    pos = end + (-length % 8)
    i += 1
print(f"read {len(buf)} bytes")
