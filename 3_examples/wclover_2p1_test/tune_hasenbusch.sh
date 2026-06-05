#!/usr/bin/env bash
# Hasenbusch mass tuning script.
#
# Usage:
#   ./tune_hasenbusch.sh m_H1 m_H2 m_H3 m_H4 [--submit]
#
# Patches ip_hmc_hasenbusch_tune.xml with the given masses, runs 10 HMC
# trajectories on an 8^3x16 lattice, parses FORCES lines, and reports the
# balance metric (max/min PF force ratio).  Repeat until metric < 1.5.
#
# Examples:
#   ./tune_hasenbusch.sh -0.021 0.212 0.459 0.721
#   ./tune_hasenbusch.sh -0.05  0.18  0.42  0.70 --submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/lustre2/nplqcd/vayyar/grid_qcd
BIN=${SCRIPT_DIR}/bin/wclover_hasenbusch_tune_no_eo
TEMPLATE=${SCRIPT_DIR}/inputs/ip_hmc_hasenbusch_tune.xml
WORKDIR=${BASE}/runs/hasenbusch_tune
mkdir -p "${WORKDIR}"

m_H1=${1:?Usage: $0 m_H1 m_H2 m_H3 m_H4 [--submit]}
m_H2=${2:?}
m_H3=${3:?}
m_H4=${4:?}
SUBMIT=${5:-""}

TAG="${m_H1}_${m_H2}_${m_H3}_${m_H4}"
RUNXML="${WORKDIR}/input_${TAG}.xml"
LOGFILE="${WORKDIR}/forces_${TAG}.log"

# ── Patch XML ────────────────────────────────────────────────────────────────
python3 - "${TEMPLATE}" "${RUNXML}" "${m_H1}" "${m_H2}" "${m_H3}" "${m_H4}" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET

src, dst, h1, h2, h3, h4 = sys.argv[1:]
ET.register_namespace('', '')
tree = ET.parse(src)
root = tree.getroot()

def set_val(tag, val):
    el = root.find(f'.//Hasenbusch/{tag}')
    if el is None:
        raise RuntimeError(f'XML tag Hasenbusch/{tag} not found in {src}')
    el.text = val

set_val('m_H1', h1)
set_val('m_H2', h2)
set_val('m_H3', h3)
set_val('m_H4', h4)
tree.write(dst, xml_declaration=True, encoding='unicode')
print(f'Wrote {dst}')
PYEOF

echo "==> Masses: m_H1=${m_H1}  m_H2=${m_H2}  m_H3=${m_H3}  m_H4=${m_H4}"

# ── Run or submit ─────────────────────────────────────────────────────────────
if [[ "${SUBMIT}" == "--submit" ]]; then
    JOBSCRIPT="${WORKDIR}/tune_hasenbusch_${TAG}.sbatch"
    cat > "${JOBSCRIPT}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=hb_tune
#SBATCH --partition=lq2_gpu
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=${LOGFILE}

source ${BASE}/grid-lqcd-workflow/config.sh
${BIN} --ParameterFile ${RUNXML} --grid 8.8.8.16 --mpi 1.1.1.1
SBATCH
    echo "==> Sbatch script written: ${JOBSCRIPT}"
    echo "    Submit with: sbatch ${JOBSCRIPT}"
else
    if [[ ! -x "${BIN}" ]]; then
        echo "ERROR: binary not found at ${BIN}. Run: bash build.sh"
        exit 1
    fi
    source "${BASE}/grid-lqcd-workflow/config.sh"
    echo "==> Running interactively (log: ${LOGFILE})"
    "${BIN}" --ParameterFile "${RUNXML}" \
             --grid 8.8.8.16 --mpi 1.1.1.1 2>&1 | tee "${LOGFILE}"
fi

# ── Parse force output ────────────────────────────────────────────────────────
if [[ ! -f "${LOGFILE}" ]]; then
    echo "==> Log file not yet written (submitted to SLURM). Run parse after job completes:"
    echo "    python3 ${SCRIPT_DIR}/tune_hasenbusch.sh --parse ${LOGFILE}"
    exit 0
fi

python3 - "${LOGFILE}" "${m_H1}" "${m_H2}" "${m_H3}" "${m_H4}" <<'PYEOF'
import sys, re, numpy as np

logfile = sys.argv[1]
masses  = sys.argv[2:]

pf_keys = ['PF0', 'PF1', 'PF2', 'PF3', 'PF4']
pattern = re.compile(r'FORCES\s+traj=\d+((?:\s+\w+=[\d.eE+\-]+)+)')
data = {k: [] for k in pf_keys + ['Strange', 'Gauge']}
ntrajs = 0

for line in open(logfile, errors='replace'):
    m = pattern.search(line)
    if not m:
        continue
    ntrajs += 1
    for kv in m.group(1).split():
        k, v = kv.split('=')
        if k in data:
            try:
                data[k].append(float(v))
            except ValueError:
                pass

if ntrajs == 0:
    print('No FORCES lines found. Job may still be running or binary failed.')
    sys.exit(1)

print(f'\n==> Force norm summary ({ntrajs} trajectories):')
means = {}
for k in pf_keys + ['Strange', 'Gauge']:
    v = data[k]
    means[k] = np.mean(v) if v else float('nan')
    print(f'    {k:8s}: {means[k]:.4e}')

pf_vals = [means[k] for k in pf_keys if not np.isnan(means[k])]
if not pf_vals:
    print('No PF force data found.')
    sys.exit(1)

ratio = max(pf_vals) / min(pf_vals)
print(f'\n    Balance metric (max/min PF): {ratio:.3f}')

if ratio < 1.5:
    print('    STATUS: GOOD — masses are balanced (metric < 1.5)')
elif ratio < 3.0:
    print('    STATUS: OK   — one more iteration recommended')
else:
    print('    STATUS: POOR — significant rebalancing needed')

# Suggest which boundary to move
max_k = max(pf_keys, key=lambda k: means[k])
min_k = min(pf_keys, key=lambda k: means[k])
idx_max = pf_keys.index(max_k)
idx_min = pf_keys.index(min_k)
print(f'\n    Dominant level: {max_k} (force {means[max_k]:.4e})')
print(f'    Weakest level:  {min_k} (force {means[min_k]:.4e})')

if ratio >= 1.5:
    print('\n    Suggested adjustment:')
    if idx_max < idx_min:
        print(f'    Decrease m_H{idx_max+1} (tighten the gap below {max_k})')
    else:
        print(f'    Increase m_H{idx_max} (widen the gap above {min_k})')
    print('    Rule of thumb: move boundary by ~30% of current gap in mass.')
PYEOF
