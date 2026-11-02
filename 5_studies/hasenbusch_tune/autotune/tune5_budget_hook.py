#!/usr/bin/env python3
"""Claude Code PreToolUse hook: the tune 5 compute budget, a guard against the agent's MISTAKES.

Loaded ONLY in the tune 5 session (claude --settings ~/.claude/bin/tune5_settings.json), for the
Bash and Monitor tools. A command that touches Slurm, a launcher, the budget, or a child Claude
session must have one of the shapes below; otherwise it is blocked (exit 2) and never runs. Every
allowed submission is charged, at REQUESTED nodes x hours, to a ledger. Other commands pass.

Not a security boundary: an agent determined to evade it, as the same Unix user, can (a script
that submits internally, a variable-built command name). It catches wrong sizes and times, a
forgotten count, too many jobs in flight, the wrong allocation. Python 3.6 compatible.

Shapes (one plain line: printable ASCII, absolute paths, no quotes, shell operators or globs):
  window   bash ~B/tune5_alloc.sh fo <nodes 1-4> <hours 1-4>  # tune5:fo   (or screen ... # tune5:screen)
  release  bash ~B/tune5_release.sh <jobid>                 (tune5_window_* jobs only)
  run      SLURM_JOB_ID=<id> [VAR=value ...] bash <HB>/perlmutter/forces48_puregrid.sh onto a RUNNING
           tune5_window_fo of this user; traj48_puregrid.sh (N_TRAJ <= 3) onto a tune5_window_screen
  batch    env N_TRAJ=<n> RUN=<name> [VAR=value ...] sbatch --time=HH:MM:SS -J tune5_<new name> --export=ALL
           <HB>/perlmutter/traj48_puregrid_batch.sh  # tune5:screen   (no RESUME_FROM / RESUME_TRAJ;
           # tune5:traj20 jobs are always blocked: 20-trajectory runs are outside autotune, user 11-02)
  cancel   bash ~B/tune5_cancel.sh <jobid>                  (tune5_* batch jobs only)
  status   /usr/bin/python3 ~B/tune5_budget_hook.py --status
  (~B = /global/homes/v/vpa/.claude/bin, HB = the hasenbusch_tune study directory)

Budgets, until the USER resets one from their own terminal (python3 <this file> --reset <bucket>):
  fo 96 node-hours; screen 192 node-hours; traj20 0 (the traj20 code below is kept but never
  reached).
Every job <= 8 nodes and <= 14 h (today every shape is <= 4 nodes); batch jobs >= 1 h and
<= 2 h x N_TRAJ + 1 h; at most 2 tune5_* batch jobs queued or running, by job id, each batch job name
used once. Charges are never refunded: a submission that Slurm or the user then refuses stays
charged (errs safe). Ledger: ~/.claude/tune5/ledger.jsonl, one JSON object per line.
"""
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
B = HOME + '/.claude/bin'
TEST_DIR = '/pscratch/sd/v/vpa/tune5_tools/'
# The test-ledger override works only for a copy OUTSIDE ~/.claude/bin: the installed hook always
# uses the real ledger, whatever the environment says.
_test = os.environ.get('TUNE5_LEDGER_TEST', '')
_installed = os.path.realpath(__file__).startswith(os.path.realpath(B) + '/')
LEDGER = _test if (_test.startswith(TEST_DIR) and not _installed) else HOME + '/.claude/tune5/ledger.jsonl'
SELF, ALLOC, RELEASE, CANCEL = (B + '/tune5_budget_hook.py', B + '/tune5_alloc.sh',
                                B + '/tune5_release.sh', B + '/tune5_cancel.sh')
BASE = '/global/cfs/cdirs/m4599/Users/vayyar/grid_qcd'
HB = BASE + '/grid-lqcd-workflow/5_studies/hasenbusch_tune'
BATCH = HB + '/perlmutter/traj48_puregrid_batch.sh'
LAUNCHERS = {HB + '/perlmutter/forces48_puregrid.sh': 'fo', HB + '/perlmutter/traj48_puregrid.sh': 'screen'}
BATCH_NODES = 4                       # the batch script's #SBATCH --nodes; the launcher needs exactly 4
CAP_NODE_HOURS = {'fo': 96.0, 'screen': 192.0, 'traj20': 0.0}
CAP_TRAJ20_CANDIDATES = 0
MAX_NODES = 8
MAX_JOB_HOURS = 14.0
MIN_BATCH_HOURS = 1.0                 # less cannot finish one trajectory (the script also needs > 30 min)
MAX_IN_FLIGHT = 2                     # tune5_* batch jobs pending or running
RECENT_S = 300                        # an approved batch job not yet visible in squeue still counts
MAX_NTRAJ = {'screen': 3, 'traj20': 10}
DEADLINE_S = 20                       # whole hook; Claude Code lets a command through at its 30 s timeout
COST = re.compile(r'^(LADDER_PROFILE|CUSTOM_[A-Z0-9_]+|HASEN_GRID_[A-Z_]+|GRID_MG_[A-Z0-9_]+|DIAG_TRAJL|BIN|DEVICE_MEM_MB)$')
_NAMES = (r'N_TRAJ|RUN|BIN|HMC_SEED_OFFSET|RESUME_FROM|RESUME_TRAJ|DIAG_START_CFG|DIAG_START_TRAJ|'
          r'DEVICE_MEM_MB|LADDER_PROFILE|CUSTOM_[A-Z0-9_]+|HASEN_GRID_[A-Z_]+|GRID_MG_[A-Z0-9_]+|DIAG_TRAJL')
ENV_NAME = re.compile(r'^(%s)$' % _NAMES)                                        # batch jobs
RUN_NAME = re.compile(r'^(SLURM_JOB_ID|FO_PROFILE|FORCES_SAMPLES|TIMEOUT_MIN|%s)$' % _NAMES)   # runs on a window
RESUME = ('RESUME_FROM', 'RESUME_TRAJ')    # chains only through traj20 batch jobs (blocked since 11-02)

# Anything that can allocate, submit, cancel, change jobs, run a launcher, touch the budget, the
# tune5 files, or start another Claude; plus every command that starts with env or VAR= (the
# session's allow rules approve such commands, so the hook must see them all). Tested after
# removing quotes, backslashes and expansion characters, which catches s''batch, s\batch,
# sb[a]tch and sb$@atch; a wildcard standing in for a letter (sb?tch) is not caught.
TRIGGER = re.compile(r'sbat|sallo|srun|scron|strig|scontrol|scancel|alloc_g|release_alloc|tune5_alloc|'
                     r'tune5_release|tune5_cancel|tune5_budget|tune5_settings|tune5/ledger|\.claude/tune5|'
                     r'\.claude/bin|puregrid|SLURM_JOB_ID|(^|[\s/(])claude(\s|$)|^\s*(env\s|[A-Za-z_][A-Za-z0-9_]*=)')
NOISE = re.compile(r'[\'"\\\[\]?*$@{}`]')
TAG = re.compile(r' +# *tune5:(fo|screen|traj20)(?::([A-Za-z0-9._-]{1,40}))? *$')
PLAIN = re.compile(r'^[A-Za-z0-9 ._,/:=+-]*$')


def block(msg):
    sys.stderr.write('tune5 budget hook: %s\n' % msg)
    sys.exit(2)


def hours_of(t):
    """sbatch --time as [D-]HH:MM:SS, ASCII digits, rounded UP to whole minutes (Slurm's unit)."""
    m = re.match(r'^(?:([0-9]{1,2})-)?([0-9]{1,2}):([0-9]{2}):([0-9]{2})$', t)
    if not m or int(m.group(3)) > 59 or int(m.group(4)) > 59:
        return None
    minutes = int(m.group(1) or 0) * 1440 + int(m.group(2)) * 60 + int(m.group(3)) + (1 if int(m.group(4)) else 0)
    return minutes / 60.0


def under(path, roots):
    """True if path is inside one of roots (no '..' components)."""
    return '..' not in path.split('/') and any(path.startswith(r) for r in roots)


def assignments(tok, allowed):
    """Leading VAR=value tokens -> (dict, index of the first other token)."""
    env, i = {}, 0
    while i < len(tok) and '=' in tok[i] and not tok[i].startswith('-'):
        k, v = tok[i].split('=', 1)
        if not allowed.match(k):
            block('assignment %s is not allowed here' % k)
        if k in env:
            block('%s given twice' % k)
        env[k] = v
        i += 1
    return env, i


def check_paths(env):
    if 'BIN' in env and not under(env['BIN'], [HB + '/bin/']):
        block('BIN must be a binary in %s/bin/' % HB)
    if 'RESUME_FROM' in env and not under(env['RESUME_FROM'], [BASE + '/runs/']):
        block('RESUME_FROM must be a run directory under %s/runs/' % BASE)
    if 'DIAG_START_CFG' in env and not under(env['DIAG_START_CFG'], ['/pscratch/sd/v/vpa/cfgs48/', BASE + '/data/']):
        block('DIAG_START_CFG must be in /pscratch/sd/v/vpa/cfgs48/ or %s/data/' % BASE)


def squeue(args):
    """Lines of `squeue -h <args>` for this user, or None if it fails."""
    try:
        p = subprocess.Popen(['/usr/bin/squeue', '-h', '--me'] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True)
        out = p.communicate(timeout=8)[0]
    except Exception:
        return None
    return out.splitlines() if p.returncode == 0 else None


def run_on_window(tok):
    """A launcher run onto one of this user's RUNNING tune5_window allocations (already charged)."""
    env, i = assignments(tok, RUN_NAME)
    if tok[i:] != ['bash', tok[i + 1] if i + 1 < len(tok) else ''] or tok[i + 1] not in LAUNCHERS:
        block('a run is: SLURM_JOB_ID=<id> [VAR=value ...] bash <forces48_puregrid.sh | traj48_puregrid.sh>')
    if not re.match(r'^[0-9]+$', env.get('SLURM_JOB_ID', '')):
        block('SLURM_JOB_ID=<the tune5_window job id> is required')
    kind = LAUNCHERS[tok[i + 1]]
    if kind == 'screen' and not re.match(r'^[1-3]$', env.get('N_TRAJ', '1')):
        block('a trajectory screen on a window runs N_TRAJ 1-3')
    if any(k in env for k in RESUME):
        block('RESUME_FROM / RESUME_TRAJ are not allowed: chains are outside autotune')
    check_paths(env)
    lines = squeue(['-j', env['SLURM_JOB_ID'], '-o', '%j %T'])
    if lines is None:
        block('cannot read squeue for job %s; try again' % env['SLURM_JOB_ID'])
    if lines != ['tune5_window_%s RUNNING' % kind]:
        block('job %s is not a RUNNING tune5_window_%s of yours (%s): a %s run goes only onto a window opened '
              'with tune5_alloc.sh %s, by the id it returned' % (env['SLURM_JOB_ID'], kind, ' / '.join(lines) or 'not found',
                                                                  kind, kind))


def parse_alloc(tok, bucket):
    if len(tok) != 5 or tok[:2] != ['bash', ALLOC]:
        block('a window is exactly: bash %s <fo|screen> <nodes 1-4> <hours 1-4>  # tune5:<the same fo|screen>' % ALLOC)
    if bucket not in ('fo', 'screen') or tok[2] != bucket:
        block('a window is for fo or screen, and its kind must match its tag')
    if not (re.match(r'^[1-4]$', tok[3]) and re.match(r'^[1-4]$', tok[4])):
        block('tune5_alloc.sh needs nodes 1-4 and hours 1-4')
    return int(tok[3]), float(tok[4]), {}, None


def parse_sbatch(tok, bucket):
    if bucket not in MAX_NTRAJ:
        block('a batch job is tagged  # tune5:screen')
    if not tok or tok[0] != 'env':
        block('a batch job is: env N_TRAJ=<n> RUN=<name> [VAR=value ...] sbatch --time=HH:MM:SS -J tune5_<name> --export=ALL %s' % BATCH)
    env, i = assignments(tok[1:], ENV_NAME)
    i += 1
    if i >= len(tok) or tok[i] != 'sbatch':
        block('after the env assignments comes sbatch')
    if not re.match(r'^[1-9][0-9]*$', env.get('N_TRAJ', '')) or int(env['N_TRAJ']) > MAX_NTRAJ[bucket]:
        block('N_TRAJ must be given, 1..%d for %s' % (MAX_NTRAJ[bucket], bucket))
    if not re.match(r'^[A-Za-z0-9._-]{1,60}$', env.get('RUN', '')):
        block('RUN=<name> is required (the batch script exits without it)')
    if bucket != 'traj20' and any(k in env for k in RESUME):
        block('RESUME_FROM / RESUME_TRAJ are not allowed: chains are outside autotune')
    check_paths(env)
    opts = tok[i + 1:]
    hours, name, export, script, j = None, None, False, None, 0
    while j < len(opts):
        a, nxt = opts[j], (opts[j + 1] if j + 1 < len(opts) else None)
        if a.startswith('--time='):
            hours = hours_of(a[7:])
        elif a in ('--time', '-t') and nxt is not None:
            hours, j = hours_of(nxt), j + 1
        elif a.startswith('--job-name='):
            name = a[11:]
        elif a == '-J' and nxt is not None:
            name, j = nxt, j + 1
        elif a == '--export=ALL':
            export = True
        elif a in ('--nodes=4', '-N4'):
            pass
        elif a in ('--nodes', '-N') and nxt == '4':
            j += 1
        elif a == BATCH and j == len(opts) - 1:
            script = a
        else:
            block('sbatch argument %s is not allowed (the batch script %s comes last)' % (a, BATCH))
        j += 1
    if script is None:
        block('the batch script must be %s, last' % BATCH)
    if hours is None:
        block('--time must be given as [D-]HH:MM:SS')
    top = min(MAX_JOB_HOURS, 2.0 * int(env['N_TRAJ']) + 1.0)
    if not MIN_BATCH_HOURS <= hours <= top:
        block('--time %.2f h: a batch job with N_TRAJ=%s takes %.0f to %.0f h' % (hours, env['N_TRAJ'], MIN_BATCH_HOURS, top))
    if not name or not re.match(r'^tune5_[A-Za-z0-9._-]{1,50}$', name) or name.startswith('tune5_window'):
        block('-J tune5_<name> is required (not tune5_window*)')
    if not export:
        block('--export=ALL is required (the env assignments reach the job only through it)')
    return BATCH_NODES, hours, env, name


def ledger_rows():
    """All ledger rows, each checked strictly: a malformed row blocks everything (fail closed)."""
    if not os.path.exists(LEDGER):
        return []
    rows = []
    with open(LEDGER) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ok = isinstance(r, dict)
            if ok and 'reset' in r:
                ok = set(r) == {'t', 'reset'} and r['reset'] in CAP_NODE_HOURS
            elif ok:
                nh = r.get('node_hours')
                ts = r.get('ts')
                ok = (r.get('bucket') in CAP_NODE_HOURS and isinstance(nh, (int, float)) and not isinstance(nh, bool)
                      and math.isfinite(nh) and nh >= 0 and isinstance(r.get('t'), str)
                      and isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts)
                      and all(r.get(k) is None or isinstance(r.get(k), str) for k in ('cand', 'key', 'job')))
            if not ok:
                raise ValueError('malformed ledger row: %s. Report this to the user; do not edit the ledger.' % line.strip()[:200])
            rows.append(r)
    return rows


def since_reset(rows, bucket):
    out = []
    for r in rows:
        if r.get('reset') == bucket:
            out = []
        elif r.get('bucket') == bucket:
            out.append(r)
    return out


def append(row):
    if os.path.exists(LEDGER) and os.path.getsize(LEDGER) > 0:
        with open(LEDGER, 'rb') as f:
            f.seek(-1, 2)
            if f.read(1) != b'\n':
                raise ValueError('ledger does not end with a newline. Report this to the user; do not edit it.')
    with open(LEDGER, 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')


def lock():
    """Exclusive lock on the ledger, or block after 10 s."""
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    lk = open(LEDGER + '.lock', 'w')
    deadline = time.time() + 10
    while True:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lk
        except (IOError, OSError):
            if time.time() > deadline:
                block('ledger lock busy for 10 s; try again later')
            time.sleep(0.2)


def cli():
    """--status (also allowed in the session) and --reset <bucket> (the user's, from a terminal)."""
    a = sys.argv[1:]
    if a == ['--status']:
        rows = ledger_rows()
        for b in sorted(CAP_NODE_HOURS):
            rs = since_reset(rows, b)
            tags = sorted(set(r['cand'] for r in rs if r.get('cand')))
            extra = ', %d of %d candidates (%s)' % (len(tags), CAP_TRAJ20_CANDIDATES, ', '.join(tags)) if b == 'traj20' else ''
            print('%-7s %6.1f of %5.0f node-hours%s' % (b, sum(r['node_hours'] for r in rs), CAP_NODE_HOURS[b], extra))
        return True
    if len(a) == 2 and a[0] == '--reset' and a[1] in CAP_NODE_HOURS:
        lk = lock()
        ledger_rows()
        append({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'reset': a[1]})
        lk.close()
        print('tune5 budget: %s reset' % a[1])
        return True
    if a:
        sys.stderr.write('usage: %s [--status | --reset fo|screen|traj20]\n' % sys.argv[0])
        sys.exit(2)
    return False


def main():
    if cli():
        return
    ev = json.load(sys.stdin)
    if not isinstance(ev, dict) or ev.get('tool_name') not in ('Bash', 'Monitor'):
        return
    ti = ev.get('tool_input')
    cmd = ti.get('command') if isinstance(ti, dict) else None
    if cmd is None:
        return
    if not isinstance(cmd, str):
        block('command is not a string')
    if not TRIGGER.search(NOISE.sub('', cmd)):
        return                                    # nothing Slurm-, launcher-, budget- or Claude-related
    if ev['tool_name'] == 'Monitor':
        block('Monitor cannot run Slurm, launcher, budget or claude commands. To watch for a launcher '
              'failure, match "s.un: error" rather than the literal word.')
    if '\n' in cmd or not PLAIN.match(cmd.replace('#', '')):
        block('one plain line only: printable ASCII, absolute paths, no quotes, shell operators, globs or ~. '
              'Search files with the Grep or Read tools, not grep or cat.')
    tok = [t for t in cmd.split(' ') if t]
    if tok in (['/usr/bin/python3', SELF, '--status'], ['/usr/bin/python3', '-I', SELF, '--status'], ['python3', SELF, '--status']):
        return
    if len(tok) == 3 and tok[0] == 'bash' and tok[1] in (CANCEL, RELEASE) and re.match(r'^[0-9]+$', tok[2]):
        return                                    # the wrappers check owner, name and QOS themselves
    m = TAG.search(cmd)
    if not m and ('puregrid.sh' in cmd or 'SLURM_JOB_ID=' in cmd) and ' sbatch ' not in ' %s ' % cmd:
        run_on_window(tok)
        return
    if not m:
        block('not an allowed shape: a tagged window or batch job, a run onto a tune5_window, tune5_release.sh, '
              'tune5_cancel.sh or --status. Budget resets are the user\'s.')
    bucket, cand = m.group(1), m.group(2)
    if bucket == 'traj20':
        block('20-trajectory runs are outside autotune (user, 11-02): put the candidate on the shortlist '
              'and report it to the user')
    if '#' in cmd[:m.start()]:
        block('one budget tag, at the end')
    tok = [t for t in cmd[:m.start()].split(' ') if t]
    if tok and tok[0] == 'bash':
        nodes, hours, env, job = parse_alloc(tok, bucket)
    else:
        nodes, hours, env, job = parse_sbatch(tok, bucket)
    if (bucket == 'traj20') != bool(cand):
        block('traj20 jobs carry a candidate in the tag (# tune5:traj20:<candidate>); other buckets do not')
    if nodes > MAX_NODES or hours > MAX_JOB_HOURS:
        block('%d nodes x %.2f h: at most %d nodes and %.0f h' % (nodes, hours, MAX_NODES, MAX_JOB_HOURS))
    key = None
    if bucket == 'traj20':
        key = hashlib.sha1(json.dumps(sorted((k, v) for k, v in env.items() if COST.match(k))).encode()).hexdigest()[:12]
    charge = nodes * hours
    lk = lock()
    rows = ledger_rows()
    if job is not None:                           # a batch job: count tune5_* batch jobs in flight
        if any(r.get('job') == job for r in rows):
            block('job name %s was used before: every batch job gets a new -J (a lost submission may still '
                  'be queued: check squeue before resubmitting)' % job)
        lines = squeue(['-o', '%i %j %q'])
        if lines is None:
            block('cannot read squeue to count tune5_* jobs in flight; try again')
        queued = [l.split() for l in lines]
        queued = [q for q in queued if len(q) == 3 and q[1].startswith('tune5_') and 'interactive' not in q[2]]
        names = set(q[1] for q in queued)
        now = time.time()
        recent = [r['job'] for r in rows if r.get('job') and r['job'] not in names
                  and 0 <= now - r.get('ts', 0) < RECENT_S]
        n = len(queued) + len(set(recent))
        if n >= MAX_IN_FLIGHT:
            block('%d tune5_* batch jobs queued, running or just submitted (at most %d): wait for one to end' % (n, MAX_IN_FLIGHT))
    rs = since_reset(rows, bucket)
    used = sum(r['node_hours'] for r in rs)
    if used + charge > CAP_NODE_HOURS[bucket] + 1e-9:
        block('%s budget: %.1f used + %.1f requested > %.0f node-hours. Stop and report to the user; '
              'only the user resets a budget.' % (bucket, used, charge, CAP_NODE_HOURS[bucket]))
    if key is not None:
        tag2key = dict((r['cand'], r['key']) for r in rs if r.get('cand'))
        key2tag = dict((r['key'], r['cand']) for r in rs if r.get('key'))
        if tag2key.get(cand, key) != key:
            block('candidate %s was first submitted with other settings: repeat that job\'s settings exactly '
                  '(plus RESUME_FROM), or report to the user' % cand)
        if key2tag.get(key, cand) != cand:
            block('these settings were submitted as candidate %s: use that tag' % key2tag[key])
        if key not in key2tag and len(key2tag) >= CAP_TRAJ20_CANDIDATES:
            block('traj20 budget: %d candidates already (%s). Stop and report to the user.'
                  % (len(key2tag), ', '.join(sorted(set(key2tag.values())))))
    append({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'ts': round(time.time(), 1), 'bucket': bucket, 'cand': cand,
            'key': key, 'job': job, 'nodes': nodes, 'hours': round(hours, 4), 'node_hours': round(charge, 4), 'cmd': cmd})
    lk.close()


def _timeout(signum, frame):
    sys.stderr.write('tune5 budget hook: took over %d s, blocking to be safe\n' % DEADLINE_S)
    os._exit(2)


if __name__ == '__main__':
    # Fail closed: Claude Code treats any exit code other than 2, and a hook killed at its timeout, as
    # non-blocking. So: an overall deadline below that timeout, and every error blocks.
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(DEADLINE_S)
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        block('internal error, blocking to be safe: %r' % (e,))
    sys.exit(0)
