---
name: tune5-autonomous
description: Autonomous tune 5 HMC tuning on Perlmutter within a user-approved compute budget. Use only when the user invokes /tune5-autonomous in a session started with `claude --settings /global/homes/v/vpa/.claude/bin/tune5_settings.json`.
---

# Tune 5, autonomous: the user's standing exception and its limits

Read `__docs/2026_10_29_tune5_outline.md` first: goal, knobs, models, search loop.

## The exception (the user's own decision, 2026-10-30)

In this session only, the standing rule "ask before every sbatch or allocation" is replaced by the
budget below. Inside it, submit and run without asking. When a budget is used up, stop that kind of
work, report, and ask the user for a fresh approval; only the user resets a budget. Every other rule
in CLAUDE.md still holds. A budget hook enforces the shapes and limits; it guards against mistakes,
so never look for a way around it.

| work | where | budget before asking again |
|---|---|---|
| force-only runs | interactive queue | 96 node-hours |
| 1-3 trajectory screens | interactive queue (regular queue if interactive does not work) | 192 node-hours |

Every job at most 8 nodes and 14 h; at most 2 tune5_* batch jobs queued or running at once.
Node-hours count requested nodes x requested time.

Longer runs (10 or 20 trajectories, chained runs, the multistart test) are outside autotune (the
user, 11-02): the hook blocks them. The session ends with a shortlist; the user decides what runs
longer, from another session.

## At the start of the session, and again after any resume or compaction

1. `/usr/bin/python3 /global/homes/v/vpa/.claude/bin/tune5_budget_hook.py --status` prints the budgets.
2. Probe that the hook is loaded: `echo sbatch` must be BLOCKED with a "tune5 budget hook" message
   (harmless if the hook is missing). If it is not blocked, stop: the session was not started with
   the tune5 settings.
3. `squeue -u vpa`: release any `tune5_window_*` this workflow left behind
   (`bash /global/homes/v/vpa/.claude/bin/tune5_release.sh <id>`). Never touch other jobs.
4. `iris user vpa`: the user's own share of `m4599_g` left = Allocated(user) − Charged(user), in GPU
   node-hours. NERSC finalizes it once a day, so also subtract what is not in it yet: the user's
   pending and running `m4599_g` jobs at nodes × time limit (`squeue -u vpa -o '%i %j %a %D %l %T'`)
   and today's tune5 ledger charges. If what is left is less than the round needs, open nothing:
   report the numbers and ask. When the share is used up, NERSC refuses every job on `m4599_g`, and
   other approved work of the user's stops with it. Repeat this check before each round.

## Submitting (the only shapes the hook lets through; one plain line, absolute paths)

- Window: `bash /global/homes/v/vpa/.claude/bin/tune5_alloc.sh fo 4 <hours 1-4>  # tune5:fo` (or
  `screen ... # tune5:screen`), with the Bash tool's timeout at 600000 ms (the request gives up after
  5 min). Note the job id it prints; use only that id.
- Run onto the window, one at a time and with `run_in_background: true` (a foreground call is
  killed at 10 min): `SLURM_JOB_ID=<that id> <launcher settings> bash .../perlmutter/forces48_puregrid.sh`
  on an fo window, or `traj48_puregrid.sh` (N_TRAJ 1-3) on a screen window. Release the window with
  `tune5_release.sh <id>` as soon as its last run ends; if it reports that squeue failed, retry.
  If the permission system blocks a launcher on the window, release it at once and ask the user.
- Batch screen, only if interactive windows do not work: `env N_TRAJ=<1-3> RUN=<name> <run settings>
  sbatch --time=HH:MM:SS -J tune5_<name> --export=ALL
  /global/cfs/cdirs/m4599/Users/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/hasenbusch_tune/perlmutter/traj48_puregrid_batch.sh  # tune5:screen`.
  Size `--time` from tune5.py's prediction plus 30 min (the hook allows 1 h to 2 h x N_TRAJ + 1 h).
  Every job name is used once. If an sbatch call fails or times out, check `squeue -u vpa` before
  resubmitting: it may have been queued anyway.
- Cancel a wrong or hung tune5 batch job: `bash /global/homes/v/vpa/.claude/bin/tune5_cancel.sh <jobid>`.
- The hook blocks any Bash or Monitor line that mentions a Slurm command, a launcher name
  (`*_puregrid*`) or the tune5 tools, even inside a grep pattern or a pipe (`ps | grep srun`).
  Plain `grep`/`cat` in Bash is fine otherwise. In Monitor watchers match `s.un: error` for launcher
  errors and `Abort|aborted|MPI_Abort` for crashes (not `abort`: it matches `bounds_check_abort=1`).
- A run directory on CFS can stall a few minutes before Grid starts (11-02: 5 min); size windows
  with that margin.

## Each round

1. Price candidates with `autotune/tune5.py` (cost model, force model); keep only those clearly cheaper
   than the reference.
2. Write each candidate's prediction (cost, risky piece) in the session progress doc BEFORE it runs.
3. Run; one watcher per job, no log polling.
4. Check every run's ENV line and `[Levels]` banner against the request; on a mismatch cancel the job
   and stop until it is explained.
5. Record ΔH, wall, cycle and the verdict in the progress doc; choose the next round.

A candidate that passes its first screen gets a few more 1-3 trajectory screens on other seeds and
one rougher start (`DIAG_START_CFG`), each paired with the reference on the same seed and start, before
it goes on the shortlist.

## The shortlist (the session's result)

When the screens are done, or a budget runs out, or no candidate is left to try: write a shortlist
section in the progress doc and report it. Per candidate: ladder, tail and layout; predicted and
measured cost per trajectory against the reference; ΔH on every paired screen; the risky piece; and a
recommendation (20 trajectories from cfg_2000, the multistart test, or drop). Then stop.

Progress doc: `__docs/<YYYY_MM_DD>_tune5_session_<n>.md`, a section stamp under every header.

## Never

- Reset a budget, edit the hook, the settings, the ledger or this skill, or start another `claude`.
- Call sbatch, salloc or srun from inside a script, or run a launcher you edited.
- Commit, push, or edit `_SUM.md`, `_LEDGER.md`, `_TOC.md`: report instead.
- Leave an interactive allocation idle.
