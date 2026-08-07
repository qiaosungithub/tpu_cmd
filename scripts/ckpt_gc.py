#!/usr/bin/env python3
"""Delete checkpoints a finished run will never read again.

WHY THIS EXISTS. `save_checkpoint` writes `step_<N>/` and nothing ever removed
one: the code uses orbax's `StandardCheckpointer`, and `max_to_keep` belongs to
`CheckpointManager`, which it does not use. A 150k-step sudoku run at
`checkpoint_interval_steps: 2500` therefore leaves 60 directories of 331 MiB --
17 GiB for one run, of which two directories are ever read again. 1850 of them
across one cell is what pushed a personal CNS quota over its 500 GiB ceiling
and poisoned every subsequent write in that cell.

WHAT IT KEEPS, per run directory:

  1. the LATEST checkpoint -- ALWAYS, no exceptions
  2. the newest COMPLETE checkpoint, when the latest is a torn write
  3. every step that is a multiple of --milestone (default 50000)
  4. every COMPLETE best-checkpoint copy -- the run's peaks
  5. anything that is not a `step_<N>` directory

Everything else goes. Quota counts bytes AFTER replication, so the space
recovered is roughly 2.9x the payload deleted under `r=3.2` (see
wiki_agents/storage.md).

A PEAK HAS TWO NAMES, AND BOTH ARE PERMANENT:

    checkpoint_best_<slug>_<N>   one per tracked metric, e.g.
                                 `checkpoint_best_D16_ema_acc_95000`
    checkpoint_best_<N>          LEGACY, from when retention tracked a single
                                 metric. Still on CNS, and for several FINISHED
                                 runs it is the ONLY surviving peak.

The training job keeps the best under EACH tracked metric because a
single-metric policy inherits that metric's bugs irreversibly: `solution_acc`
scored only the painted cells, and three of four completed runs therefore
retained a step that was not the peak while ordinary retention deleted the real
one. Both shapes are matched here, and neither is ever swept.

BOTH ARE ALREADY SAFE FROM RULE 5 -- the names are deliberately outside the
`step_<N>` namespace `_STEP_RE` matches, precisely so no step-based GC has to be
taught about them. They are called out anyway because a peak is the one
checkpoint that cannot be re-created: it survives its own run's retention, so
the only thing left that could delete it is a human sweeping a cell by hand.
The plan therefore REPORTS the space they occupy rather than hiding it -- two
extra checkpoints per run is real when a cell holds a thousand of them, and a
number nobody can see is a number nobody can decide about.

The ONE it does delete is a best copy with no `extra.json`: the training job
writes that marker last, so its absence means a promotion was preempted
mid-copy. Nothing else will ever clean that up -- the very invisibility that
protects a good peak protects a torn one.

SAFETY. Dry-run is the default: it prints the plan and touches nothing; `--go`
is required to delete. Keeping the latest unconditionally is what makes this
safe to run against a cell with live jobs on it: the newest directory is either
the one auto-resume needs or the one a running job is writing, and the script
never has to guess which.

USAGE
  tpu gc                                  # dry run over the default root
  tpu gc --go                             # actually delete
  tpu gc --root /cns/<cell>-d/home/$USER  # another cell
  tpu gc --only eqr_diff_l4_v2 --go       # one run
  tpu gc --milestone 25000                # keep a denser ladder
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys

_STEP_RE = re.compile(r"^step_(\d+)(?:_.*)?$")
# The promoted peaks, BOTH SHAPES: `checkpoint_best_<slug>_<N>` and the legacy
# `checkpoint_best_<N>` (the optional group). Neither is matched by `_STEP_RE`,
# by design -- see the module docstring; the training job
# (`utils/ckpt_util.py::promote_best_checkpoint`) picks the names for exactly
# that reason.
#
# The step is read unambiguously even out of a slug full of underscores and
# digits (`D16_ema_acc`): `\d+$` is anchored, so the separator can only be an `_`
# whose entire tail is digits, and a digits-only tail contains no `_` -- exactly
# one position qualifies.
_BEST_RE = re.compile(r"^checkpoint_best_(?:.+?_)?(\d+)$")
_DEFAULT_ROOT = f"/cns/yuskedq-d/home/{os.environ.get('USER', 'qiaos')}"
_PARALLEL = 24


def _run(args: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 1, ""


def _ls(path: str) -> list[str]:
    """Immediate children of `path`, basenames only, [] on any failure."""
    rc, out = _run(["fileutil", "ls", path])
    if rc != 0:
        return []
    return [ln.rstrip("/").rsplit("/", 1)[-1] for ln in out.splitlines() if ln.strip()]


def _du_bytes(path: str) -> int:
    rc, out = _run(["fileutil", "du", "-s", path], timeout=240)
    if rc != 0:
        return 0
    try:
        return int(out.split()[0])
    except (IndexError, ValueError):
        return 0


def _gib(n: int) -> float:
    return n / (1024.0 ** 3)


def find_ckpt_dir(root: str, run: str) -> str | None:
    """Resolve `<root>/<run>/logs/<project>/<xid_dir>/checkpoints`.

    The layout nests the experiment id under a project name, and the xid
    directory carries the whole experiment title, so it cannot be predicted --
    it has to be listed. Returns None when the run has no checkpoints dir.
    """
    base = f"{root}/{run}/logs"
    for project in _ls(base):
        for xid_dir in _ls(f"{base}/{project}"):
            cand = f"{base}/{project}/{xid_dir}/checkpoints"
            if _ls(cand):
                return cand
    return None


def plan_for(ckpt_dir: str, milestone: int) -> tuple[list[str], list[str], str | None, list[str]]:
    """(keep, delete, skip_reason, best) for one checkpoints/ directory.

    `best` is the COMPLETE best-checkpoint copies, in either naming shape,
    reported so the space they hold is visible; they are never in `delete`. One
    with no `extra.json` IS in `delete` -- see the module docstring.

    Keyed by NAME and not by step, because a run tracking two metrics has two
    peaks and they need not agree on a step; keying by step would silently drop
    one of them from the report and from the retained set.
    """
    steps: dict[int, str] = {}
    bests: dict[str, int] = {}
    extras: list[str] = []
    for name in _ls(ckpt_dir):
        m = _STEP_RE.match(name)
        if m:
            steps[int(m.group(1))] = name
            continue
        b = _BEST_RE.match(name)
        if b:
            bests[name] = int(b.group(1))
        else:
            extras.append(name)
    if not steps and not bests:
        return [], [], "no step_ dirs", []

    # ONE recursive listing, not one per directory. Probing each `step_<N>/` for
    # its extra.json serially cost ~4 minutes on a 60-checkpoint run: the round
    # trips dominate, and there are `len(steps)` of them. `ls -R` pays a single
    # round trip for the whole tree (see wiki_agents/storage.md: "cost is round
    # trips, not bytes").
    rc, out = _run(["fileutil", "ls", "-R", ckpt_dir], timeout=600)
    listed = rc == 0 and bool(out.strip())

    def _has_extra(name: str) -> bool:
        if listed:
            return f"/{name}/extra.json" in out
        return "extra.json" in _ls(f"{ckpt_dir}/{name}")  # slow probe, never a guess

    # THE PEAK IS NEVER DELETED, and a torn one always is. `extra.json` is
    # written last by the promotion, so its absence means the copy was preempted
    # part-way; nothing else will ever collect that, because the name is outside
    # every step-based sweep's namespace -- the same property that keeps a good
    # peak safe from this script.
    best = [n for n in sorted(bests, key=lambda n: (bests[n], n)) if _has_extra(n)]
    torn_best = [n for n in sorted(bests, key=lambda n: (bests[n], n)) if n not in set(best)]
    if not steps:
        # A run swept down to just its peak. `best` still travels back, and the
        # caller prints it before honouring the skip.
        return [], torn_best, None if torn_best else "no step_ dirs", best

    # THE LATEST STEP IS ALWAYS KEPT, unconditionally.
    #
    # It is the one directory that cannot be re-derived: auto-resume restores
    # from it, and if the run is still alive it is the file being written right
    # now. Deciding "is this run dead?" from the filesystem is guesswork -- a
    # torn write and a mid-save write look identical -- so this script does not
    # try. One extra checkpoint per run is a rounding error against the tens of
    # GiB the rest of the sweep frees; deleting a live run's newest state is not
    # recoverable at any price. Keep the newest COMPLETE step too, so a run
    # whose latest is a torn write still retains something restorable.
    latest = max(steps)
    complete = [s for s in steps if _has_extra(steps[s])]

    keep_steps = {latest}
    if complete:
        keep_steps.add(max(complete))
    keep_steps |= {s for s in steps if milestone > 0 and s % milestone == 0}
    keep = [steps[s] for s in sorted(keep_steps)]
    delete = [steps[s] for s in sorted(steps) if s not in keep_steps] + torn_best
    return keep, delete, None, best


def _report_best(bests: list[str], *, measure: bool) -> None:
    """Print the retained peaks and what they cost. Never deletes anything.

    Separate from the delete plan on purpose: these are RETAINED, and printing
    them beside the victims would read as a deletion list.
    """
    if not bests:
        return
    print(f"\nRETAINED PEAKS ({len(bests)}) -- never deleted, and no run's own retention removes them either:")
    held = 0
    if measure:
        with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
            sizes = list(pool.map(_du_bytes, bests))
        held = sum(sizes)
        for path, size in zip(bests, sizes):
            print(f"  {_gib(size):6.1f} GiB  {path}")
        print(f"  payload {_gib(held):.1f} GiB  ->  about {_gib(held) * 2.89:.0f} GiB of quota at r=3.2")
    else:
        for path in bests:
            print(f"  {path}")
        print("  (size not measured; drop --no-size to price them)")
    print("  Delete one only by hand, and only knowingly: it is the peak-scoring")
    print("  checkpoint of its run, and its step is off the retention ladder.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prune checkpoints: keep the latest and every --milestone-th step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default=_DEFAULT_ROOT, help=f"CNS home to sweep (default {_DEFAULT_ROOT})")
    ap.add_argument("--milestone", type=int, default=50000, help="keep every N-th step (default 50000; 0 disables)")
    ap.add_argument("--only", action="append", default=[], help="restrict to these run dirs (repeatable)")
    ap.add_argument("--go", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    ap.add_argument("--no-size", action="store_true",
                    help="skip the du pass; prints the plan in seconds instead of minutes")
    args = ap.parse_args()

    runs = args.only or sorted(_ls(args.root))
    if not runs:
        print(f"No run directories under {args.root} (or it is unreadable).")
        return 1

    print(f"{'DELETING' if args.go else 'DRY RUN'} | root={args.root} | keep: latest + every {args.milestone} steps + every checkpoint_best_* (both naming shapes)")
    print(f"Scanning {len(runs)} run director{'y' if len(runs)==1 else 'ies'}...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
        ckpt_dirs = dict(zip(runs, pool.map(lambda r: find_ckpt_dir(args.root, r), runs)))
        live = {r: d for r, d in ckpt_dirs.items() if d}
        plans = dict(zip(live, pool.map(lambda d: plan_for(d, args.milestone), live.values())))

    total_del = total_keep = 0
    victims: list[str] = []
    bests: list[str] = []
    skipped: list[tuple[str, str]] = []
    for run in sorted(plans):
        keep, delete, skip, best = plans[run]
        # BEFORE the skip, deliberately. A run whose steps have all been swept
        # already still holds its peak, and that is exactly the run whose peak
        # would otherwise be invisible -- there is nothing else left to print.
        bests += [f"{live[run]}/{b}" for b in best]
        if skip:
            skipped.append((run, skip))
            continue
        total_keep += len(keep)
        total_del += len(delete)
        victims += [f"{live[run]}/{d}" for d in delete]
        if delete and not args.quiet:
            kept = ", ".join(k.replace("step_", "") for k in keep[:6]) + (" ..." if len(keep) > 6 else "")
            peak = "".join(f" +peak {b[len('checkpoint_best_'):]}" for b in best)
            print(f"  {run}: delete {len(delete)}, keep {len(keep)} ({kept}){peak}")

    for run, why in skipped:
        print(f"  SKIP {run}: {why}")

    # THE PEAKS ARE REPORTED EVEN WHEN THERE IS NOTHING TO DELETE. They are the
    # one class of checkpoint this script will not touch and no run's own
    # retention will either, so a cell fills with them quietly. Someone sweeping
    # a cell needs to see the space before deciding, and "invisible to the GC"
    # must not also mean invisible to the human.
    _report_best(bests, measure=not args.no_size)

    if not victims:
        print("\nNothing to delete.")
        return 0

    # `fileutil du` is the slow part of a dry run by an order of magnitude --
    # it walks every object -- so it is skippable when you only want the plan.
    freed = 0
    if args.no_size:
        print(f"\n  keeping {total_keep} checkpoints, deleting {total_del} (size not measured)")
    else:
        print(f"\nMeasuring {len(victims)} doomed checkpoints...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
            freed = sum(pool.map(_du_bytes, victims))
        print(f"  payload {_gib(freed):.1f} GiB  ->  about {_gib(freed) * 2.89:.0f} GiB of quota at r=3.2")
        print(f"  keeping {total_keep} checkpoints, deleting {total_del}")

    if not args.go:
        print("\nDry run. Re-run with --go to delete.")
        return 0

    print("\nDeleting...")
    def _rm(p: str) -> tuple[str, bool]:
        rc, _ = _run(["fileutil", "rm", "-R", "-f", p], timeout=300)
        return p, rc == 0
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
        for path, good in pool.map(_rm, victims):
            ok += good
            if not good:
                print(f"  FAILED {path}")
    freed_msg = f" Freed about {_gib(freed) * 2.89:.0f} GiB of quota." if freed else ""
    print(f"\nDeleted {ok}/{len(victims)}.{freed_msg}")
    print("Quota release is not instant; `fileutil quota <user> <cell>` lags by a few minutes.")
    return 0 if ok == len(victims) else 1


if __name__ == "__main__":
    sys.exit(main())
