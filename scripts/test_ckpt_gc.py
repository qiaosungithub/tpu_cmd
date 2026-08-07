#!/usr/bin/env python3
"""`tpu gc` must never sweep a peak, in either naming shape.

WHY THIS FILE EXISTS. The safety of a promoted peak is a NAMING CONTRACT held
across two repositories: EqR-jax writes `checkpoint_best_<slug>_<N>/` (and
historically `checkpoint_best_<N>/`), and this script must recognise both. The
contract is invisible -- nothing fails loudly when it breaks; a peak is simply
deleted, and a peak is the one checkpoint that cannot be re-created. Three of
four completed runs already retained the WRONG step because retention tracked a
single buggy metric, and the true peaks were gone by the time anyone looked.

Runs without CNS, without `fileutil`, and without pytest:

    python3 scripts/test_ckpt_gc.py       # standalone
    python3 -m pytest scripts/test_ckpt_gc.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ckpt_gc  # noqa: E402

# A run as `fileutil ls` reports it: {dir: [child names]}, plus the files each
# checkpoint directory holds. `extra.json` present means COMPLETE.
_LADDER = [f"step_{s}" for s in range(50000, 150001, 25000)]


def _fake_fs(names, *, torn=()):
    """Serve `_ls` and the `ls -R` fast path out of a dict, no fileutil needed."""
    root = "/cns/fake-d/home/u/run/logs/EqR-jax/xid_1/checkpoints"
    recursive = "\n".join(
        f"{root}/{n}/state/shard" + ("" if n in torn else f"\n{root}/{n}/extra.json")
        for n in names
    )

    def _run(args, timeout=180):
        if args[:2] == ["fileutil", "ls"] and "-R" in args:
            return 0, recursive
        if args[:2] == ["fileutil", "ls"]:
            path = args[-1]
            if path == root:
                return 0, "\n".join(f"{root}/{n}" for n in names)
            base = path.rstrip("/").rsplit("/", 1)[-1]
            if base in names:
                kids = ["state"] + ([] if base in torn else ["extra.json"])
                return 0, "\n".join(f"{path}/{k}" for k in kids)
            return 1, ""
        return 1, ""

    return root, _run


def _plan(names, *, torn=(), milestone=50000):
    root, runner = _fake_fs(names, torn=torn)
    original = ckpt_gc._run
    ckpt_gc._run = runner
    try:
        return ckpt_gc.plan_for(root, milestone)
    finally:
        ckpt_gc._run = original


# --------------------------------------------------------------------------- #
# the regex contract, both shapes
# --------------------------------------------------------------------------- #


def test_both_naming_shapes_are_recognised_as_peaks():
    for name, step in [
        ("checkpoint_best_130000", 130000),          # legacy, on CNS today
        ("checkpoint_best_D16_ema_acc_120000", 120000),
        ("checkpoint_best_D16_ema_solution_acc_95000", 95000),
        ("checkpoint_best_metric_0", 0),
    ]:
        match = ckpt_gc._BEST_RE.match(name)
        assert match, name
        assert int(match.group(1)) == step, (name, match.group(1))


def test_a_peak_never_looks_like_a_step_to_the_sweeper():
    """The structural property the whole design rests on."""
    for name in ("checkpoint_best_130000", "checkpoint_best_D16_ema_acc_120000"):
        assert ckpt_gc._STEP_RE.match(name) is None


def test_the_step_survives_a_slug_full_of_underscores_and_digits():
    """A slug is not a clean token: `D16_ema_acc16` has both separators and
    digits in it. Reading the step out of the middle of the number would make
    two peaks collide on the wrong step and drop one from the retained set."""
    for name, step in [
        ("checkpoint_best_D16_ema_acc_120000", "120000"),
        ("checkpoint_best_D4_ema_acc16_95000", "95000"),
        ("checkpoint_best_D16_ema_solution_acc_5000", "5000"),
    ]:
        assert ckpt_gc._BEST_RE.match(name).group(1) == step, name


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #


def test_neither_shape_is_ever_in_the_delete_list():
    peaks = ["checkpoint_best_130000", "checkpoint_best_D16_ema_acc_120000"]
    keep, delete, skip, best = _plan(_LADDER + peaks)
    assert skip is None
    assert sorted(best) == sorted(peaks)
    for peak in peaks:
        assert peak not in delete
    assert delete, "the ladder must still be swept, or this proves nothing"


def test_two_peaks_at_different_steps_are_both_reported():
    """Keyed by NAME, not by step: a run tracking two metrics has two peaks and
    they need not agree. Keying by step drops one silently."""
    peaks = [
        "checkpoint_best_D16_ema_solution_acc_130000",
        "checkpoint_best_D16_ema_acc_80000",
    ]
    _keep, delete, _skip, best = _plan(_LADDER + peaks)
    assert sorted(best) == sorted(peaks)
    assert not [p for p in peaks if p in delete]


def test_a_torn_peak_is_collected_in_either_shape():
    """No `extra.json` means a promotion died mid-copy. Nothing else can see it:
    the invisibility that protects a good peak protects a torn one."""
    good = "checkpoint_best_D16_ema_acc_120000"
    torn_new = "checkpoint_best_D16_ema_solution_acc_95000"
    torn_old = "checkpoint_best_33000"
    _keep, delete, _skip, best = _plan(
        _LADDER + [good, torn_new, torn_old], torn=(torn_new, torn_old)
    )
    assert best == [good]
    assert torn_new in delete and torn_old in delete
    assert good not in delete


def test_a_run_swept_down_to_only_its_peaks_keeps_them():
    """The case where the peaks are ALL that is left -- and where deleting one
    is unrecoverable."""
    peaks = ["checkpoint_best_130000", "checkpoint_best_D16_ema_acc_80000"]
    keep, delete, _skip, best = _plan(peaks)
    assert keep == [] and delete == []
    assert sorted(best) == sorted(peaks)


def test_the_ladder_and_the_latest_still_survive():
    """The GC's own contract, unchanged by any of this."""
    keep, delete, _skip, _best = _plan(_LADDER + ["checkpoint_best_D16_ema_acc_120000"])
    assert "step_150000" in keep and "step_50000" in keep and "step_100000" in keep
    assert "step_75000" in delete and "step_125000" in delete


def test_a_directory_that_is_neither_is_left_alone():
    """`eval_preds/` and friends: rule 5 is unchanged."""
    _keep, delete, _skip, best = _plan(_LADDER + ["eval_preds", "checkpoint_best_130000"])
    assert "eval_preds" not in delete
    assert best == ["checkpoint_best_130000"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except Exception as exc:  # noqa: BLE001 - a broken contract raises
                # NOT just AssertionError: a regex that stops matching makes
                # `.group()` an AttributeError, and a runner that only catches
                # assertions would die on the first one and hide the rest.
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
