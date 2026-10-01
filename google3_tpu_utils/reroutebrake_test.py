# Copyright 2026 Google LLC. All Rights Reserved.
"""Re-route churn brake (global rate cap) and the no-give-up rule.

Measured 2026-09-01: one eval job burned 7 XIDs in 2.5 hours -- placed,
cancelled, re-queued, placed again -- while the queue showed reroutes=2 because
the first 5 lived on a row that had since been dequeued. So the brake is a
global rate cap keyed on NOTHING: it counts what the re-router itself did, so
renaming or re-enqueuing the car cannot evade it. The re-route deadline itself
is fixed (route_lib.patience_s); it does not grow with the reroute count.

There is deliberately NO give-up bound (removed 2026-09-11 by operator request):
a job is re-routed however many times it takes and is never auto-parked as HELD
for churning. entry.reroutes is surfaced on the board instead, so a human can
see a high count and step in -- that is the safety valve that replaced the bound.

The NEGATIVE CONTROLS carry this file. The brake can only make the re-router
act LESS, and the failure mode of "less" is a genuinely stuck car that nobody
rescues -- so it is paired with tests proving it stays off below its limit.
"""

import time as _time

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

_BASE = 600.0


def _entry(reroutes=0, age_s=10_000.0, now=1_000_000.0,
           state=R.JobState.SUBMITTED):
  e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
  e.state = state
  e.xid = '285502876'
  e.reroutes = reroutes
  e.submitted_at = now - age_s
  return e


class NoGiveUpBoundTest(googletest.TestCase):
  """The give-up bound was REMOVED 2026-09-11 (operator request). A job is
  re-routed as many times as it takes and is never auto-parked as HELD for
  churning; the pair cooldown and the global rate cap are the only brakes."""

  def test_high_reroute_count_still_reroutes(self):
    # 99 moves in -- far past the old bound of 6 -- and it is STILL eligible
    # once its deadline has passed. No bound turns it off.
    now = 1_000_000.0
    e = _entry(reroutes=99, age_s=999_999.0, now=now)
    self.assertTrue(R.needs_reroute(e, now, _BASE))

  def test_a_thousand_moves_in_still_reroutes(self):
    # There is no ceiling at all: even absurd counts stay eligible. The full
    # run_reroute path (never parking a high-count row as HELD) is covered by
    # route_check_test.test_high_reroute_count_is_still_rechecked_and_rerouted.
    now = 1_000_000.0
    e = _entry(reroutes=1000, age_s=999_999.0, now=now)
    self.assertTrue(R.needs_reroute(e, now, _BASE))


class GlobalBrakeTest(googletest.TestCase):

  def test_brake_trips_at_the_limit(self):
    now = 1_000_000.0
    times = [now - i * 10 for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR)]
    self.assertTrue(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: below the limit the brake must stay OFF ---
  def test_NEGCTL_below_limit_does_not_trip(self):
    now = 1_000_000.0
    times = [now - i * 10 for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR - 1)]
    self.assertFalse(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: OLD churn must age out of the window ---
  def test_NEGCTL_stale_history_does_not_trip_it(self):
    now = 1_000_000.0
    times = [now - 3601.0 - i for i in range(50)]   # all just outside the hour
    self.assertFalse(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: empty history fails OPEN, never suspends ---
  def test_NEGCTL_empty_history_fails_open(self):
    self.assertFalse(R.global_reroute_brake([], 1_000_000.0))

  def test_tripped_brake_leaves_the_candidate_alone(self):
    now = _time.time()
    hist = [now - i for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR + 2)]
    path = self.create_tempfile().full_path
    RC._save_reroute_history(hist, path)
    e = _entry(reroutes=0, age_s=99_999.0, now=now)
    entries, log = RC.run_reroute([e], now=now, probe=_Probe('PENDING'),
                                  dry_run=False, history_file=path)
    self.assertEqual(entries[0].state, R.JobState.SUBMITTED)  # untouched
    self.assertIn('GLOBAL BRAKE ENGAGED', '\n'.join(log))


class HistoryPersistenceTest(googletest.TestCase):

  def test_roundtrip_keeps_only_the_window(self):
    now = _time.time()
    path = self.create_tempfile().full_path
    RC._save_reroute_history([now - 10, now - 20, now - 7200], path)
    got = RC._load_reroute_history(path)
    self.assertLen(got, 2)

  # --- NEGATIVE CONTROL: a corrupt/missing file must fail OPEN (empty) ---
  def test_NEGCTL_corrupt_history_reads_empty_not_crash(self):
    path = self.create_tempfile(content='{not json').full_path
    self.assertEqual(RC._load_reroute_history(path), [])
    self.assertEqual(RC._load_reroute_history('/nonexistent/x.json'), [])


class _Probe:

  def __init__(self, state):
    self._s = state

  def status(self, xid):
    del xid
    return self._s


if __name__ == '__main__':
  googletest.main()
