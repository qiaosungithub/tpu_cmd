# Copyright 2026 Google LLC. All Rights Reserved.
"""A job killed at Borg TURN-UP is retried, not left terminal FAILED.

Measured incident xid 291448297 (exp ouro-lr4e-3-wd005-clip5, a b200-8): the
work unit died at turn-up with

    Turning up failed: INVALID_ARGUMENT: Try to create job in <cell>...

Its logs/ dir was EMPTY -- it never entered training (not OOM, not a code bug).
reconcile marked it FAILED and did NOT auto-retry, so a human had to re-enqueue
it by hand (the manual retry landed on another, healthy cell). This class of
error is transient; an immediate re-submit works.

This is a THIRD failure taxonomy, distinct from both existing auto-resume paths:
  * plan_pruned_restart keys on CHECKPOINT evidence (a healthy run killed after
    it trained); a turn-up death has no checkpoint and never trained.
  * classify_failure keys on the queue row's last_reason STRING; a turn-up death
    leaves no such reason.
The only positive signal is the XM termination MESSAGE, threaded in through a
NEW fail-open failure_message() channel that leaves status()'s contract intact.

The NEGATIVE CONTROLS are the load-bearing half of this file. Turning "I could
not read why it died" into "retry it" is exactly the mistake that would burn
PROD budget replaying a real bug, so every retry is paired with a control that
must leave the row FAILED:
  * budget spent            -> FAILED (cannot loop forever);
  * unreadable/empty message -> FAILED (fail closed);
  * a code-bug signature     -> FAILED (a real crash outranks a turn-up phrase);
  * a non-matching message   -> FAILED (only the whitelist retries);
  * the flag OFF             -> FAILED (default behaviour unchanged);
  * the global brake engaged -> FAILED (fleet-wide churn is not self-healed);
  * a normal preemption      -> still takes the EXISTING pruned/preempted path.
"""

import time as _time

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

# The exact xid-291448297 termination text: it matches all three whitelist
# needles ('turning up failed', 'invalid_argument', 'try to create job').
_TURNUP_MSG = ('Turning up failed: INVALID_ARGUMENT: Try to create job in '
               'yutulpz-borg: allocation not found')
_TORCH_CKPT = '/cns/x/steps/step_1024.pt'


# ---------------------------------------------------------------------------
# Fakes -- scripted, offline, no XManager / CNS / shell.
# ---------------------------------------------------------------------------
class _Status:
  """A fake work-unit status object with a .message, like the XM proto."""

  def __init__(self, message):
    self.message = message


class _WU:
  """A fake XM work unit exposing only the attrs the message readers touch."""

  def __init__(self, message='', failed=False, stopped=False,
               status_message='', error_message='', failure_reason='',
               has_status=True):
    self.status = _Status(message) if has_status else None
    self.is_failed = failed
    self.is_stopped = stopped
    self.status_message = status_message
    self.error_message = error_message
    self.failure_reason = failure_reason


class _Exp:

  def __init__(self, wus):
    self._wus = wus

  def get_work_units(self, populate_detailed_executable_status=False):
    del populate_detailed_executable_status
    return list(self._wus)


class _FakeClient:
  """get_experiment returns an _Exp or raises (a probe failure)."""

  def __init__(self, by_xid):
    self._by_xid = by_xid

  def get_experiment(self, xid):
    v = self._by_xid[int(xid)]
    if isinstance(v, Exception):
      raise v
    return v


def _msg_probe(by_xid):
  p = RC.XManagerStatusProbe()
  p._client = _FakeClient(by_xid)  # pylint: disable=protected-access
  return p


class _FakeProbe:
  """status() + failure_message() in one object -- production's
  XManagerStatusProbe carries both, so one instance serves as both the
  _StatusProbe and the fail-open _FailureProbe. `raise_msg` makes
  failure_message() throw, to prove the caller fails OPEN on an exploding probe.
  """

  def __init__(self, status_by_xid, msg_by_xid=None, raise_msg=False):
    self._status = status_by_xid
    self._msg = msg_by_xid or {}
    self._raise_msg = raise_msg

  def status(self, xid):
    return self._status.get(xid, RC.STATUS_UNKNOWN)

  def failure_message(self, xid):
    if self._raise_msg:
      raise RuntimeError('probe blew up reading the message')
    return self._msg.get(xid)


class _FakeRestartEvidence:
  """Scripted (code_bug, checkpoint, trained) per xid, standing in for
  CnsRestartEvidence so the code-bug guard and the pruned/preempted path run
  with no CNS I/O. Default (None, None, False)."""

  def __init__(self, by_xid=None):
    self._by_xid = by_xid or {}

  def code_bug_and_checkpoint(self, entry):
    return self._by_xid.get(entry.xid, (None, None, False))


def _entry(job_id='j', xid='291448297', state=R.JobState.SUBMITTED,
           submitted_at=0.0, auto_resumes=0, last_reason='', **launch_kwargs):
  e = R.QueueEntry(job_id=job_id, power='b200-8', allowed_archs=['b200'])
  e.state = state
  e.xid = xid
  e.cell = 'yutulpz'
  e.arch = 'b200'
  e.chips = 8
  e.submitted_at = submitted_at
  e.auto_resumes = auto_resumes
  if last_reason:
    e.last_reason = last_reason
  e.launch_kwargs = dict(launch_kwargs) if launch_kwargs else {
      'config': 'ouro', 'exp_name': 'ouro-lr4e-3-wd005-clip5'}
  return e


# ===========================================================================
# 1. PURE: the turn-up whitelist (route_lib.is_turnup_retryable).
# ===========================================================================
class IsTurnupRetryableTest(googletest.TestCase):

  def test_the_exact_incident_message_matches(self):
    self.assertTrue(R.is_turnup_retryable(_TURNUP_MSG))

  def test_each_whitelist_phrase_matches_on_its_own(self):
    for m in ('Turning up failed for the slice',
              'INVALID_ARGUMENT from the allocator',
              'Try to create job in some-cell'):
      self.assertTrue(R.is_turnup_retryable(m), m)

  def test_match_is_case_insensitive(self):
    self.assertTrue(R.is_turnup_retryable(_TURNUP_MSG.upper()))
    self.assertTrue(R.is_turnup_retryable(_TURNUP_MSG.lower()))

  # --- NEGATIVE CONTROL: empty / None fails CLOSED (unreadable != retry) ---
  def test_NEGCTL_empty_or_none_is_not_retryable(self):
    self.assertFalse(R.is_turnup_retryable(None))
    self.assertFalse(R.is_turnup_retryable(''))
    self.assertFalse(R.is_turnup_retryable('   '))

  # --- NEGATIVE CONTROL: a real crash / preemption is NOT a turn-up error ---
  def test_NEGCTL_unrelated_failures_do_not_match(self):
    for m in ('RESOURCE_EXHAUSTED: OOM killed',
              'Traceback (most recent call last): ValueError',
              'DESCHEDULED: resource-guarantee reclaim',
              'Preempted by a higher-priority workload',
              'CUDA error: an illegal memory access'):
      self.assertFalse(R.is_turnup_retryable(m), m)


# ===========================================================================
# 2. PURE: the turn-up decision (route_lib.plan_turnup_retry).
# ===========================================================================
class PlanTurnupRetryTest(googletest.TestCase):

  def test_terminal_matching_within_budget_retries(self):
    e = _entry(auto_resumes=0)
    verdict, _ = R.plan_turnup_retry(
        e, xm_terminal=True, failure_message=_TURNUP_MSG, max_auto_resumes=3)
    self.assertEqual(verdict, R.RETRY_TURNUP)

  # --- NEGATIVE CONTROL: budget spent -> HOLD (bounded, cannot loop) ---
  def test_NEGCTL_budget_exhausted_holds(self):
    e = _entry(auto_resumes=3)
    verdict, why = R.plan_turnup_retry(
        e, xm_terminal=True, failure_message=_TURNUP_MSG, max_auto_resumes=3)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('budget spent', why)

  # --- NEGATIVE CONTROL: a code bug outranks a turn-up phrase -> HOLD ---
  def test_NEGCTL_code_bug_holds_even_if_message_matches(self):
    e = _entry()
    verdict, why = R.plan_turnup_retry(
        e, xm_terminal=True, failure_message=_TURNUP_MSG,
        code_bug='segfault (SIGSEGV)', max_auto_resumes=3)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('code bug', why)

  # --- NEGATIVE CONTROL: unreadable / non-matching message -> HOLD ---
  def test_NEGCTL_unreadable_or_nonmatching_holds(self):
    e = _entry()
    for msg in (None, '', 'RESOURCE_EXHAUSTED: OOM'):
      verdict, _ = R.plan_turnup_retry(
          e, xm_terminal=True, failure_message=msg, max_auto_resumes=3)
      self.assertEqual(verdict, R.HOLD, repr(msg))

  # --- NEGATIVE CONTROL: not terminal on XM -> HOLD (nothing to retry) ---
  def test_NEGCTL_not_terminal_holds(self):
    e = _entry()
    verdict, _ = R.plan_turnup_retry(
        e, xm_terminal=False, failure_message=_TURNUP_MSG, max_auto_resumes=3)
    self.assertEqual(verdict, R.HOLD)


# ===========================================================================
# 3. PURE + PROBE: the detailed-message channel (route_check).
# ===========================================================================
class WuStatusMessageTest(googletest.TestCase):

  def test_reads_status_message_first(self):
    self.assertEqual(RC._wu_status_message(_WU(message=_TURNUP_MSG)),
                     _TURNUP_MSG)

  def test_falls_back_to_flat_attrs(self):
    self.assertEqual(
        RC._wu_status_message(_WU(has_status=False, status_message='sm')), 'sm')
    self.assertEqual(
        RC._wu_status_message(_WU(has_status=False, error_message='em')), 'em')
    self.assertEqual(
        RC._wu_status_message(_WU(has_status=False, failure_reason='fr')), 'fr')

  def test_no_message_anywhere_is_none(self):
    self.assertIsNone(RC._wu_status_message(_WU(has_status=False)))


class FailureMessageProbeTest(googletest.TestCase):

  def test_reads_the_failed_wus_message(self):
    p = _msg_probe({1: _Exp([_WU(message=_TURNUP_MSG, failed=True)])})
    self.assertEqual(p.failure_message('1'), _TURNUP_MSG)

  def test_prefers_the_FAILED_wu_over_a_running_one(self):
    p = _msg_probe({1: _Exp([_WU(message='still coming up', failed=False),
                             _WU(message=_TURNUP_MSG, failed=True)])})
    self.assertEqual(p.failure_message('1'), _TURNUP_MSG)

  # --- NEGATIVE CONTROL: a probe FAILURE fails OPEN (None), never a phrase ---
  def test_NEGCTL_probe_exception_is_none(self):
    p = _msg_probe({1: RuntimeError('rpc down')})
    self.assertIsNone(p.failure_message('1'))

  # --- NEGATIVE CONTROL: zero work units -> None (nothing to read) ---
  def test_NEGCTL_no_work_units_is_none(self):
    p = _msg_probe({1: _Exp([])})
    self.assertIsNone(p.failure_message('1'))


# ===========================================================================
# 4. END TO END through run_reconcile -- what the daemon calls. The (a)-(e).
# ===========================================================================
class RunReconcileTurnupRetryTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # A private history file so the global-brake read/write NEVER touches the
    # live daemon's ~/.tpu_reroute_history.json.
    self._hist = self.create_tempfile().full_path

  def _run(self, entries, *, status, msg=None, raise_msg=False,
           auto_retry_turnup=True, auto_resume_pruned=False,
           restart_evidence=None, dry_run=False, now=100.0, history=None):
    probe = _FakeProbe(status, msg, raise_msg=raise_msg)
    return RC.run_reconcile(
        entries, now=now, probe=probe, dry_run=dry_run,
        auto_resume_pruned=auto_resume_pruned,
        restart_evidence=restart_evidence,
        auto_retry_turnup=auto_retry_turnup,
        failure_probe=probe,
        history_file=history or self._hist)

  # ---- (a) turn-up error within budget -> re-queued QUEUED, resumes++ -------
  def test_a_turnup_within_budget_is_requeued(self):
    e = _entry(auto_resumes=0)
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG})
    self.assertEqual(e.state, R.JobState.FAILED)          # dead row retired
    new = [x for x in out if x.state == R.JobState.QUEUED]
    self.assertLen(new, 1)                                # exactly one retry
    self.assertEqual(new[0].auto_resumes, 1)              # SAME counter bumped
    self.assertNotIn('load_from', new[0].launch_kwargs)   # COLD: never trained
    self.assertNotIn('restart_from', new[0].launch_kwargs)
    self.assertIn('291448297', new[0].prior_xids)         # lineage preserved
    self.assertTrue(any('COLD retry queued' in l for l in log))

  def test_a_second_retry_bumps_the_counter_again(self):
    e = _entry(auto_resumes=1)
    out, _ = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                       msg={'291448297': _TURNUP_MSG})
    new = [x for x in out if x.state == R.JobState.QUEUED]
    self.assertLen(new, 1)
    self.assertEqual(new[0].auto_resumes, 2)

  # ---- (b) budget exhausted -> FAILED, no new row --------------------------
  def test_b_budget_exhausted_stays_failed(self):
    e = _entry(auto_resumes=3)
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG})
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    self.assertTrue(any('HOLD' in l and 'budget spent' in l for l in log))

  # ---- (c) unreadable / empty failure message -> FAILED (fail CLOSED) ------
  def test_c_empty_message_stays_failed(self):
    e = _entry()
    out, _ = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                       msg={})                              # probe returns None
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])

  def test_c_probe_that_raises_stays_failed(self):
    e = _entry()
    out, _ = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                       raise_msg=True)                      # probe throws
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])

  def test_c_no_failure_probe_supplied_stays_failed(self):
    # failure_probe=None is the fail-open default: behave exactly as today.
    e = _entry()
    probe = _FakeProbe({'291448297': RC.STATUS_TERMINAL})
    out, _ = RC.run_reconcile(
        [e], now=100.0, probe=probe, dry_run=False,
        auto_retry_turnup=True, failure_probe=None, history_file=self._hist)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])

  # ---- (d) genuine failures -> still FAILED (no regression) ----------------
  def test_d_code_bug_stays_failed_even_if_message_matches(self):
    # A real in-training crash whose message HAPPENS to contain a whitelisted
    # phrase must NOT be auto-retried: the code-bug guard holds it.
    e = _entry()
    ev = _FakeRestartEvidence({'291448297': ('segfault (SIGSEGV)', None, True)})
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG}, restart_evidence=ev)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    self.assertTrue(any('HOLD' in l and 'code bug' in l for l in log))

  def test_d_nonmatching_terminal_reason_stays_failed(self):
    e = _entry()
    out, log = self._run(
        [e], status={'291448297': RC.STATUS_TERMINAL},
        msg={'291448297': 'RESOURCE_EXHAUSTED: OOM killed'})
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    # A genuinely-unrelated failure is silent on the turn-up channel.
    self.assertFalse(any('turnup-retry' in l for l in log))

  # ---- (e) normal preemption -> still the EXISTING pruned/preempted path ---
  def test_e_preemption_takes_the_existing_warm_path_not_turnup(self):
    # A healthy run preempted after it trained, with a surviving checkpoint:
    # both flags on. The pruned path must claim it WARM; turn-up must not fire.
    e = _entry(last_reason='guarantee reclaim: preempted out of yutulpz')
    ev = _FakeRestartEvidence({'291448297': (None, _TORCH_CKPT, True)})
    out, log = self._run(
        [e], status={'291448297': RC.STATUS_TERMINAL},
        msg={'291448297': 'DESCHEDULED: resource-guarantee reclaim'},
        auto_resume_pruned=True, restart_evidence=ev)
    new = [x for x in out if x.state == R.JobState.QUEUED]
    self.assertLen(new, 1)
    self.assertEqual(new[0].launch_kwargs.get('load_from'), _TORCH_CKPT)  # WARM
    self.assertTrue(any('warm-restart queued' in l for l in log))
    self.assertFalse(any('turnup-retry' in l for l in log))  # turn-up skipped

  # ---- flag OFF: the default is byte-for-byte the old behaviour ------------
  def test_NEGCTL_flag_off_never_retries(self):
    e = _entry()
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG},
                         auto_retry_turnup=False)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    self.assertFalse(any('turnup-retry' in l for l in log))

  # ---- dry-run: log the plan, mutate NOTHING ------------------------------
  def test_dry_run_mutates_nothing(self):
    e = _entry()
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG}, dry_run=True)
    self.assertEqual(e.state, R.JobState.SUBMITTED)        # untouched
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    self.assertTrue(any('[DRY][turnup-retry]' in l for l in log))

  # ---- global brake: a fleet-wide churn suspends turn-up retries too -------
  def test_global_brake_leaves_row_failed(self):
    # REAL time: _load_reroute_history filters by time.time() over a 3600s
    # window, so seeded timestamps must be recent (mirrors reroutebrake_test's
    # test_tripped_brake_leaves_the_candidate_alone). In production now ==
    # time.time(), so this is a fixture detail, not a behaviour change.
    now = _time.time()
    e = _entry()
    # Seed the private history at the brake threshold.
    hist = [now - i for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR + 1)]
    RC._save_reroute_history(hist, self._hist)
    out, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                         msg={'291448297': _TURNUP_MSG}, now=now)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual([x for x in out if x.state == R.JobState.QUEUED], [])
    self.assertTrue(any('GLOBAL BRAKE' in l for l in log))

  def test_live_retry_records_a_reroute_in_history(self):
    # A successful turn-up retry is counted toward the SAME global brake, so a
    # systematically-broken cell cannot loop the fleet. REAL time so the
    # appended stamp survives _load_reroute_history's window filter on readback.
    now = _time.time()
    e = _entry()
    before = len(RC._load_reroute_history(self._hist))
    self._run([e], status={'291448297': RC.STATUS_TERMINAL},
              msg={'291448297': _TURNUP_MSG}, now=now)
    after = len(RC._load_reroute_history(self._hist))
    self.assertEqual(after, before + 1)

  # ---- the summary line reflects the retry ---------------------------------
  def test_summary_counts_the_retry(self):
    e = _entry()
    _, log = self._run([e], status={'291448297': RC.STATUS_TERMINAL},
                       msg={'291448297': _TURNUP_MSG})
    summary = [l for l in log if 'checked:' in l]
    self.assertIn('1 auto restart(s) queued', summary[0])


if __name__ == '__main__':
  googletest.main()
