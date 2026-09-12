"""Self-asserting checks for `infra_check.classify_failure_reason`'s taxonomy.

Same shape as preflight/*_test.py: a plain binary that sys.exit(1)s on failure,
declared as pytype_strict_contrib_test so `blaze test` actually runs it.

Why this file exists: the WHY column's whole job is to answer "is it my code or
is it Borg". Every rule used to be an infra verdict, so an application crash
fell through to the raw-message branch and printed Borg's job-name prefix --
the column literally read `qiaos_group_275707651` as the cause of that job's
own segfault. The regression risk now runs BOTH ways, so both are asserted:
application failures must be labelled CODE BUG, and infra failures must NOT be.
"""

import sys

from google3.experimental.users.qiaos.tpu_utils import infra_check


_FAILURES: list[str] = []


def check(label: str, actual, expected) -> None:
  if actual != expected:
    _FAILURES.append(f'{label}: expected {expected!r}, got {actual!r}')
    print(f'FAIL {label}: expected {expected!r}, got {actual!r}')
  else:
    print(f'ok   {label}')


def check_contains(label: str, message: str, needle: str | None) -> None:
  """`needle=None` asserts the message is NOT classified as an application error."""
  got = infra_check._application_error(message.upper())  # pylint: disable=protected-access
  if needle is None:
    if got is not None:
      _FAILURES.append(f'{label}: infra message misclassified as {got!r}')
      print(f'FAIL {label}: infra message misclassified as {got!r}')
    else:
      print(f'ok   {label} (correctly not an application error)')
    return
  if got is None or needle.lower() not in got.lower():
    _FAILURES.append(f'{label}: expected a verdict containing {needle!r}, got {got!r}')
    print(f'FAIL {label}: expected {needle!r}, got {got!r}')
  else:
    print(f'ok   {label} -> {got}')


# --- Application failures, verbatim from real status.message payloads. -------

check_contains(
    'segfault (XID 275707651)',
    'qiaos_group_275707651.1.main/qiaos: A task received a segmentation fault. '
    'This is usually an application level error, check the logs for more '
    'details.\n\nInternal message: qiaos_group_275707651.1.main/qiaos: '
    'Killed by signal 11!',
    'segfault')

check_contains(
    'stdlib mkdir on /cns (XID 275703523)',
    "qiaos_group_275703523.1.main/qiaos: Unrecoverable failure "
    "(go/task-retry-blocking): PermissionError: [Errno 13] "
    "Permission denied: '/cns'",
    '/cns')

check_contains(
    'missing wandb attribute (XID 275697405)',
    "AttributeError: module 'wandb' has no attribute 'util'",
    'AttributeError')

check_contains('oom', 'Task exited: out of memory', 'out of memory')
check_contains('hbm oom', 'RESOURCE_EXHAUSTED: OOM when allocating', 'out of memory')
check_contains('missing module', "ModuleNotFoundError: No module named 'clu'",
               'missing module')
check_contains('sigabrt', 'Killed by signal 6!', 'abort')
check_contains('traceback', 'Traceback (most recent call last):\n  File "x"',
               'unhandled Python exception')

# --- Infra failures must NOT be labelled CODE BUG. ---------------------------
# These are the regressions that would matter most: mislabelling a preemption
# as a code bug sends someone to read a traceback that does not exist.

check_contains('preemption', 'Preempted by higher priority workload', None)
check_contains('descheduled',
               'Workload was DESCHEDULED due to guaranteed capacity reclaim', None)
check_contains('gqm price', 'GQM_RESOURCE_DEFICIT_INFO: waiting for price', None)
check_contains('capacity', 'Pool capacity exhausted', None)
check_contains('defrag', 'Preempted (SLICE_DEFRAGMENTATION)', None)

# --- A price cap and an empty auction are DIFFERENT verdicts. ----------------
# These three messages used to collapse into one string, "Queued (GQM price
# over limit order)", which pointed at the wrong lever for two of them: a real
# v6p-64 probe was reported as price-capped while its group had no row in the
# cap table at all and the market cleared 10x below the cap. The fix a reader
# takes differs in each case, so the verdict has to differ too.


def check_reason(label: str, message: str, expected: str) -> None:
  """Assert the full derive_failure_reason verdict for a work-unit message."""

  class _Status:

    def __init__(self, text: str):
      self.message = text

  class _WU:

    def __init__(self, text: str):
      self.status = _Status(text)

  check(label, infra_check.derive_failure_reason('xid', _WU(message), {}),
        expected)


check_reason(
    'a real price cap says so',
    'Workload paused: market price above LIMIT ORDER for this experiment',
    'Queued (GQM price over limit order)')
check_reason(
    'an auction shortfall is NOT a price cap',
    'GQM_RESOURCE_DEFICIT_INFO ... deficit: tier HighlyAvailable '
    '{ GHOSTFISH=19.00 } in cell yucbfiv',
    'Queued (GQM auction short of chips)')
check_reason(
    'an oversold cell names the cell as the fix',
    'Your workload can afford GHOSTFISH on a global market, but in cell '
    'yutulpz the demand from admitted jobs exceeds the available cell supply. '
    'See GQM_OVERSOLD_MARKET.',
    'Queued (cell oversold; try another cell)')

# --- The job-name prefix must be stripped from unrecognised messages. --------

check('strip job prefix',
      infra_check._strip_job_prefix(  # pylint: disable=protected-access
          'qiaos_group_275707651.1.main/qiaos: Killed by signal 11!'),
      'Killed by signal 11!')
check('leave a normal message alone',
      infra_check._strip_job_prefix('Pool capacity exhausted'),  # pylint: disable=protected-access
      'Pool capacity exhausted')
check('leave a message with no prefix alone',
      infra_check._strip_job_prefix('Queued, no reason reported'),  # pylint: disable=protected-access
      'Queued, no reason reported')

# --- `tpu clear` archive behaviour: board + local-queue cascade. -------------
# Exercise the real _clear_jobs against temp board/legacy/queue files (not a
# mock): the whole point of the change is that the two stores move together, and
# that a LIVE queue row is never dropped.

import json as _json           # pylint: disable=g-import-not-at-top
import os as _os               # pylint: disable=g-import-not-at-top
import tempfile as _tempfile   # pylint: disable=g-import-not-at-top

from google3.experimental.users.qiaos.tpu_utils import route_check as _RC  # pylint: disable=g-import-not-at-top
from google3.experimental.users.qiaos.tpu_utils import route_lib as _RL    # pylint: disable=g-import-not-at-top

_NO_MAP = '/no/such/mapping/dir'


def _clear_setup(board=None, queue_rows=None):
  """Point infra_check at fresh temp board/legacy files and a temp queue."""
  bd = _tempfile.mkstemp(suffix='.jobs.json')[1]
  lg = _tempfile.mkstemp(suffix='.legacy.json')[1]
  q = _tempfile.mkstemp(suffix='.queue.json')[1]
  with open(bd, 'w') as f:
    _json.dump(board or {}, f)
  with open(lg, 'w') as f:
    _json.dump({}, f)
  infra_check._JOBS_FILE = bd     # pylint: disable=protected-access
  infra_check._LEGACY_FILE = lg   # pylint: disable=protected-access
  _RC.save_queue(q, queue_rows or [])
  _os.environ['TPU_LOCAL_QUEUE_FILE'] = q
  return bd, lg, q


def _qrow(job_id, state, xid):
  e = _RL.QueueEntry(job_id=job_id, power='v7-32', allowed_archs=['v7'])
  e.state = state
  e.xid = xid
  return e


# `tpu clear all` must be REFUSED -- a blanket sweep is the one un-undoable
# mistake this command must not make easy.
bd, lg, q = _clear_setup(board={'111': {'name': 'r1'}},
                         queue_rows=[_qrow('j', _RL.JobState.DONE, '111')])
infra_check._clear_jobs(['all'], _NO_MAP)  # pylint: disable=protected-access
check('clear all leaves the board untouched', '111' in _json.load(open(bd)), True)
check('clear all leaves the queue untouched', len(_RC.load_queue(q)), 1)

# A FINISHED run: board entry AND queue row archived together, into one legacy
# record carrying both views.
bd, lg, q = _clear_setup(board={'111': {'name': 'r1'}},
                         queue_rows=[_qrow('j1', _RL.JobState.DONE, '111')])
infra_check._clear_jobs(['111'], _NO_MAP)  # pylint: disable=protected-access
_live = _json.load(open(bd))
_leg = _json.load(open(lg))
check('finished: board entry removed', '111' in _live, False)
check('finished: archived to legacy', '111' in _leg, True)
check('finished: legacy record folds in the queue row',
      'queue_row' in _leg.get('111', {}), True)
check('finished: queue row removed', len(_RC.load_queue(q)), 0)

# A LIVE run (RUNNING queue row): the board is still tidied, but the queue row is
# REFUSED -- dropping it would strand a job on the cluster.
bd, lg, q = _clear_setup(board={'222': {'name': 'r2'}},
                         queue_rows=[_qrow('j2', _RL.JobState.RUNNING, '222')])
infra_check._clear_jobs(['222'], _NO_MAP)  # pylint: disable=protected-access
check('live: board entry still archived', '222' in _json.load(open(bd)), False)
check('live: queue row NOT removed', len(_RC.load_queue(q)), 1)
check('live: refused row not folded as queue_row',
      'queue_row' in _json.load(open(lg)).get('222', {}), False)

# A board-ONLY XID (no queue row at all -- the common case): archived cleanly,
# no error, nothing reported missing.
bd, lg, q = _clear_setup(board={'333': {'name': 'r3'}}, queue_rows=[])
infra_check._clear_jobs(['333'], _NO_MAP)  # pylint: disable=protected-access
check('board-only: archived off the board',
      '333' not in _json.load(open(bd)), True)
check('board-only: present in legacy', '333' in _json.load(open(lg)), True)

if _FAILURES:
  print(f'\n{len(_FAILURES)} check(s) FAILED')
  sys.exit(1)
print('\nAll checks passed.')
