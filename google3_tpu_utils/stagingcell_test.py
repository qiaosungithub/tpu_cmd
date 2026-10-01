"""Staging/test 'twin' cells are NEVER a valid PROD placement.

WHY THIS EXISTS. Borg STAGING/TEST twin cells -- names matching
`-c(-staging|-test)?$`, e.g. `yucbfad-c-staging` -- show FREE CHIPS in
GetCellAvailability because nobody competes for them, so the placement path
finds them the roomiest and most attractive candidate around. But they are not
configured to run PROD workloads: every PROD job placed on one fails 100% at
Borg turn-up with `INVALID_ARGUMENT: Try to create job`. Measured live: across
one b200 sweep, 13/13 submissions that landed on `yucbfad-c-staging` FAILED at
turn-up while 21/21 on the real cell `sj` were healthy; the reroute path flung
jobs onto the staging cell after cooling their source cell.

The market/pricing layer already excludes these cells unconditionally
(`remeasure_cell_locality.market_cells`), but the availability -> placement path
did not, so they leaked into the router's candidate pool. The fix is a hard,
unconditional filter in `avail_provider.build_availability` -- the single choke
point that produces `avail_by_cell` for every placement path (pick_cell, the
reroute/tick in route_check, the queue-status planner). These tests pin that:

  * `build_availability` drops the twin cell from BOTH `avail_by_cell` and the
    arch-pool magnitude;
  * the placement path (`best_cell_for_shape`, `plan_one`) then picks the normal
    cell even though the twin advertised far more free chips;
  * NEGATIVE CONTROL: fed the twin cell DIRECTLY (bypassing the filter), the
    placement path WOULD select it -- so the filter is load-bearing and these
    tests can actually fail;
  * the regex matches the market layer's exactly (`-c`, `-c-staging`, `-c-test`
    tails only), and does not over-match a legitimate cell.
"""

from typing import Optional, TypeVar

from google3.experimental.users.qiaos.tpu_utils import avail_provider as AP
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest as absltest

_T = TypeVar('_T')

_NOW = 1e9


def _ok(x: Optional[_T]) -> _T:
  """Assert a router result is not None and return it type-narrowed.

  best_cell_for_shape / plan_one return Optional; a test that then reads `.cell`
  or indexes the tuple both asserts placement happened AND satisfies the strict
  type checker (mirrors route_lib_test._ok)."""
  assert x is not None, 'expected a placement, got None'
  return x


def _entry(arch='b200', power='b200-8', **kw):
  e = R.QueueEntry(job_id='j', power=power, allowed_archs=[arch], **kw)
  e.state = R.JobState.QUEUED
  return e


class StagingCellExcludedFromAvailabilityTest(absltest.TestCase):
  """The filter lives in build_availability, the sole producer of avail_by_cell."""

  def test_twin_cell_absent_from_avail_by_cell(self):
    # The twin advertises 512 free chips; the real cell only 16. Roominess is
    # exactly what would make the twin win if it were a candidate at all.
    per_arch = {'b200': {'yucbfad-c-staging': (512, False), 'sj': (16, False)}}
    avail, _, pool = AP.build_availability(per_arch, {'b200': 5.0})
    self.assertEqual(sorted(avail), ['sj|b200'],
                     'the staging twin must never enter the candidate pool')
    self.assertNotIn('yucbfad-c-staging|b200', avail)

  def test_twin_cell_chips_excluded_from_pool_magnitude(self):
    # The pool magnitude drives the pool-size price discount; twin chips are
    # phantom capacity and must not inflate it.
    per_arch = {'b200': {'yucbfad-c-staging': (512, False), 'sj': (16, False)}}
    _, _, pool = AP.build_availability(per_arch, {'b200': 5.0})
    self.assertEqual(pool['b200'], 16.0, 'twin chips must not count toward the pool')

  def test_c_test_and_bare_c_twins_also_excluded(self):
    # The regex covers the '-c', '-c-staging' and '-c-test' tails alike, exactly
    # as the market layer does.
    per_arch = {'v6e': {
        'yufoo-c-staging': (99, False),
        'yufoo-c-test': (99, False),
        'yufoo-c': (99, False),
        'sj': (8, False),
    }}
    avail, _, pool = AP.build_availability(per_arch, {'v6e': 5.0})
    self.assertEqual(sorted(avail), ['sj|v6e'])
    self.assertEqual(pool['v6e'], 8.0)

  def test_legitimate_cell_not_over_matched(self):
    # A production cell whose name merely CONTAINS 'staging'/'test' in the
    # middle, or has a suffix past the tail, must survive -- the market layer
    # anchors the match at end-of-string ($) and so do we.
    per_arch = {'v6e': {
        'yucbfad-c-staging-x': (4, False),   # suffix past the tail
        'yutulpz': (4, False),               # ordinary cell
    }}
    avail, _, _ = AP.build_availability(per_arch, {'v6e': 5.0})
    self.assertEqual(sorted(avail),
                     ['yucbfad-c-staging-x|v6e', 'yutulpz|v6e'])

  def test_is_nonprod_cell_helper(self):
    for cell in ('yucbfad-c-staging', 'yufoo-c-test', 'yubar-c'):
      self.assertTrue(AP.is_nonprod_cell(cell), cell)
    for cell in ('sj', 'yutulpz', 'je', 'yucbfad-c-staging-x'):
      self.assertFalse(AP.is_nonprod_cell(cell), cell)


class StagingCellNeverPlacedTest(absltest.TestCase):
  """End to end: the twin cell is never chosen even when it looks best."""

  def test_normal_cell_wins_over_roomier_twin(self):
    # Full path: build_availability -> best_cell_for_shape. The twin has 32x the
    # free chips AND a cheaper price; the ONLY reason it must lose is the filter.
    per_arch = {'b200': {'yucbfad-c-staging': (512, False), 'sj': (16, False)}}
    cell_price = {'b200': {'yucbfad-c-staging': 1.0, 'sj': 15.0}}
    avail, _, _ = AP.build_availability(per_arch, {'b200': 15.0}, cell_price)
    hit = _ok(R.best_cell_for_shape('b200', 8, _entry(), avail, now=_NOW))
    self.assertEqual(hit[0].cell, 'sj', 'the real cell sj must still place')

  def test_plan_one_never_returns_a_twin(self):
    # The top-level per-job planner, same result.
    per_arch = {'b200': {'yucbfad-c-staging': (512, False), 'sj': (16, False)}}
    avail, arch_price, arch_pool = AP.build_availability(
        per_arch, {'b200': 5.0})
    p = _ok(R.plan_one(_entry(), avail, _NOW, arch_price=arch_price,
                       arch_pool=arch_pool))
    self.assertEqual(p.cell, 'sj', 'the job must place on the real cell')

  def test_twin_only_pool_places_nothing(self):
    # If the twin is the ONLY cell with capacity, the correct answer is "nothing
    # placeable" (job stays QUEUED), NOT "submit to the twin and fail turn-up".
    per_arch = {'b200': {'yucbfad-c-staging': (512, False)}}
    avail, _, _ = AP.build_availability(per_arch, {'b200': 5.0})
    self.assertEqual(avail, {})
    self.assertIsNone(R.best_cell_for_shape('b200', 8, _entry(), avail, now=_NOW))
    self.assertIsNone(R.plan_one(_entry(), avail, _NOW))

  def test_negctl_unfiltered_twin_would_be_chosen(self):
    """NEGATIVE CONTROL: proves the filter is what does the work.

    Fed the twin cell DIRECTLY into avail_by_cell (bypassing build_availability),
    the placement path has no built-in staging guard and picks it -- it is
    roomier and cheaper. So the exclusion genuinely comes from the filter, and
    the positive tests above would fail without it (a test that cannot fail
    proves nothing)."""
    avail = {
        'yucbfad-c-staging|b200': R.CellAvail(
            cell='yucbfad-c-staging', arch='b200', free_chips=512,
            oversold=False, price=1.0, metro='cbf'),
        'sj|b200': R.CellAvail(
            cell='sj', arch='b200', free_chips=16, oversold=False,
            price=15.0, metro='sin'),
    }
    hit = _ok(R.best_cell_for_shape('b200', 8, _entry(), avail, now=_NOW))
    self.assertEqual(hit[0].cell, 'yucbfad-c-staging',
                     'without the filter the placement path DOES pick the twin')


if __name__ == '__main__':
  absltest.main()
