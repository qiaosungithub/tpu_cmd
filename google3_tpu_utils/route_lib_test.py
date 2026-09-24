"""Unit tests for the pure scheduling core. No I/O, no RPC, no real clock."""

import random
from typing import Optional, TypeVar
import unittest
from unittest import mock

from google3.experimental.users.qiaos.tpu_utils import route_lib as R

_T = TypeVar('_T')


def _ok(x: Optional[_T]) -> _T:
  """Assert a router result is not None and return it type-narrowed.

  plan_one/best_cell return Optional; a test that then reads `.cell` both
  asserts placement happened AND satisfies the strict type checker."""
  assert x is not None, 'expected a placement, got None'
  return x


def _avail(cell, arch, free, oversold=False, price=8.0, metro=''):
  return R.CellAvail(cell=cell, arch=arch, free_chips=free, oversold=oversold,
                     price=price, metro=metro or cell)


def _entry(job_id='j1', power='v7-32', archs=('v7',), **kw):
  return R.QueueEntry(job_id=job_id, power=power, allowed_archs=list(archs), **kw)


class PowerTest(unittest.TestCase):

  def test_parse_power(self):
    self.assertEqual(R.parse_power('v5p-32'), 32.0)
    self.assertEqual(R.parse_power('v7-32'), 4.34 * 32)
    self.assertEqual(R.parse_power('64'), 64.0)

  def test_candidate_shapes_orders_newer_first(self):
    # power v6p-32 == 138.88 v5p-eq; v7-32 == same. Both accepted, v7 first.
    e = _entry(power='v6p-32', archs=('v6p', 'v7'))
    shapes = R.candidate_shapes(e)
    self.assertEqual(shapes[0][0], 'v7')       # ARCH_PREF puts v7 ahead
    self.assertIn(('v6p', 32), shapes)

  def test_candidate_shapes_respects_allowed_archs(self):
    e = _entry(power='v7-32', archs=('v6p',))   # only v6p allowed
    shapes = R.candidate_shapes(e)
    self.assertTrue(all(a == 'v6p' for a, _ in shapes))


class GpuCandidateShapesTest(unittest.TestCase):
  """★The operator's rule: a GPU power spec names a fixed-width BOARD, not a
  compute budget. The chip count is preserved verbatim; the only substitution
  allowed is a different CARD at the SAME width, biggest-first. This is the
  regression guard for xid 288485310, which asked for h100-8, allowed b200, and
  was handed b200-4 by the TPU compute-equivalence window."""

  def test_h100_8_allowing_b200_stays_8_chips_biggest_first(self):
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    self.assertEqual(R.candidate_shapes(e), [('b200', 8), ('h100', 8)])

  def test_h100_8_only_h100_no_substitution(self):
    e = _entry(power='h100-8', archs=('h100',))
    self.assertEqual(R.candidate_shapes(e), [('h100', 8)])

  def test_b200_8_kept_at_8_not_rescaled(self):
    e = _entry(power='b200-8', archs=('b200', 'h100'))
    self.assertEqual(R.candidate_shapes(e), [('b200', 8), ('h100', 8)])

  def test_gpu_never_substitutes_a_tpu_even_if_allowed(self):
    # A GPU ask lists a TPU arch too: the TPU must NOT appear -- a CUDA binary
    # power-matched onto a v5p slice does not run. Only GPU archs at the exact
    # width are emitted.
    e = _entry(power='h100-8', archs=('h100', 'b200', 'v5p'))
    shapes = R.candidate_shapes(e)
    self.assertTrue(all(R.is_gpu(a) for a, _ in shapes), shapes)
    self.assertEqual(shapes, [('b200', 8), ('h100', 8)])

  def test_gpu_width_with_no_matching_legal_size_yields_nothing(self):
    # h100 has no legal 16-chip board (NVLink domain caps at 8), so an ask for
    # h100-16 with only h100 allowed is unplaceable rather than downsized.
    e = _entry(power='h100-16', archs=('h100',))
    self.assertEqual(R.candidate_shapes(e), [])

  def test_gpu_power_tolerance_is_ignored(self):
    # Even a wide tolerance must not pull in a different chip count: the board
    # width is exact, not a compute window.
    e = _entry(power='b200-8', archs=('b200',), power_tolerance=0.9)
    self.assertEqual(R.candidate_shapes(e), [('b200', 8)])

  def test_parse_gpu_shape_and_is_gpu_helpers(self):
    self.assertEqual(R.parse_gpu_shape('h100-8'), ('h100', 8))
    self.assertEqual(R.parse_gpu_shape('b200-4'), ('b200', 4))
    self.assertIsNone(R.parse_gpu_shape('v5p-32'))   # a TPU is not a GPU shape
    self.assertIsNone(R.parse_gpu_shape('64'))        # bare int names no board
    self.assertTrue(R.is_gpu('B200'))                 # case-insensitive
    self.assertFalse(R.is_gpu('v7'))


class PlacementTest(unittest.TestCase):

  def test_oversold_cell_is_skipped(self):
    e = _entry()
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=5, oversold=True),   # THE bug cell
        'yukulwh': _avail('yukulwh', 'v7', free=3244, oversold=False),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertIsNotNone(p)
    self.assertEqual(p.cell, 'yukulwh')         # NOT the oversold one

  def test_fragmentation_zero_slices_skipped(self):
    # free chips present but < one slice (32): unplaceable.
    e = _entry()
    avail = {'c1': _avail('c1', 'v7', free=31)}   # 31 < 32 -> 0 slices
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_obtainable_does_not_help_only_free_counts(self):
    # We never pass obtainable in; a cell with 0 free is unplaceable regardless.
    e = _entry()
    avail = {'c1': _avail('c1', 'v7', free=0)}
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_price_cap_excludes_expensive_cell(self):
    e = _entry(max_price=20.0)
    avail = {
        'pricey': _avail('pricey', 'v7', free=3200, price=25.0),   # over cap
        'cheap': _avail('cheap', 'v7', free=64, price=15.0),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'cheap')

  def test_metro_filter(self):
    e = _entry(allowed_metros=['cbf'])
    avail = {
        'yukulwh': _avail('yukulwh', 'v7', free=3244, metro='kul'),
        'yucbfiv': _avail('yucbfiv', 'v7', free=64, metro='cbf'),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'yucbfiv')

  def test_rank_trades_slices_against_price(self):
    """Capacity and price are weighed together; capacity does not simply win.

    ★This asserted "slices dominate price" until 2026-09-01, encoding the old
    lexicographic sort (-n_slices, price, -free_chips). Under that order price
    was never reached in practice, because per-cell prices all collapsed to one
    global value -- so the router would pay 2.5x to fit one more slice. cell_score
    divides price by a BOUNDED slice_weight, so a 2.5x price gap outweighs the
    capacity bonus. The negative control below keeps 'cheapest always wins' from
    passing as well.
    """
    e = _entry()
    avail = {
        'few': _avail('few', 'v7', free=64, price=10.0),      # 2 slices, cheap
        'many': _avail('many', 'v7', free=3200, price=25.0),  # 100 slices, 2.5x
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'few')      # 2.5x price beats the capped bonus

  def test_rank_prefers_capacity_when_price_is_close(self):
    """Negative control: with prices near-equal, the roomier cell must win.

    Without this, the test above would also pass if slice_weight were ignored
    entirely and the router had silently become 'always pick the cheapest'.
    """
    e = _entry()
    avail = {
        'few': _avail('few', 'v7', free=64, price=10.0),
        'many': _avail('many', 'v7', free=3200, price=10.5),  # 5% dearer
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'many')

  def test_tpu_type_fallback_when_first_arch_unavailable(self):
    # v7 allowed first but no v7 anywhere; v6p available -> falls back.
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    avail = {'c1': _avail('c1', 'v6p', free=320)}   # v6p-32 fits (138.88 pw)
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.arch, 'v6p')
    self.assertEqual(p.chips, 32)

  def test_cooldown_cell_downweighted_not_excluded(self):
    """A cooling cell is penalised, not removed from the candidate set.

    ★This test asserted the opposite until 2026-09-01: that a cell on cooldown
    is SKIPPED. That hard exclusion was the bug -- a mechanism whose whole job
    is "every 10 minutes, pick the best cell again" deleted the cell it had
    just used, and returned None when every candidate was cooling, so nothing
    could be placed at all. The replacement multiplies cell_score by
    cooldown_penalty(), which decays to 1.0 across the window.
    Consequence, asserted here: a much better cell still wins WHILE cooling.
    """
    e = _entry()
    e.cooldown_pairs = {'hot|v7': {'until': 100.0, 'strikes': 1}}
    avail = {
        'hot': _avail('hot', 'v7', free=3200),
        'cool': _avail('cool', 'v7', free=64),
    }
    # 50x the capacity beats a penalty bounded by COOLDOWN_WEIGHT.
    self.assertEqual(_ok(R.plan_one(e, avail, now=50.0)).cell, 'hot')
    # ...and it still wins once the cooldown has expired.
    self.assertEqual(_ok(R.plan_one(e, avail, now=150.0)).cell, 'hot')

  def test_cooldown_penalty_flips_a_close_call(self):
    """The penalty must actually change an outcome, or it is decoration.

    Negative control for the test above: with two cells of EQUAL standing, the
    one on cooldown must lose. Without this, 'downweighted' could mean 'weight
    ignored' and both tests would still pass.
    """
    e = _entry()
    e.cooldown_pairs = {'hot|v7': {'until': 100.0, 'strikes': 1}}
    avail = {
        'hot': _avail('hot', 'v7', free=64),
        'cool': _avail('cool', 'v7', free=64),
    }
    self.assertEqual(_ok(R.plan_one(e, avail, now=50.0)).cell, 'cool')


class BatchSchedulingTest(unittest.TestCase):

  def test_priority_high_first(self):
    lo = _entry('lo', priority=0)
    hi = _entry('hi', priority=10)
    avail = {'c': _avail('c', 'v7', free=32)}   # exactly ONE slice
    plcs = R.select_and_plan([lo, hi], avail, now=0.0, rng=random.Random(0))
    self.assertEqual(len(plcs), 1)
    self.assertEqual(plcs[0].job_id, 'hi')      # high priority took the one slice

  def test_equal_priority_random_but_fair(self):
    # two equal-priority jobs, one slice: which one wins should vary with seed.
    a = _entry('a', priority=5)
    b = _entry('b', priority=5)
    avail = {'c': _avail('c', 'v7', free=32)}
    winners = set()
    for seed in range(20):
      plcs = R.select_and_plan([a, b], avail, now=0.0, rng=random.Random(seed))
      winners.add(plcs[0].job_id)
    self.assertEqual(winners, {'a', 'b'})       # both win under some seed

  def test_thundering_herd_drawdown(self):
    # one cell with exactly 2 slices, three jobs -> only 2 placed, and NOT all
    # into the same cell beyond capacity.
    jobs = [_entry(f'j{i}') for i in range(3)]
    avail = {'c': _avail('c', 'v7', free=64)}    # 2 slices
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(1))
    self.assertEqual(len(plcs), 2)               # third stays queued

  def test_drawdown_spreads_across_cells(self):
    jobs = [_entry(f'j{i}') for i in range(2)]
    avail = {
        'a': _avail('a', 'v7', free=32),   # 1 slice
        'b': _avail('b', 'v7', free=32),   # 1 slice
    }
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(2))
    self.assertEqual(len(plcs), 2)
    self.assertEqual({p.cell for p in plcs}, {'a', 'b'})  # one each, not both on one

  def test_drawdown_composite_key_dual_arch_cell(self):
    # A cell keyed per (cell, arch) as 'cell|arch' -- the shape the live
    # provider emits for a dual-generation cell like `je` (v6e + v7). The
    # decrement must find the right entry BY CONTENT, not by assuming the key
    # is the bare cell name, or the second job sees stale free chips.
    jobs = [_entry(f'j{i}', power='v7-32', archs=('v7',)) for i in range(3)]
    avail = {
        'je|v6e': _avail('je', 'v6e', free=999),   # same cell, other gen
        'je|v7': _avail('je', 'v7', free=64),      # 2 v7 slices only
    }
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(3))
    self.assertEqual(len(plcs), 2)                 # v7 drawn down to 0, third waits
    self.assertTrue(all(p.arch == 'v7' and p.cell == 'je' for p in plcs))


class RerouteTest(unittest.TestCase):

  def test_needs_reroute_timing(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 1000.0
    self.assertFalse(R.needs_reroute(e, now=1000.0 + 599, reroute_after_s=600))
    self.assertTrue(R.needs_reroute(e, now=1000.0 + 600, reroute_after_s=600))

  def test_needs_reroute_only_for_submitted(self):
    e = _entry()
    e.state = R.JobState.QUEUED
    e.submitted_at = 0.0
    self.assertFalse(R.needs_reroute(e, now=1e9, reroute_after_s=600))

  def test_reroute_deadline_has_no_exponential_backoff(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 1000.0
    for r in (0, 1, 4, 10):
      e.reroutes = r
      self.assertEqual(R.reroute_deadline_s(e, base_s=300.0), 300.0)
      self.assertTrue(R.needs_reroute(e, now=1300.0, reroute_after_s=300.0))

  def test_mark_reroute_sets_cooldown_and_requeues(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell = 'yulpptr'
    e.submitted_at = 0.0
    e.xid = '12345'
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertIsNone(e.xid)
    # e.arch was never set, so the cell is blamed under every accepted arch.
    self.assertEqual(e.cooldown_pairs,
                     {'yulpptr|v7': {'until': 700.0 + 1800.0, 'strikes': 1}})
    # ★A re-route is NOT a build failure: `attempts` feeds the 3-strikes brake
    # that parks a row HELD, so bumping it here parked healthy jobs that the
    # router had merely moved between oversold cells (infra-v17, measured on
    # elt's cars: attempts=3 with zero real build failures). Re-routes are
    # counted separately, and the old XID is preserved for the audit.
    self.assertEqual(e.attempts, 0)
    self.assertEqual(e.reroutes, 1)
    self.assertEqual(e.prior_xids, ['12345'])

  def test_repeated_reroutes_never_trip_the_build_brake(self):
    # Regression: an oversold-cell rotation must not park a healthy job.
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 0.0
    for i in range(10):
      e.cell = f'cell{i}'
      e.xid = str(1000 + i)
      R.mark_reroute(e, now=700.0 + i, cooldown_s=1800.0)
    self.assertEqual(e.attempts, 0)
    self.assertEqual(e.reroutes, 10)
    self.assertEqual(len(e.prior_xids), 10)

  def test_reroute_then_replan_avoids_hot_cell(self):
    # end-to-end: job stuck in yulpptr, re-routed, next plan picks another cell.
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell = 'yulpptr'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=3200),   # now looks free but on cooldown
        'yukulwh': _avail('yukulwh', 'v7', free=3200),
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    self.assertEqual(p.cell, 'yukulwh')      # avoided the cooled-down cell

  def test_reroute_falls_through_to_next_arch_when_top_arch_cell_cooling(self):
    """The b200->b200 loop (parcae, 2026-09-09).

    A multi-arch GPU job (power h100-8, archs=[h100,b200]) whose TOP-preferred
    arch (b200, biggest-card-first) has exactly ONE usable cell in the allowed
    metros. It gets stuck there, re-routes -> that cell is cooled. But cooldown
    is a soft cell_score penalty, not a gate, and with only one b200 cell there
    is nothing to reorder, so plan_one used to hand the job straight back to the
    same cell every pass -- never trying h100, which had free capacity in a
    DIFFERENT cell. The fix: a shape that resolves only to a still-cooling cell
    is a fallback; keep scanning later archs first.
    """
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    e.state = R.JobState.SUBMITTED
    e.cell = 'sj'
    e.arch = 'b200'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)   # cools sj|b200
    # Prices under each family's limit-order cap (h100<=10, b200<=20) so the
    # price-cap gate is not what decides this test -- the cooldown fallthrough is.
    avail = {
        'sj': _avail('sj', 'b200', free=1024, price=8.0),  # only b200 cell, cooling
        'sh': _avail('sh', 'h100', free=1024, price=8.0),  # h100 free elsewhere
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    self.assertEqual(p.arch, 'h100')     # fell through to the next arch
    self.assertEqual(p.cell, 'sh')       # NOT back to the cooled b200 cell

  def test_reroute_uses_cooled_fallback_when_every_arch_is_cooling(self):
    """Negative control: if EVERY arch resolves only to a cooling cell, the job
    still gets placed (going back is no worse than the pre-fix behaviour), not
    left unplaced."""
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    e.state = R.JobState.SUBMITTED
    e.cell = 'sj'
    e.arch = 'b200'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)   # cools sj|b200
    e.cooldown_pairs['sh|h100'] = {'until': 2500.0, 'strikes': 1}  # h100 too
    avail = {
        'sj': _avail('sj', 'b200', free=1024, price=8.0),
        'sh': _avail('sh', 'h100', free=1024, price=8.0),
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    # b200 is the top arch, so its cooled cell is the first fallback recorded.
    self.assertEqual(p.cell, 'sj')
  # --- hardening pure logic (2026-08-24) ---
  def test_output_is_fresh_within_window(self):
    self.assertTrue(R.output_is_fresh(latest_mtime=640.0, now=700.0,
                                      fresh_within_s=1200.0))   # 60s ago

  def test_output_is_fresh_none_means_no_evidence(self):
    self.assertFalse(R.output_is_fresh(latest_mtime=None, now=700.0,
                                       fresh_within_s=1200.0))  # missing = not alive

  def test_output_is_fresh_boundary_is_stale(self):
    # EXACTLY fresh_within_s ago counts as stale, so the window can never make
    # reroute a permanent no-op.
    self.assertFalse(R.output_is_fresh(latest_mtime=800.0, now=2000.0,
                                       fresh_within_s=1200.0))  # 1200s ago == boundary
    self.assertTrue(R.output_is_fresh(latest_mtime=801.0, now=2000.0,
                                      fresh_within_s=1200.0))   # 1199s ago < window

  def test_decide_reroute_both_pending_no_output_reroutes(self):
    self.assertTrue(R.decide_reroute('PENDING', 'PENDING', output_fresh=False))

  def test_decide_reroute_fresh_output_blocks(self):
    self.assertFalse(R.decide_reroute('PENDING', 'PENDING', output_fresh=True))

  def test_decide_reroute_second_running_blocks(self):
    self.assertFalse(R.decide_reroute('PENDING', 'RUNNING', output_fresh=False))

  def test_decide_reroute_second_unknown_blocks(self):
    # ambiguity protects: never cancel on a second UNKNOWN.
    self.assertFalse(R.decide_reroute('PENDING', 'UNKNOWN', output_fresh=False))

  def test_decide_reroute_first_not_pending_never_reroutes(self):
    self.assertFalse(R.decide_reroute('RUNNING', None, output_fresh=False))
    self.assertFalse(R.decide_reroute('TERMINAL', None, output_fresh=False))

  def test_decide_reroute_none_second_is_defensive_noop(self):
    self.assertFalse(R.decide_reroute('PENDING', None, output_fresh=False))


class InplaceRerouteDecisionTest(unittest.TestCase):
  """decide_inplace_reroute + record_eviction: the pure core of the in-place
  preemption-thrash detector (route_check owns the CNS measurement)."""

  def test_running_with_enough_restarts_reroutes(self):
    self.assertTrue(R.decide_inplace_reroute('RUNNING', 3, threshold=3))
    self.assertTrue(R.decide_inplace_reroute('RUNNING', 9, threshold=3))

  def test_below_threshold_does_not_reroute(self):
    self.assertFalse(R.decide_inplace_reroute('RUNNING', 2, threshold=3))
    self.assertFalse(R.decide_inplace_reroute('RUNNING', 0, threshold=3))

  def test_none_signal_is_never_a_verdict(self):
    # The probe could not measure it -> do nothing, the same fail-safe bias as
    # every other unknown in the reroute path.
    self.assertFalse(R.decide_inplace_reroute('RUNNING', None, threshold=3))

  def test_only_running_rows_are_eligible(self):
    # PENDING/TERMINAL/UNKNOWN/COMPLETED are handled by the other branches; this
    # guard must never fire on them, even with a huge restart count.
    for st in ('PENDING', 'TERMINAL', 'UNKNOWN', 'COMPLETED'):
      self.assertFalse(R.decide_inplace_reroute(st, 99, threshold=3))

  def test_threshold_is_configurable(self):
    self.assertFalse(R.decide_inplace_reroute('RUNNING', 4, threshold=5))
    self.assertTrue(R.decide_inplace_reroute('RUNNING', 5, threshold=5))

  def test_record_eviction_stamps_current_pair(self):
    e = _entry()
    e.cell = 'ej'
    e.arch = 'v7'
    R.record_eviction(e, now=1000.0)
    self.assertEqual(e.evictions, {'ej|v7': {'strikes': 1, 'last': 1000.0}})

  def test_record_eviction_accumulates_strikes(self):
    e = _entry()
    e.cell = 'ej'
    e.arch = 'v7'
    R.record_eviction(e, now=1000.0)
    R.record_eviction(e, now=2000.0)
    self.assertEqual(e.evictions['ej|v7']['strikes'], 2)
    self.assertEqual(e.evictions['ej|v7']['last'], 2000.0)   # most recent wins

  def test_NEGCTL_eviction_does_not_penalise_other_arch_in_that_cell(self):
    # Evicted off v7 in ej: v6p in ej is a different pair and ranks unpenalised,
    # so the roomier ej still beats ek for v6p.
    e = _entry(power='v6p-32', archs=('v6p',))
    e.cell = 'ej'
    e.arch = 'v7'
    R.record_eviction(e, now=1000.0)
    avail = {'ej|v6p': _avail('ej', 'v6p', free=3200),
             'ek|v6p': _avail('ek', 'v6p', free=64)}
    hit = R.best_cell_for_shape('v6p', 32, e, avail, now=1000.0)
    assert hit is not None
    got, _ = hit
    self.assertEqual(got.cell, 'ej')

  def test_record_eviction_noop_without_cell(self):
    e = _entry()
    e.cell = None
    R.record_eviction(e, now=1000.0)
    self.assertEqual(e.evictions, {})

  def test_recorded_eviction_penalises_that_cell_in_ranking(self):
    # End-to-end with the ranker: a struck cell sorts worse than a clean one of
    # equal price, so the next placement avoids it. This is the whole reason the
    # write side exists (the read side was 'wired, not fed' until now).
    e = _entry(power='v7-32', archs=('v7',))
    e.cell = 'yulpptr'
    R.record_eviction(e, now=1000.0)
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=3200, price=8.0),  # just evicted us
        'yukulwh': _avail('yukulwh', 'v7', free=3200, price=8.0),  # clean, same price
    }
    p = _ok(R.plan_one(e, avail, now=1000.0))
    self.assertEqual(p.cell, 'yukulwh')   # avoided the cell that evicted us

  def test_eviction_penalty_decays_so_an_old_strike_does_not_exclude(self):
    # A strike older than EVICT_DECAY_S is spent: the cheaper (here equal) cell
    # is chosen on price again, proving the record is a fading hint not a ban.
    e = _entry(power='v7-32', archs=('v7',))
    e.cell = 'yulpptr'
    R.record_eviction(e, now=0.0)
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=3200, price=8.0),
        'yukulwh': _avail('yukulwh', 'v7', free=3200, price=8.0),
    }
    # now well past EVICT_DECAY_S -> penalty back to 1.0, tie broken on roominess
    p = _ok(R.plan_one(e, avail, now=R.EVICT_DECAY_S + 100.0))
    self.assertIn(p.cell, ('yulpptr', 'yukulwh'))  # placed (not excluded)


class PairCooldownTest(unittest.TestCase):
  """The (cell, arch) pair cooldown (operator 2026-09-23). A re-route cools the
  ONE pair the job was stuck on; the same cell under another arch, the same arch
  in another cell, and the metro are left alone. Strikes stack inside the window
  and multiply the cell score."""

  # --- the multiplier ---
  def test_penalty_scales_with_strikes_and_caps(self):
    self.assertEqual(R.cooldown_penalty(1800.0, 0.0, 1800.0, strikes=1), 2.0)
    self.assertEqual(R.cooldown_penalty(1800.0, 0.0, 1800.0, strikes=2), 3.0)
    self.assertEqual(R.cooldown_penalty(1800.0, 0.0, 1800.0, strikes=99),
                     1.0 + R.PAIR_COOLDOWN_MAX_STRIKES)

  def test_penalty_is_flat_until_the_last_fade_window(self):
    # A 7200 s cooldown read with the 1800 s fade: full strength at 5400 s left.
    self.assertEqual(R.cooldown_penalty(7200.0, 1800.0, 1800.0, strikes=1), 2.0)
    self.assertAlmostEqual(
        R.cooldown_penalty(7200.0, 6300.0, 1800.0, strikes=1), 1.5)

  def test_NEGCTL_expired_absent_or_no_strikes_is_neutral(self):
    self.assertEqual(R.cooldown_penalty(None, 0.0), 1.0)
    self.assertEqual(R.cooldown_penalty(100.0, 100.0, strikes=3), 1.0)
    self.assertEqual(R.cooldown_penalty(1800.0, 0.0, 1800.0, strikes=0), 1.0)

  # --- write side ---
  def test_mark_reroute_cools_exactly_the_stuck_pair(self):
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    e.state = R.JobState.SUBMITTED
    e.cell = 'sj'
    e.arch = 'b200'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.cooldown_pairs,
                     {'sj|b200': {'until': 2500.0, 'strikes': 1}})
    self.assertFalse(hasattr(e, 'cooldown_cells'))
    self.assertFalse(hasattr(e, 'cooldown_archs'))
    self.assertFalse(hasattr(e, 'cooldown_metros'))

  def test_strikes_stack_inside_the_window_and_reset_after(self):
    e = _entry(power='v7-32', archs=('v7',))

    def reroute(now):
      e.state = R.JobState.SUBMITTED
      e.cell, e.arch, e.submitted_at = 'nm', 'v7', now - 1
      R.mark_reroute(e, now=now, cooldown_s=1000.0)
      return e.cooldown_pairs['nm|v7']

    self.assertEqual(reroute(0.0), {'until': 1000.0, 'strikes': 1})
    self.assertEqual(reroute(500.0), {'until': 1500.0, 'strikes': 2})
    self.assertEqual(reroute(3000.0), {'until': 4000.0, 'strikes': 1})

  def test_NEGCTL_other_pairs_do_not_stack(self):
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    for cell, arch in (('nm', 'v7'), ('nf', 'v7'), ('nm', 'v6p')):
      e.state = R.JobState.SUBMITTED
      e.cell, e.arch, e.submitted_at = cell, arch, 0.0
      R.mark_reroute(e, now=100.0, cooldown_s=1000.0)
    self.assertEqual({k: v['strikes'] for k, v in e.cooldown_pairs.items()},
                     {'nm|v7': 1, 'nf|v7': 1, 'nm|v6p': 1})

  def test_unknown_arch_blames_the_cell_under_every_accepted_arch(self):
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    e.state = R.JobState.SUBMITTED
    e.cell = 'sj'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=0.0, cooldown_s=100.0)
    self.assertEqual(sorted(e.cooldown_pairs), ['sj|b200', 'sj|h100'])

  def test_NEGCTL_no_cell_cools_nothing(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 0.0
    R.mark_reroute(e, now=0.0, cooldown_s=100.0)
    self.assertEqual(e.cooldown_pairs, {})
    self.assertEqual(e.state, R.JobState.QUEUED)

  # --- read side, end to end ---
  def test_same_cell_other_arch_is_not_penalised(self):
    # Stuck on b200 in sj. h100 in sj is a different pair: it keeps its full
    # standing and, being roomier, beats h100 in sh. The old per-cell and
    # per-metro cooldowns penalised sj for h100 too and sent the job to sh.
    e = _entry(power='h100-8', archs=('h100', 'b200'))
    e.state = R.JobState.SUBMITTED
    e.cell, e.arch, e.submitted_at = 'sj', 'b200', 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    avail = {
        'sj|b200': _avail('sj', 'b200', free=1024, metro='sjc'),
        'sj|h100': _avail('sj', 'h100', free=1024, metro='sjc'),
        'sh|h100': _avail('sh', 'h100', free=16, metro='shx'),
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    self.assertEqual((p.arch, p.cell), ('h100', 'sj'))

  def test_same_arch_other_cell_is_not_penalised(self):
    # Stuck on v7 in nm. v7 in nf is untouched, so the job stays on its
    # preferred arch in another cell. The old arch cooldown demoted v7
    # everywhere and pushed the job onto v6p.
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    e.state = R.JobState.SUBMITTED
    e.cell, e.arch, e.submitted_at = 'nm', 'v7', 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    avail = {
        'nm|v7': _avail('nm', 'v7', free=3200),
        'nf|v7': _avail('nf', 'v7', free=3200),
        'x|v6p': _avail('x', 'v6p', free=3200),
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    self.assertEqual((p.arch, p.cell), ('v7', 'nf'))

  def test_stacked_strikes_push_a_pair_below_a_dearer_rival(self):
    e = _entry(power='v7-32', archs=('v7',))
    avail = {'nm|v7': _avail('nm', 'v7', free=64, price=8.0),
             'nf|v7': _avail('nf', 'v7', free=64, price=20.0)}
    e.cooldown_pairs = {'nm|v7': {'until': 1e9, 'strikes': 1}}
    hit = R.best_cell_for_shape('v7', 32, e, avail, now=0.0)
    assert hit is not None
    got, _ = hit
    self.assertEqual(got.cell, 'nm')            # 8 * 2 = 16 < 20
    e.cooldown_pairs = {'nm|v7': {'until': 1e9, 'strikes': 3}}
    hit = R.best_cell_for_shape('v7', 32, e, avail, now=0.0)
    assert hit is not None
    got, _ = hit
    self.assertEqual(got.cell, 'nf')            # 8 * 4 = 32 > 20

  def test_pair_cooldown_expires(self):
    e = _entry(power='v7-32', archs=('v7',))
    e.state = R.JobState.SUBMITTED
    e.cell, e.arch, e.submitted_at = 'sj', 'v7', 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    avail = {'sj|v7': _avail('sj', 'v7', free=6400),   # roomier
             'nm|v7': _avail('nm', 'v7', free=3200)}
    p = _ok(R.plan_one(e, avail, now=700.0 + 1800.0 + 100))
    self.assertEqual(p.cell, 'sj')     # not banned: the roomier cell wins again

  def test_global_brake_is_120_per_hour(self):
    self.assertEqual(R.REROUTE_GLOBAL_MAX_PER_HOUR, 120)


class SerdeTest(unittest.TestCase):

  def test_roundtrip(self):
    e = _entry(priority=3, allowed_metros=['cbf', 'tul'],
               launch_kwargs={'config': 'remote_run', 'exp_name': 'x'})
    e.state = R.JobState.SUBMITTED
    e.cooldown_pairs = {'c|v7': {'until': 9.0, 'strikes': 2}}
    d = e.to_dict()
    self.assertEqual(d['state'], 'SUBMITTED')
    e2 = R.QueueEntry.from_dict(d)
    self.assertEqual(e2.priority, 3)
    self.assertEqual(e2.state, R.JobState.SUBMITTED)
    self.assertEqual(e2.launch_kwargs['config'], 'remote_run')
    self.assertEqual(e2.cooldown_pairs, {'c|v7': {'until': 9.0, 'strikes': 2}})

  def test_old_row_with_single_axis_cooldowns_loads_clean(self):
    d = _entry().to_dict()
    d.pop('cooldown_pairs')
    d['cooldown_cells'] = {'nm': 5.0}
    d['cooldown_archs'] = {'v7': {'until': 5.0, 'strikes': 2}}
    d['cooldown_metros'] = {'tul': 5.0}
    e = R.QueueEntry.from_dict(d)
    self.assertEqual(e.cooldown_pairs, {})
    self.assertNotIn('cooldown_cells', e.to_dict())

  def test_last_build_duration_roundtrips_and_defaults_none(self):
    # New diagnostic field: defaults None (a row built by an older binary), and
    # survives the JSON round-trip when set. A missing key must restore None
    # (from_dict drops unknown keys), so an old queue file loads without error.
    e = _entry()
    self.assertIsNone(e.last_build_duration)          # default
    e.last_build_duration = 73.5
    e2 = R.QueueEntry.from_dict(e.to_dict())
    self.assertEqual(e2.last_build_duration, 73.5)     # round-trips
    d = e.to_dict()
    del d['last_build_duration']                       # simulate old row
    self.assertIsNone(R.QueueEntry.from_dict(d).last_build_duration)


class SubmissionsViewTest(unittest.TestCase):
  """Phase 1 of the local-job-identity redesign: `submissions` is a DERIVED view
  over the still-authoritative xid/prior_xids, so this whole class asserts two
  things at once -- the v1->v2 migration is correct, AND it is a no-op on data
  (xid/prior_xids stay the source of truth; only serialization gains a key)."""

  def test_migration_from_v1_dict_prior_and_current(self):
    # The v1->v2 migration runs in from_dict on a queue file that predates
    # `submissions` (scalar xid + prior_xids, no submissions key). A RUNNING row
    # re-routed once must yield oldest-first submissions: the old id SUPERSEDED,
    # the current one RUNNING (state taken from the ROW) carrying its
    # cell/arch/chips. This is the exact shape the attnfilm incident lacked.
    v1 = {'job_id': 'j', 'power': 'v7-32', 'allowed_archs': ['v7'],
          'state': 'RUNNING', 'xid': '289723858', 'prior_xids': ['289686907'],
          'cell': 'sj', 'arch': 'v6e', 'chips': 32,
          'submitted_at': 1789500000.0}
    e = R.QueueEntry.from_dict(v1)
    subs = e.submissions
    self.assertEqual([s.seq for s in subs], [1, 2])
    self.assertEqual([s.xid for s in subs], ['289686907', '289723858'])
    self.assertEqual(subs[0].state, 'SUPERSEDED')
    self.assertEqual(subs[1].state, 'RUNNING')
    self.assertEqual((subs[1].cell, subs[1].arch, subs[1].chips),
                     ('sj', 'v6e', 32))
    # and the derived views still read the way every caller expects
    self.assertEqual(e.xid, '289723858')
    self.assertEqual(e.prior_xids, ['289686907'])

  def test_derived_helpers(self):
    e = _entry(xid='3', prior_xids=['1', '2'])
    e.state = R.JobState.SUBMITTED
    self.assertEqual(e.all_xids, ['1', '2', '3'])       # oldest first, incl. current
    self.assertIsNotNone(e.current_submission)
    self.assertEqual(e.current_submission.xid, '3')     # newest = the one xid points at

  def test_unplaced_row_has_no_submissions(self):
    e = _entry()                                        # QUEUED, xid=None
    self.assertEqual(e.submissions, [])
    self.assertIsNone(e.current_submission)
    self.assertEqual(e.all_xids, [])

  def test_held_with_attempts_but_no_xid_has_no_submissions(self):
    # The incident's false reason was "never produced an XID"; a row that truly
    # never produced one has an EMPTY submissions list -- a computed HELD reason
    # can trust that, where the hardcoded string could not.
    e = _entry(attempts=3)
    e.state = R.JobState.HELD
    self.assertEqual(e.submissions, [])

  def test_mid_reroute_current_xid_none_keeps_history(self):
    # mark_reroute pushes the live xid into prior_xids and sets xid=None. The
    # history must survive: all_xids still lists every id, none orphaned.
    e = _entry(xid=None, prior_xids=['1', '2', '3'])
    self.assertEqual(e.all_xids, ['1', '2', '3'])

  def test_to_dict_publishes_submissions_and_is_deterministic(self):
    # to_dict feeds merge_and_save_touched's baseline-equality check, so equal
    # inputs MUST serialize equal -- a nondeterministic submissions view would
    # make every tick look like a concurrent conflict.
    e = _entry(xid='9', prior_xids=['7', '8'])
    e.state = R.JobState.SUBMITTED
    d = e.to_dict()
    self.assertIn('submissions', d)
    self.assertEqual([s['xid'] for s in d['submissions']], ['7', '8', '9'])
    self.assertEqual(e.to_dict(), e.to_dict())          # deterministic

  def test_v1_read_compat_submissions_key_is_derived_not_stored(self):
    # A v2 file loaded by from_dict must ignore the serialized `submissions`
    # (it is a projection) and rebuild it from the authoritative xid/prior_xids,
    # so an OLD binary that never wrote the key and a NEW one round-trip to the
    # SAME bytes. Also proves from_dict tolerates the extra key without error.
    e = _entry(xid='2', prior_xids=['1'])
    e.state = R.JobState.RUNNING
    d = e.to_dict()
    e2 = R.QueueEntry.from_dict(d)                       # loads a v2 dict
    self.assertEqual(e2.xid, '2')
    self.assertEqual(e2.prior_xids, ['1'])
    self.assertEqual(e2.to_dict()['submissions'], d['submissions'])
    d_v1 = dict(d)
    del d_v1['submissions']                             # simulate a v1 row
    e3 = R.QueueEntry.from_dict(d_v1)                    # must load fine
    self.assertEqual(e3.all_xids, ['1', '2'])           # and re-derive the view


class FourSameNameNoOrphanTest(unittest.TestCase):
  """Regression for the 2026-09-15 attnfilm incident (design §6 asks for this).

  ROOT CAUSE: a row was re-routed repeatedly, each attempt creating an XManager
  experiment with the SAME exp_name; the scalar `xid` was overwritten and the
  recovery-by-name helper adopted `max(rows)` -- the NEWEST same-name id --
  while an EARLIER experiment was still RUNNING. The running experiment had no
  local row pointing at it, so it billed invisibly and the row's HELD reason
  claimed it had 'never produced an XID'.

  With submissions authoritative, EVERY id a re-route leaves behind is retained
  (SUPERSEDED, never dropped), so none can be orphaned; and the HELD reason is
  computed from that history, so it can never again lie."""

  def test_four_reroutes_same_name_orphan_none_and_reason_is_true(self):
    e = _entry(job_id='attnfilm-4n-film-2e4')
    e.launch_kwargs = {'exp_name': 'attnfilm-4n-film-2e4'}   # the single shared name
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 0.0
    created = []
    # Four placements of the SAME job, each a distinct XID, re-routed between.
    for i, x in enumerate(['288001', '288002', '288003', '288004']):
      p = R.Placement(job_id=e.job_id, cell=f'cell{i}', arch='v6e', chips=16,
                      price=8.0, reason='placed', geometry=None)
      R.apply_placement(e, p, xid=x, now=100.0 + i)
      created.append(x)
      if i < 3:
        R.mark_reroute(e, now=150.0 + i, cooldown_s=1800.0)

    # 1) NO XID IS ORPHANED: every experiment ever created is still tracked.
    self.assertEqual(e.all_xids, created)
    self.assertEqual(set(R.entry_xids(e)), set(created))

    # 2) exactly one live submission -- the last placement -- and it is `xid`.
    live = [s for s in e.submissions if s.state in R.SUBMISSION_LIVE_STATES]
    self.assertEqual([s.xid for s in live], ['288004'])
    self.assertEqual(e.xid, '288004')
    self.assertEqual(e.prior_xids, ['288001', '288002', '288003'])

    # 3) the earlier ids are retained as SUPERSEDED, never dropped.
    superseded = [s.xid for s in e.submissions if s.state == 'SUPERSEDED']
    self.assertEqual(superseded, ['288001', '288002', '288003'])

    # 4) the HELD reason tells the TRUTH -- it can no longer say "never produced
    # an XID" for a row that produced four.
    R.hold_entry(e, 'build failed 3 times')
    self.assertNotIn('never produced', e.last_reason.lower())
    for x in created:
      self.assertIn(x, e.last_reason)         # every real id is named

  def test_held_row_that_truly_never_created_says_so(self):
    # The honest negative: a row that genuinely never created an experiment must
    # still read as such -- the fix must not paper over the real no-XID case.
    e = _entry(job_id='never-built')
    e.state = R.JobState.BUILDING
    R.hold_entry(e, 'build failed 3 times')
    self.assertIn('no experiment was ever created', e.last_reason)



class TopologyLockTest(unittest.TestCase):

  def test_geometry_equivalence(self):
    self.assertTrue(R.same_topology('v6p', 32, 'v7', 32))    # both 2x4x4
    self.assertTrue(R.same_topology('v4', 32, 'v5p', 32))    # both 2x4x4
    self.assertFalse(R.same_topology('v6p', 32, 'v6e', 32))  # 2x4x4 vs 4_8
    self.assertFalse(R.same_topology('v6p', 32, 'v6p', 64))  # 2x4x4 vs 4x4x4

  def test_unlocked_job_ignores_geometry(self):
    # not locked: v6e is a valid fallback even though its geometry differs from
    # v6p. Power math: v6p-32 == 138.88 v5p-eq, window [104.16, 208.32]; the
    # equivalent v6e shape is v6e-64 (128), NOT v6e-32 (64, below window). This
    # doubles as a check that power-equivalence picks the right chip count.
    e = _entry(power='v6p-32', archs=('v6e',))
    shapes = R.candidate_shapes(e)
    self.assertIn(('v6e', 64), shapes)
    self.assertNotEqual(R.geometry_of('v6e', 64), '2x4x4')  # geometry differs, still allowed

  def test_locked_job_restricts_to_pinned_geometry(self):
    # paligemma-style: locked on 2x4x4, allows v6p+v7+v6e families
    e = _entry(power='v6p-32', archs=('v7', 'v6p', 'v6e'),
               topology_locked=True, locked_geometry='2x4x4')
    shapes = R.candidate_shapes(e)
    # v7-32 and v6p-32 are 2x4x4 -> allowed; v6e-* (4_8/etc) -> excluded
    self.assertIn(('v7', 32), shapes)
    self.assertIn(('v6p', 32), shapes)
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))

  def test_locked_job_can_move_v6p_to_v7(self):
    # the operator's example: locked job pending on v6p-32 re-routes to v7-32
    e = _entry(power='v6p-32', archs=('v7', 'v6p'),
               topology_locked=True, locked_geometry='2x4x4')
    avail = {'c': _avail('c', 'v7', free=320)}   # only v7 free
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertIsNotNone(p)
    self.assertEqual((p.arch, p.chips), ('v7', 32))

  def test_locked_job_refuses_different_geometry_even_if_free(self):
    # locked on 2x4x4; only v6e (4_8) is free -> must NOT place
    e = _entry(power='v6p-32', archs=('v6p', 'v6e'),
               topology_locked=True, locked_geometry='2x4x4')
    avail = {'c': _avail('c', 'v6e', free=3200)}   # tons of v6e free
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_locked_unpinned_uses_power_spec_geometry_as_anchor(self):
    # REGRESSION: a locked v6p-32 (2x4x4) job NOT yet placed, allowing v6e too.
    # Only a v6e-64 (8_8) slice is free and its power (64 v5p-eq) falls inside
    # the v6p-32 tolerance window -- but 8_8 != 2x4x4, so it MUST be refused.
    # Before the fix the unpinned branch applied no geometry filter and this
    # job landed on v6e-64, unrestorable for a 2x4x4-sharded checkpoint.
    e = _entry(power='v6p-32', archs=('v7', 'v6p', 'v6e'),
               topology_locked=True)
    self.assertIsNone(e.locked_geometry)
    avail = {'c': _avail('c', 'v6e', free=3200)}   # only v6e free
    self.assertIsNone(R.plan_one(e, avail, now=0.0))
    # candidate_shapes for this job must all be 2x4x4, never a v6e shape
    shapes = R.candidate_shapes(e)
    self.assertTrue(shapes)                          # v7-32 / v6p-32 qualify
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))
    self.assertFalse(any(a == 'v6e' for a, _ in shapes))

  def test_locked_bare_int_power_is_unplaceable(self):
    # A locked job whose power is a bare int names no arch => no anchor mesh.
    # Placing it would guess a geometry for a sharded checkpoint, so refuse.
    e = _entry(power='32', archs=('v6p', 'v7'), topology_locked=True)
    avail = {'c': _avail('c', 'v7', free=320)}
    self.assertIsNone(R.plan_one(e, avail, now=0.0))
    self.assertEqual(R.candidate_shapes(e), [])

  def test_power_geometry_helper(self):
    self.assertEqual(R.power_geometry('v6p-32'), '2x4x4')
    self.assertEqual(R.power_geometry('v7-32'), '2x4x4')
    self.assertEqual(R.power_geometry('v6e-64'), '8_8')
    self.assertIsNone(R.power_geometry('32'))          # bare int: no geometry

  def test_apply_placement_freezes_geometry_on_first_submit(self):
    # locked but not yet pinned: first placement freezes the mesh
    e = _entry(power='v6p-32', archs=('v7', 'v6p'), topology_locked=True)
    self.assertIsNone(e.locked_geometry)
    avail = {'c': _avail('c', 'v7', free=320)}
    p = _ok(R.plan_one(e, avail, now=0.0))
    R.apply_placement(e, p, xid='999', now=10.0)
    self.assertEqual(e.locked_geometry, '2x4x4')   # frozen from v7-32
    # after re-route, it stays pinned to 2x4x4
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.locked_geometry, '2x4x4')
    shapes = R.candidate_shapes(e)
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))

  def test_apply_placement_unlocked_does_not_pin(self):
    e = _entry(power='v6p-32', archs=('v7',))   # not locked
    avail = {'c': _avail('c', 'v7', free=320)}
    p = _ok(R.plan_one(e, avail, now=0.0))
    R.apply_placement(e, p, xid='1', now=0.0)
    self.assertIsNone(e.locked_geometry)



class TypeSelectionWeightTest(unittest.TestCase):

  def test_pool_weight_bounds(self):
    # ★Derive the ceiling from the constant instead of hardcoding it. This test
    # asserted a literal 1.20 and went red the moment POOL_BONUS was retuned
    # 0.20 -> 0.50, which reads as a regression in the code when it is only the
    # test restating an old value. A bound test should assert the SHAPE (0 pool
    # earns nothing, a full pool earns exactly the bonus, partial is strictly
    # between) so retuning the knob does not manufacture a false failure.
    ceiling = 1.0 + R.POOL_BONUS
    self.assertEqual(R.pool_weight(0), 1.0)
    self.assertAlmostEqual(R.pool_weight(R.POOL_FULL_BONUS_CHIPS), ceiling,
                           places=2)
    self.assertGreater(R.pool_weight(R.POOL_FULL_BONUS_CHIPS * 10),
                       ceiling - 1e-9)
    mid = R.pool_weight(64)
    self.assertGreater(mid, 1.0)
    self.assertLess(mid, ceiling)

  def test_effective_price_big_pool_reads_cheaper(self):
    # big pool (full 20% bonus) at 24 vs thin pool at 23: 24/1.2=20.0 beats
    # 23/~1.05=~21.9. A big pool rescues a modestly-pricier type.
    big = R.effective_price(24.0, 4096)
    thin = R.effective_price(23.0, 10)
    self.assertLess(big, thin)

  def test_effective_price_respects_pool_bonus_ceiling(self):
    """The pool bonus is capped: a big pool cannot forgive an arbitrary price.

    ★Derived from POOL_BONUS rather than hardcoded. The old version compared
    30.0 against a literal 24.0 chosen for POOL_BONUS=0.20; at 0.50 the bonus
    legitimately covers that gap, so the test failed while the ceiling it meant
    to check was working. Pick the probe price from the constant instead: just
    above the ceiling must stay more expensive, just below must come out cheaper.
    """
    ceiling = 1.0 + R.POOL_BONUS
    big = R.effective_price(30.0, 1e9)          # 30 / ceiling
    just_over = 30.0 / ceiling * 1.05
    just_under = 30.0 / ceiling * 0.95
    self.assertLess(R.effective_price(just_under, 0), big)
    self.assertGreater(R.effective_price(just_over, 0), big)

  def test_candidate_shapes_effective_price_ordering(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    shapes = R.candidate_shapes(
        e, arch_price={'v7': 20.0, 'v6p': 9.77},
        arch_pool={'v7': 4096, 'v6p': 30})
    self.assertEqual(shapes[0][0], 'v6p')

  def test_candidate_shapes_big_pool_wins_when_close(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    shapes = R.candidate_shapes(
        e, arch_price={'v7': 20.0, 'v6p': 18.0},
        arch_pool={'v7': 8192, 'v6p': 20})
    self.assertEqual(shapes[0][0], 'v7')

  def test_no_market_data_falls_back_to_arch_pref(self):
    e = _entry(power='v6p-32', archs=('v6p', 'v7'))
    shapes = R.candidate_shapes(e)
    self.assertEqual(shapes[0][0], 'v7')

  def test_plan_one_uses_effective_price_to_pick_type(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    avail = {
        'v7cell': _avail('v7cell', 'v7', free=3200, price=20.0),
        'v6pcell': _avail('v6pcell', 'v6p', free=3200, price=9.77),
    }
    p = _ok(R.plan_one(e, avail, now=0.0,
                       arch_price={'v7': 20.0, 'v6p': 9.77},
                       arch_pool={'v7': 4096, 'v6p': 30}))
    self.assertEqual(p.arch, 'v6p')


class SerialWorkerInvariantTest(unittest.TestCase):

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_count_building(self):
    es = [self._q('a', R.JobState.QUEUED),
          self._q('b', R.JobState.BUILDING, build_started_at=100.0),
          self._q('c', R.JobState.SUBMITTED)]
    self.assertEqual(R.count_building(es), 1)

  def test_can_claim_false_when_live_build(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=100.0)]
    self.assertFalse(R.can_claim_build(es, now=150.0, stale_after_s=1800.0))

  def test_can_claim_true_when_build_is_stale(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=100.0)]
    # 100 + 1800 = 1900 < 2000 -> stale, slot is free again
    self.assertTrue(R.can_claim_build(es, now=2000.0, stale_after_s=1800.0))

  def test_can_claim_true_when_none_building(self):
    es = [self._q('a', R.JobState.QUEUED)]
    self.assertTrue(R.can_claim_build(es, now=0.0, stale_after_s=1800.0))

  def test_building_no_timestamp_is_stale(self):
    e = self._q('b', R.JobState.BUILDING, build_started_at=None)
    self.assertTrue(R.building_is_stale(e, now=0.0, stale_after_s=1800.0))

  def test_reclaim_stale_building(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=0.0, worker_id='w1')]
    reclaimed = R.reclaim_stale_building(es, now=2000.0, stale_after_s=1800.0)
    self.assertEqual([e.job_id for e in reclaimed], ['b'])
    self.assertEqual(es[0].state, R.JobState.QUEUED)
    self.assertIsNone(es[0].build_started_at)
    self.assertIsNone(es[0].worker_id)

  def test_reclaim_leaves_live_building(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=1000.0)]
    reclaimed = R.reclaim_stale_building(es, now=1100.0, stale_after_s=1800.0)
    self.assertEqual(reclaimed, [])
    self.assertEqual(es[0].state, R.JobState.BUILDING)

  def test_next_queued_priority(self):
    es = [self._q('lo', R.JobState.QUEUED, priority=1),
          self._q('hi', R.JobState.QUEUED, priority=9),
          self._q('bld', R.JobState.BUILDING, build_started_at=0.0)]
    self.assertEqual(_ok(R.next_queued(es)).job_id, 'hi')

  def test_next_queued_none_when_all_building_or_terminal(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=0.0),
          self._q('d', R.JobState.DONE)]
    self.assertIsNone(R.next_queued(es))

  def test_claim_for_build_marks_and_stamps(self):
    e = self._q('a', R.JobState.QUEUED)
    R.claim_for_build(e, now=500.0, worker_id='w7')
    self.assertEqual(e.state, R.JobState.BUILDING)
    self.assertEqual(e.build_started_at, 500.0)
    self.assertEqual(e.worker_id, 'w7')


class ReclaimEarlyBoundTest(unittest.TestCase):
  """§5.4: reclaim_stale_building resolves a stale BUILDING claim by READING the
  row's early-bound submission -- no exp_name lookup, no adopt_check_name park.

  The disaster this guards (2026-09-02, elt-dit-50k-fid-v3b): a build that
  SUCCEEDED and is RUNNING on XManager, reclaimed to QUEUED and re-dispatched,
  puts a SECOND writer on the first's output path. Early binding persists the
  XID before the build returns, so the row itself says escaped-vs-crashed."""

  def _building(self, job_id='j', started=1000.0):
    e = _entry(job_id)
    e.state = R.JobState.BUILDING
    e.build_started_at = started
    e.worker_id = 'w1'
    return e

  def test_escaped_build_is_adopted_not_rebuilt(self):
    # The build created an experiment (early-bound CREATING submission carrying
    # the xid) before the worker died. Reclaim must NOT requeue -- that is the
    # double-write -- but move it to SUBMITTED for reconcile to verify.
    e = self._building()
    e.open_creating(xid='285706173', cell='sj', arch='v6p', chips=32,
                    now=1000.0)
    touched = R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual([x.job_id for x in touched], ['j'])
    self.assertEqual(e.state, R.JobState.SUBMITTED)   # NOT QUEUED
    self.assertEqual(e.xid, '285706173')
    self.assertIsNone(e.build_started_at)
    self.assertIsNone(e.worker_id)

  def test_escaped_build_does_not_bump_attempts(self):
    # §5.5: recovering a lost binding is not a build failure. The attnfilm row
    # hit HELD partly because this path did attempts += 1.
    e = self._building()
    e.attempts = 1
    e.open_creating(xid='285706173', cell='sj', arch='v6p', chips=32,
                    now=1000.0)
    R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual(e.attempts, 1)                   # unchanged

  def test_escaped_build_backfills_placement_onto_row(self):
    # apply_placement never ran (worker died), so e.cell/arch/chips are None.
    # The submission holds the resolved landing; copy it across so the liveness
    # probes are not blind (the xid 288485310 16h-wedge defect).
    e = self._building()
    e.open_creating(xid='285706173', cell='sj', arch='v6p', chips=32,
                    now=1000.0)
    self.assertIsNone(e.cell)
    R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual((e.cell, e.arch, e.chips), ('sj', 'v6p', 32))

  def test_crashed_before_create_is_requeued_and_bumps_attempts(self):
    # No experiment was ever bound (no submission), so nothing escaped: safe to
    # rebuild. This is the ordinary worker-crash case.
    e = self._building()
    e.attempts = 0
    touched = R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual([x.job_id for x in touched], ['j'])
    self.assertEqual(e.state, R.JobState.QUEUED)      # rebuild
    self.assertEqual(e.attempts, 1)                   # bumped
    self.assertIsNone(e.build_started_at)

  def test_superseded_only_submission_is_treated_as_no_live_binding(self):
    # A submission that is no longer live (e.g. SUPERSEDED) does not protect the
    # row: current_submission is not in a live state, so this is the crash case.
    e = self._building()
    sub = e.open_creating(xid='111', now=1000.0)
    sub.state = 'SUPERSEDED'
    R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.attempts, 1)

  def test_creating_submission_without_xid_is_not_trusted(self):
    # open_creating can record a CREATING row with xid=None (the early line was
    # never seen). With no xid there is nothing to double-write, so requeue.
    e = self._building()
    e.open_creating(xid=None, now=1000.0)
    R.reclaim_stale_building([e], now=3000.0, stale_after_s=1800.0)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.attempts, 1)

  def test_live_build_is_left_alone(self):
    e = self._building(started=2900.0)
    self.assertEqual(R.reclaim_stale_building([e], now=3000.0,
                                              stale_after_s=1800.0), [])
    self.assertEqual(e.state, R.JobState.BUILDING)

  def test_no_adopt_check_name_attribute_remains(self):
    # The field is gone: constructing a row and dataclasses.fields must not
    # carry it, so a stray reference would fail loudly rather than silently.
    import dataclasses
    names = {f.name for f in dataclasses.fields(R.QueueEntry)}
    self.assertNotIn('adopt_check_name', names)


class AdoptRecoveredSubmissionTest(unittest.TestCase):
  """§5.4 last resort: adopt an experiment recovered by its jobid tag onto a
  row whose router died between create and persist. The row had no XID; adoption
  records it and moves the row to SUBMITTED so a rebuild never double-writes."""

  def _reclaimed_row(self, job_id='j'):
    # The state reclaim_stale_building leaves for a 'crashed before create' row:
    # QUEUED, no live submission, one build attempt burned.
    e = _entry(job_id)
    e.state = R.JobState.QUEUED
    e.attempts = 1
    return e

  def test_adopts_xid_and_moves_to_submitted(self):
    e = self._reclaimed_row()
    R.adopt_recovered_submission(e, '285706173', cell='sj', arch='v6p',
                                 chips=32, now=5000.0)
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '285706173')
    self.assertEqual(e.current_submission.state, 'CREATING')
    self.assertIsNone(e.build_started_at)
    self.assertIsNone(e.worker_id)

  def test_backfills_placement_onto_row(self):
    e = self._reclaimed_row()
    self.assertIsNone(e.cell)
    R.adopt_recovered_submission(e, '285706173', cell='sj', arch='v6p',
                                 chips=32, now=5000.0)
    self.assertEqual((e.cell, e.arch, e.chips), ('sj', 'v6p', 32))

  def test_does_not_bump_attempts(self):
    # §5.5: recovering a lost binding is not a build failure.
    e = self._reclaimed_row()
    e.attempts = 2
    R.adopt_recovered_submission(e, '285706173', now=5000.0)
    self.assertEqual(e.attempts, 2)

  def test_adopt_with_no_placement_leaves_row_fields_but_still_binds(self):
    # A recovered experiment whose launch_args could not be parsed: xid is still
    # adopted (that is what prevents the double-write), placement stays None for
    # the reroute-side backfill to fill later. Never guess a cell.
    e = self._reclaimed_row()
    R.adopt_recovered_submission(e, '285706173', now=5000.0)
    self.assertEqual(e.xid, '285706173')
    self.assertIsNone(e.cell)

  def test_recovered_submission_is_in_all_xids(self):
    e = self._reclaimed_row()
    R.adopt_recovered_submission(e, '285706173', now=5000.0)
    self.assertIn('285706173', e.all_xids)

  def test_reason_names_the_recovery(self):
    e = self._reclaimed_row()
    R.adopt_recovered_submission(e, '285706173', now=5000.0)
    self.assertIn('recovered by jobid tag', e.last_reason)
    self.assertIn('285706173', e.last_reason)


class BuildRequestedBackpressureTest(unittest.TestCase):
  """Step1: BUILD_REQUESTED handoff token + backpressure counting."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_mark_build_requested_transitions_without_touching_attempts(self):
    e = self._q('a', R.JobState.QUEUED, attempts=2)
    R.mark_build_requested(e)
    self.assertEqual(e.state, R.JobState.BUILD_REQUESTED)
    self.assertEqual(e.attempts, 2)  # dispatch is not a failure

  def test_count_build_pending_counts_requested_and_building(self):
    es = [self._q('a', R.JobState.QUEUED),
          self._q('b', R.JobState.BUILD_REQUESTED),
          self._q('c', R.JobState.BUILDING, build_started_at=1.0),
          self._q('d', R.JobState.SUBMITTED),
          self._q('e', R.JobState.BUDGET_DEFERRED)]
    self.assertEqual(R.count_build_pending(es), 2)

  def test_count_build_pending_zero_means_builder_drained(self):
    es = [self._q('a', R.JobState.QUEUED), self._q('d', R.JobState.SUBMITTED)]
    self.assertEqual(R.count_build_pending(es), 0)

  def test_next_build_requested_priority_then_none(self):
    es = [self._q('lo', R.JobState.BUILD_REQUESTED, priority=1),
          self._q('hi', R.JobState.BUILD_REQUESTED, priority=9),
          self._q('q', R.JobState.QUEUED, priority=99)]  # QUEUED not eligible
    self.assertEqual(_ok(R.next_build_requested(es)).job_id, 'hi')
    for e in es:
      if e.state == R.JobState.BUILD_REQUESTED:
        e.state = R.JobState.BUILDING
    self.assertIsNone(R.next_build_requested(es))

  def test_next_queued_ignores_build_requested(self):
    # BUILD_REQUESTED must NOT be re-picked by next_queued (only the builder
    # claims it) -- otherwise a job dispatched this round gets double-dispatched.
    es = [self._q('r', R.JobState.BUILD_REQUESTED, priority=9),
          self._q('q', R.JobState.QUEUED, priority=1)]
    self.assertEqual(_ok(R.next_queued(es)).job_id, 'q')


class BudgetDeferredTest(unittest.TestCase):
  """Step1: BUDGET_DEFERRED soft park + per-round promote."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_mark_budget_deferred_does_not_increment_attempts(self):
    e = self._q('a', R.JobState.BUILDING, attempts=1, build_started_at=5.0,
                worker_id='w1')
    R.mark_budget_deferred(e)
    self.assertEqual(e.state, R.JobState.BUDGET_DEFERRED)
    self.assertEqual(e.attempts, 1)          # budget refusal is NOT a failure
    self.assertIsNone(e.build_started_at)     # slot freed
    self.assertIsNone(e.worker_id)

  def test_mark_budget_deferred_never_becomes_held(self):
    # Even after many rounds of deferral, a job never accrues attempts toward HELD.
    e = self._q('a', R.JobState.QUEUED, attempts=0)
    for _ in range(10):
      R.mark_budget_deferred(e)
      R.promote_deferred([e])
    self.assertEqual(e.attempts, 0)
    self.assertNotEqual(e.state, R.JobState.HELD)

  def test_promote_deferred_returns_to_queued(self):
    es = [self._q('a', R.JobState.BUDGET_DEFERRED),
          self._q('b', R.JobState.QUEUED),
          self._q('c', R.JobState.BUDGET_DEFERRED)]
    promoted = R.promote_deferred(es)
    self.assertEqual(sorted(e.job_id for e in promoted), ['a', 'c'])
    self.assertTrue(all(e.state == R.JobState.QUEUED for e in es))

  def test_promote_deferred_noop_when_none_deferred(self):
    es = [self._q('b', R.JobState.QUEUED), self._q('r', R.JobState.RUNNING)]
    self.assertEqual(R.promote_deferred(es), [])


class ReconcileTest(unittest.TestCase):
  """Step1: XM-truth reconcile pure decision (R3 zombie cleanup)."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  # --- decide_reconcile truth table ---
  def test_terminal_from_running_is_failed(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.RUNNING, 'TERMINAL'), R.JobState.FAILED)

  def test_terminal_from_submitted_is_failed(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.SUBMITTED, 'TERMINAL'), R.JobState.FAILED)

  def test_running_promotes_submitted(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.SUBMITTED, 'RUNNING'), R.JobState.RUNNING)

  def test_running_running_is_noop(self):
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'RUNNING'))

  def test_pending_is_noop(self):
    # reroute (not reconcile) owns pending>deadline; reconcile leaves it.
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'PENDING'))

  def test_unknown_never_acts(self):
    # THE safety rule: a probe hiccup must never mark a live job dead.
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'UNKNOWN'))
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'UNKNOWN'))

  def test_unrecognised_status_is_noop(self):
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'WAT'))

  # --- reconcile_entry mutator ---
  def test_reconcile_entry_cleans_zombie(self):
    e = self._q('z', R.JobState.RUNNING, xid='123')
    self.assertTrue(R.reconcile_entry(e, 'TERMINAL'))
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertIn('zombie', e.last_reason)

  def test_reconcile_entry_promotes_placement(self):
    e = self._q('p', R.JobState.SUBMITTED, xid='123')
    self.assertTrue(R.reconcile_entry(e, 'RUNNING'))
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_reconcile_entry_unknown_leaves_unchanged(self):
    e = self._q('u', R.JobState.RUNNING, xid='123')
    self.assertFalse(R.reconcile_entry(e, 'UNKNOWN'))
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_reconcile_entry_skips_non_reconcilable(self):
    # QUEUED/HELD/BUDGET_DEFERRED/DONE/FAILED are never reconciled (no live xid).
    for st in (R.JobState.QUEUED, R.JobState.HELD, R.JobState.BUDGET_DEFERRED,
               R.JobState.DONE, R.JobState.FAILED, R.JobState.BUILD_REQUESTED):
      e = self._q('x', st)
      self.assertFalse(R.reconcile_entry(e, 'TERMINAL'),
                       f'{st} should not be reconciled')
      self.assertEqual(e.state, st)

  def test_reconcilable_states_membership(self):
    self.assertEqual(
        R.RECONCILABLE_STATES,
        frozenset({R.JobState.RUNNING, R.JobState.SUBMITTED, R.JobState.BUILDING}))


class MarkCancelledTest(unittest.TestCase):
  """route_lib.mark_cancelled: a deliberate `tpu cancel` retires the row as
  FAILED (FINISHED_STATES unchanged) but labels the CURRENT submission
  CANCELLED, so the queue record says 'cancelled', not 'zombie'."""

  def _running(self, xid='123'):
    e = _entry('c', xid=xid)
    e.state = R.JobState.RUNNING
    e.submissions[-1].state = 'RUNNING'
    return e

  def _cur(self, e) -> R.Submission:
    cur = e.current_submission
    self.assertIsNotNone(cur)
    assert cur is not None  # narrows Optional for the type checker
    return cur

  def test_row_failed_submission_cancelled(self):
    e = self._running()
    self.assertTrue(R.mark_cancelled(e, when='2026-09-23 21:41:41'))
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertIn(e.state, R.FINISHED_STATES)
    cur = self._cur(e)
    self.assertEqual(cur.state, 'CANCELLED')
    self.assertIn(cur.state, R.SUBMISSION_TERMINAL_STATES)
    self.assertEqual(cur.ended_reason,
                     'cancelled via tpu cancel at 2026-09-23 21:41:41')
    self.assertIn('cancelled (tpu cancel at 2026-09-23 21:41:41', e.last_reason)
    self.assertIn('not a crash', e.last_reason)
    self.assertNotIn('zombie', e.last_reason)
    self.assertEqual(e.xid, '123')        # the id stays on the record

  def test_empty_when_has_no_dangling_at(self):
    e = self._running()
    R.mark_cancelled(e)
    self.assertEqual(self._cur(e).ended_reason, 'cancelled via tpu cancel')
    self.assertTrue(e.last_reason.startswith('cancelled (tpu cancel;'))

  def test_explicit_reason_wins(self):
    e = self._running()
    R.mark_cancelled(e, reason='custom', when='t')
    self.assertEqual(e.last_reason, 'custom')
    self.assertEqual(self._cur(e).state, 'CANCELLED')

  def test_submitted_row_also_cancelled(self):
    e = _entry('s', xid='9')
    e.state = R.JobState.SUBMITTED
    self.assertTrue(R.mark_cancelled(e, when='t'))
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual(self._cur(e).state, 'CANCELLED')

  def test_superseded_submission_not_rewritten(self):
    # Only a LIVE submission with an xid is relabelled (sync_current_submission
    # rule): a record that already ended keeps its own state and reason.
    e = self._running()
    e.submissions[-1].state = 'SUPERSEDED'
    e.submissions[-1].ended_reason = 'rerouted'
    R.mark_cancelled(e, when='t')
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertEqual(e.submissions[-1].state, 'SUPERSEDED')
    self.assertEqual(e.submissions[-1].ended_reason, 'rerouted')

  def test_non_reconcilable_rows_untouched(self):
    for st in (R.JobState.QUEUED, R.JobState.HELD, R.JobState.DONE,
               R.JobState.FAILED, R.JobState.BUDGET_DEFERRED):
      e = _entry('x', xid='5')
      e.state = st
      before = e.to_dict()
      self.assertFalse(R.mark_cancelled(e, when='t'), st)
      self.assertEqual(e.to_dict(), before, st)

  def test_serde_roundtrip_keeps_cancelled(self):
    e = self._running()
    R.mark_cancelled(e, when='t')
    back = R.QueueEntry.from_dict(e.to_dict())
    self.assertEqual(back.state, R.JobState.FAILED)
    self.assertEqual(self._cur(back).state, 'CANCELLED')
    self.assertEqual(back.xid, '123')


class NewStateSerdeTest(unittest.TestCase):
  """Step1: the two new states survive the JSON round-trip (readable strings)."""

  def test_build_requested_roundtrip(self):
    e = _entry()
    e.state = R.JobState.BUILD_REQUESTED
    d = e.to_dict()
    self.assertEqual(d['state'], 'BUILD_REQUESTED')
    self.assertEqual(R.QueueEntry.from_dict(d).state, R.JobState.BUILD_REQUESTED)

  def test_budget_deferred_roundtrip(self):
    e = _entry()
    e.state = R.JobState.BUDGET_DEFERRED
    d = e.to_dict()
    self.assertEqual(d['state'], 'BUDGET_DEFERRED')
    self.assertEqual(R.QueueEntry.from_dict(d).state, R.JobState.BUDGET_DEFERRED)


class PlanDispatchTest(unittest.TestCase):
  """Step3: greedy dispatch with in-memory pre-debit (route_lib.plan_dispatch)."""

  def _q(self, job_id, priority=0):
    e = _entry(job_id)
    e.state = R.JobState.QUEUED
    e.priority = priority
    return e

  def _plan(self, entries, headroom, costs, exempt=()):
    cost_of = lambda e: costs.get(e.job_id, 0.0)
    is_exempt = lambda e: e.job_id in exempt
    return R.plan_dispatch(entries, headroom, cost_of, is_exempt)

  def test_empty_is_empty(self):
    self.assertEqual(self._plan([], 100.0, {}), [])

  def test_all_fit_all_dispatched(self):
    es = [self._q('a'), self._q('b')]
    out = self._plan(es, 100.0, {'a': 30.0, 'b': 40.0})
    self.assertTrue(all(d.decision == R.JobState.BUILD_REQUESTED for d in out))
    self.assertAlmostEqual(out[-1].headroom_after, 30.0)  # 100-30-40

  def test_pre_debit_stops_over_dispatch(self):
    # Two 60-cost jobs, headroom 100: only the first fits (60), second deferred
    # (60 > 40 left). Without pre-debit both would be admitted against 100.
    es = [self._q('a', priority=2), self._q('b', priority=1)]
    out = self._plan(es, 100.0, {'a': 60.0, 'b': 60.0})
    self.assertEqual(out[0].decision, R.JobState.BUILD_REQUESTED)
    self.assertEqual(out[1].decision, R.JobState.BUDGET_DEFERRED)

  def test_priority_order(self):
    es = [self._q('lo', priority=1), self._q('hi', priority=9)]
    out = self._plan(es, 1000.0, {'lo': 1.0, 'hi': 1.0})
    self.assertEqual(out[0].job_id, 'hi')  # highest priority first

  def test_head_of_line_does_not_block_smaller(self):
    # Big expensive job at head does NOT fit; a smaller cheaper job behind it
    # STILL gets dispatched (fixes H2 starvation).
    es = [self._q('big', priority=9), self._q('small', priority=1)]
    out = self._plan(es, 100.0, {'big': 500.0, 'small': 50.0})
    by = {d.job_id: d.decision for d in out}
    self.assertEqual(by['big'], R.JobState.BUDGET_DEFERRED)
    self.assertEqual(by['small'], R.JobState.BUILD_REQUESTED)  # not blocked

  def test_exempt_dispatched_without_debit(self):
    # An exempt job (g5/BATCH/CPU) is dispatched and does NOT consume headroom.
    es = [self._q('ex', priority=9), self._q('paid', priority=1)]
    out = self._plan(es, 50.0, {'ex': 999.0, 'paid': 50.0}, exempt={'ex'})
    by = {d.job_id: d for d in out}
    self.assertEqual(by['ex'].decision, R.JobState.BUILD_REQUESTED)
    self.assertEqual(by['ex'].cost, 0.0)                 # no debit for exempt
    self.assertEqual(by['paid'].decision, R.JobState.BUILD_REQUESTED)  # 50<=50 still

  def test_all_over_bar_all_deferred(self):
    es = [self._q('a'), self._q('b')]
    out = self._plan(es, 10.0, {'a': 100.0, 'b': 100.0})
    self.assertTrue(all(d.decision == R.JobState.BUDGET_DEFERRED for d in out))

  def test_zero_headroom_defers_paid_admits_exempt(self):
    es = [self._q('paid'), self._q('ex')]
    out = self._plan(es, 0.0, {'paid': 1.0, 'ex': 1.0}, exempt={'ex'})
    by = {d.job_id: d.decision for d in out}
    self.assertEqual(by['paid'], R.JobState.BUDGET_DEFERRED)
    self.assertEqual(by['ex'], R.JobState.BUILD_REQUESTED)

  def test_exact_fit_admitted(self):
    es = [self._q('a')]
    out = self._plan(es, 50.0, {'a': 50.0})
    self.assertEqual(out[0].decision, R.JobState.BUILD_REQUESTED)  # <= is inclusive
    self.assertAlmostEqual(out[0].headroom_after, 0.0)


class CheckpointStepTest(unittest.TestCase):
  """checkpoint_step parses all four fleet checkpoint spellings, -1 otherwise."""

  def test_torch_file(self):
    self.assertEqual(
        R.checkpoint_step('/cns/si-d/x/steps/step_1024.pt'), 1024)

  def test_jax_dir_trailing_slash(self):
    self.assertEqual(R.checkpoint_step('/cns/x/step_6144/'), 6144)

  def test_flat_dir_no_slash(self):
    self.assertEqual(R.checkpoint_step('/cns/x/step_500'), 500)

  def test_paligemma_checkpoint_prefix(self):
    self.assertEqual(R.checkpoint_step('/cns/x/checkpoint_20000'), 20000)

  def test_suffixed_name_still_parses(self):
    # a `_state` / `_best` suffix must not defeat the parse
    self.assertEqual(R.checkpoint_step('/cns/x/step_1024_best.pt'), 1024)

  # -- negative controls: anything unrecognised is -1, never 0 --
  def test_none_is_minus_one(self):
    self.assertEqual(R.checkpoint_step(None), -1)

  def test_empty_is_minus_one(self):
    self.assertEqual(R.checkpoint_step(''), -1)

  def test_non_checkpoint_name_is_minus_one(self):
    self.assertEqual(R.checkpoint_step('/cns/x/best/'), -1)
    self.assertEqual(R.checkpoint_step('/cns/x/latest.pt'), -1)

  def test_zero_step_is_zero_not_minus_one(self):
    # a genuine step_0 is a real (if useless) parse; distinct from unparseable
    self.assertEqual(R.checkpoint_step('/cns/x/step_0.pt'), 0)


class PlanPrunedRestartTest(unittest.TestCase):
  """The checkpoint-as-evidence path. Every guard must default to HOLD; only a
  healthy run killed from outside, with a surviving checkpoint, resumes warm."""

  def _e(self, **kw):
    return _entry(**kw)

  def test_pruned_healthy_run_resumes_warm(self):
    # the dw case: terminal, no code bug, checkpoint survived, sole writer
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False)
    self.assertEqual(verdict, R.RESUME_WARM)
    self.assertIn('step 1024', why)

  def test_not_terminal_holds(self):
    verdict, _ = R.plan_pruned_restart(
        self._e(), xm_terminal=False, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False)
    self.assertEqual(verdict, R.HOLD)

  def test_code_bug_holds_even_with_checkpoint(self):
    # NEGATIVE CONTROL: a segfault must NOT auto-resume, or we replay the bug
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True,
        code_bug='CODE BUG: segfault (SIGSEGV)',
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('code bug', why)

  def test_no_checkpoint_holds(self):
    # NEGATIVE CONTROL: no checkpoint -> a warm restart is a cold start
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None,
        checkpoint=None, other_live_writer=False)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('cold start', why)

  def test_other_live_writer_holds(self):
    # NEGATIVE CONTROL: the 2026-09-10 double-write -- never add a 2nd writer
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=True)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('SECOND writer', why)

  def test_budget_spent_holds(self):
    # NEGATIVE CONTROL: after N auto-resumes, stop and let a human look
    verdict, why = R.plan_pruned_restart(
        self._e(auto_resumes=3), xm_terminal=True, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False,
        max_auto_resumes=3)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('budget', why)

  def test_budget_one_below_cap_still_resumes(self):
    verdict, _ = R.plan_pruned_restart(
        self._e(auto_resumes=2), xm_terminal=True, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False,
        max_auto_resumes=3)
    self.assertEqual(verdict, R.RESUME_WARM)

  # --- RESUME_COLD: preempted-before-first-checkpoint (Fix 2, gated) --------
  # allow_cold defaults OFF, so all the tests above exercise the UNCHANGED
  # behavior. These pass allow_cold=True to exercise the new branch.

  def test_no_ckpt_default_still_holds_when_cold_disabled(self):
    # GUARD: with allow_cold defaulting False, no-checkpoint is STILL an
    # unconditional hold -- byte-for-byte the pre-Fix2 behavior, even for a
    # clearly-preempted, clearly-trained run.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None, checkpoint=None,
        other_live_writer=False, trained=True, termination_cause='preempted')
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('cold start', why)

  def test_cold_preempted_no_ckpt_reruns(self):
    # THE FIX: terminal, no code bug, no checkpoint, but a preemption cause.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None, checkpoint=None,
        other_live_writer=False, termination_cause='preempted', allow_cold=True)
    self.assertEqual(verdict, R.RESUME_COLD)

  def test_cold_trained_no_cause_no_ckpt_reruns(self):
    # THE 4-JOB CASE: clean preemption leaves NO cause in stdout, but the log
    # shows training progress -> cold rerun.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None, checkpoint=None,
        other_live_writer=False, trained=True, allow_cold=True)
    self.assertEqual(verdict, R.RESUME_COLD)

  def test_cold_never_trained_unknown_cause_holds(self):
    # NEGATIVE CONTROL: crashed on startup (never trained, no preempt cause) ->
    # still HOLD even with allow_cold, or we loop replaying a startup crash.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None, checkpoint=None,
        other_live_writer=False, trained=False, termination_cause=None,
        allow_cold=True)
    self.assertEqual(verdict, R.HOLD)

  def test_cold_still_blocked_by_code_bug(self):
    # NEGATIVE CONTROL: a real crash signature outranks cold rerun.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug='segfault (SIGSEGV)',
        checkpoint=None, other_live_writer=False, trained=True, allow_cold=True)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('code bug', why)

  def test_cold_still_blocked_by_live_writer(self):
    # NEGATIVE CONTROL: never add a 2nd writer, cold path included.
    verdict, why = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None, checkpoint=None,
        other_live_writer=True, trained=True, allow_cold=True)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('SECOND writer', why)

  def test_cold_still_bounded_by_budget(self):
    # NEGATIVE CONTROL: the anti-loop budget caps cold reruns too.
    verdict, why = R.plan_pruned_restart(
        self._e(auto_resumes=3), xm_terminal=True, code_bug=None,
        checkpoint=None, other_live_writer=False, trained=True,
        termination_cause='preempted', allow_cold=True, max_auto_resumes=3)
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('budget', why)

  def test_cold_prefers_warm_when_ckpt_exists(self):
    # When a checkpoint DOES exist, we still warm-resume (cold is only the
    # no-checkpoint fallback), even with allow_cold on.
    verdict, _ = R.plan_pruned_restart(
        self._e(), xm_terminal=True, code_bug=None,
        checkpoint='/cns/x/steps/step_1024.pt', other_live_writer=False,
        trained=True, termination_cause='preempted', allow_cold=True)
    self.assertEqual(verdict, R.RESUME_WARM)

class LooksLikeCodeBugTest(unittest.TestCase):
  """The code-bug gate: crashes in the TAIL flag, a healthy pruned tail does not,
  and the benign boot-banner ModuleNotFoundError must NOT be read as a bug."""

  def test_segfault_flags(self):
    self.assertIsNotNone(R.looks_like_code_bug('... Killed by signal 11!'))

  def test_traceback_flags(self):
    tail = 'Traceback (most recent call last):\n  File x\nValueError: bad'
    self.assertIsNotNone(R.looks_like_code_bug(tail))

  def test_oom_flags(self):
    self.assertIsNotNone(R.looks_like_code_bug('RESOURCE_EXHAUSTED: OOM when...'))

  def test_healthy_training_tail_is_none(self):
    tail = ('[parcae-torch] step 1759 loss 3.49 gnorm 0.51 272.4k tok/s\n'
            '[parcae-torch] step 1760 loss 3.50')
    self.assertIsNone(R.looks_like_code_bug(tail))

  def test_benign_modulenotfound_boot_note_is_none(self):
    # NEGATIVE CONTROL: the dw boot banner prints this harmless readback note;
    # it must NOT be classed as a code bug (that would HOLD every pruned run).
    note = ("[parcae-torch] minloglevel READ-BACK unavailable "
            "(ModuleNotFoundError: No module named 'base'); dep cpp_flag")
    self.assertIsNone(R.looks_like_code_bug(note))

  def test_empty_is_none(self):
    self.assertIsNone(R.looks_like_code_bug(''))

  def test_banner_sigsegv_advisory_is_none(self):
    # REGRESSION (2026-09-18): the launcher boot banner prints this advisory,
    # and its word 'SIGSEGVs' contains the substring 'SIGSEGV'. A naive
    # `'SIGSEGV' in tail` test HELD 4 healthy PREEMPTED runs as if they had
    # segfaulted. The word-boundary matcher must NOT flag it.
    banner = ('[parcae-torch] \u2605 minloglevel SET BUT UNPROVEN. If rank 0 '
              'SIGSEGVs in its first collective, THIS is why. Add '
              '//base/python/clif:cpp_flag to the BUILD deps.')
    self.assertIsNone(R.looks_like_code_bug(banner))

  def test_banner_advisory_then_healthy_training_is_none(self):
    # The full shape of the 4 mis-held runs: the advisory banner in the head,
    # then clean training with NO real crash. Must read as healthy.
    tail = ('If rank 0 SIGSEGVs in its first collective, THIS is why.\n'
            '[parcae-torch] step 300 loss 4.77 gnorm 2.06 0.57 step/s\n'
            '[parcae-torch] step 512 val_loss 4.02 val_ppl 56.1')
    self.assertIsNone(R.looks_like_code_bug(tail))

  def test_real_parenthesized_sigsegv_still_flags(self):
    # GUARD AGAINST OVER-FIXING: a real crash line wraps the token in
    # punctuation, e.g. 'Fatal ... (SIGSEGV)'. The boundary is alnum-only, so
    # this MUST still match -- the fix narrows false positives, not true ones.
    self.assertIsNotNone(
        R.looks_like_code_bug('Fatal Python error: Segmentation fault (SIGSEGV)'))
    self.assertIsNotNone(R.looks_like_code_bug('caught signal 11 (SIGSEGV), dumping'))

  def test_traceback_signature_with_trailing_colon_still_flags(self):
    # GUARD: 'TRACEBACK (MOST RECENT CALL LAST)' ends in ')', and in real logs
    # is followed by ':'. A naive \b right-boundary would REGRESS here; our
    # alnum-only boundary keeps it matching.
    self.assertIsNotNone(R.looks_like_code_bug(
        'Traceback (most recent call last):\n  File a.py'))
  def test_backported_fatal_signals_flag(self):
    # 2026-09-18 twin-table sync: SIGILL/SIGBUS/SIGFPE were missing from the
    # route_lib copy, so a healthy run that trained a little then took one of
    # these with no checkpoint could slip past the code-bug gate and (once
    # cold-rerun is armed) be replayed into the same fault. They must flag.
    self.assertIsNotNone(R.looks_like_code_bug('Fatal: caught signal 4 (SIGILL)'))
    self.assertIsNotNone(R.looks_like_code_bug('worker died with signal 7'))
    self.assertIsNotNone(R.looks_like_code_bug('rank 3 got signal 8 (SIGFPE)'))

  def test_backported_exception_signatures_flag(self):
    self.assertIsNotNone(R.looks_like_code_bug('AttributeError: no attr foo'))
    self.assertIsNotNone(R.looks_like_code_bug('jaxlib JaxRuntimeError: bad'))
    self.assertIsNotNone(R.looks_like_code_bug('PermissionError: denied'))
    self.assertIsNotNone(
        R.looks_like_code_bug("OSError: Permission denied: '/cns/si-d/x'"))
    self.assertIsNotNone(
        R.looks_like_code_bug('NOT_FOUND: Could not find /cns/si-d/data'))
    self.assertIsNotNone(R.looks_like_code_bug('launcher: unrecoverable failure'))
    self.assertIsNotNone(R.looks_like_code_bug('main exited with non-zero status'))

  def test_omitted_signatures_stay_banner_safe(self):
    # DELIBERATE OMISSIONS from the twin table (see _CODE_BUG_SIGNATURES): these
    # three appear BENIGNLY in a healthy boot banner / device-info line, and the
    # tail of a short log IS the banner. They must NOT be classed as code bugs,
    # or every healthy pruned run gets HELD -- the regression this path exists
    # to prevent. This test FAILS if someone naively re-adds them.
    self.assertIsNone(R.looks_like_code_bug(
        "minloglevel readback (ModuleNotFoundError: No module named 'base')"))
    self.assertIsNone(R.looks_like_code_bug(
        'note: ImportError fallback path taken for optional dep'))
    self.assertIsNone(R.looks_like_code_bug(
        'GPU0 HBM memory limit 40.0 GiB; 8 devices visible'))



class OutDirFromLogTest(unittest.TestCase):

  def test_post_locality_line_wins(self):
    log = ("[parcae-torch] locality: out_dir /cns/is-d/x -> /cns/si-d/x\n"
           "[parcae-torch] out_dir (post-locality) = '/cns/si-d/home/q/run'\n"
           "[parcae-torch] step 1")
    self.assertEqual(R.out_dir_from_log(log), '/cns/si-d/home/q/run')

  def test_checkpoint_saved_fallback(self):
    log = '[parcae-torch] step 1024 checkpoint saved -> /cns/si-d/home/q/run/steps/step_1024.pt'
    self.assertEqual(R.out_dir_from_log(log), '/cns/si-d/home/q/run')

  def test_none_when_no_path(self):
    self.assertIsNone(R.out_dir_from_log('[parcae-torch] step 1 loss 10.6'))
    self.assertIsNone(R.out_dir_from_log(''))


class LiveConfigSiblingTest(unittest.TestCase):

  def _c(self, jid, cfg, state):
    return _entry(job_id=jid, launch_kwargs={'config': cfg}, state=state)

  def test_same_config_live_is_sibling(self):
    dead = self._c('a', 'cfgX', R.JobState.FAILED)
    live = self._c('b', 'cfgX', R.JobState.RUNNING)
    self.assertTrue(R.has_live_config_sibling(dead, [dead, live]))

  def test_same_config_but_dead_is_not_sibling(self):
    dead = self._c('a', 'cfgX', R.JobState.FAILED)
    other_dead = self._c('b', 'cfgX', R.JobState.DONE)
    self.assertFalse(R.has_live_config_sibling(dead, [dead, other_dead]))

  def test_different_config_is_not_sibling(self):
    dead = self._c('a', 'cfgX', R.JobState.FAILED)
    live = self._c('b', 'cfgY', R.JobState.RUNNING)
    self.assertFalse(R.has_live_config_sibling(dead, [dead, live]))

  def test_entry_does_not_count_itself(self):
    dead = self._c('a', 'cfgX', R.JobState.RUNNING)  # even if it were live
    self.assertFalse(R.has_live_config_sibling(dead, [dead]))

  def test_no_config_is_never_sibling(self):
    dead = _entry(job_id='a', launch_kwargs={}, state=R.JobState.FAILED)
    live = self._c('b', 'cfgX', R.JobState.RUNNING)
    self.assertFalse(R.has_live_config_sibling(dead, [dead, live]))


class BuildWarmRestartEntryTest(unittest.TestCase):

  def _dead(self, **kw):
    base = dict(
        job_id='h100-8-dead', power='h100-8', archs=('h100',),
        tier='PROD', allowed_metros=['sin', 'cbf'], state=R.JobState.FAILED,
        xid='288098495', auto_resumes=0,
        launch_kwargs={'config': 'cfgX', 'exp_name': 'parcae-dw', 'group': '9'})
    base.update(kw)
    return _entry(**base)

  def test_clones_spec_and_sets_load_from(self):
    e = R.build_warm_restart_entry(
        self._dead(), '/cns/si-d/x/steps/step_1024.pt', 'h100-8-new01')
    self.assertEqual(e.job_id, 'h100-8-new01')
    self.assertEqual(e.power, 'h100-8')
    self.assertEqual(e.allowed_archs, ['h100'])
    self.assertEqual(e.tier, 'PROD')
    self.assertEqual(e.allowed_metros, ['sin', 'cbf'])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.launch_kwargs['load_from'],
                     '/cns/si-d/x/steps/step_1024.pt')
    self.assertEqual(e.launch_kwargs['config'], 'cfgX')  # same run

  def test_increments_auto_resumes_and_names_attempt(self):
    e = R.build_warm_restart_entry(
        self._dead(auto_resumes=1), '/cns/x/steps/step_1024.pt', 'j2')
    self.assertEqual(e.auto_resumes, 2)
    self.assertEqual(e.launch_kwargs['exp_name'], 'parcae-dw-r2')

  def test_suffix_does_not_stack(self):
    dead = self._dead(auto_resumes=2,
                      launch_kwargs={'config': 'cfgX', 'exp_name': 'parcae-dw-r2'})
    e = R.build_warm_restart_entry(dead, '/cns/x/steps/step_1024.pt', 'j3')
    self.assertEqual(e.launch_kwargs['exp_name'], 'parcae-dw-r3')

  def test_records_prior_xid(self):
    e = R.build_warm_restart_entry(
        self._dead(), '/cns/x/steps/step_1024.pt', 'j2')
    self.assertIn('288098495', e.prior_xids)

  def test_does_not_mutate_dead_entry(self):
    dead = self._dead()
    R.build_warm_restart_entry(dead, '/cns/x/steps/step_1024.pt', 'j2')
    self.assertEqual(dead.auto_resumes, 0)
    self.assertNotIn('load_from', dead.launch_kwargs)


class BuildColdRestartEntryTest(unittest.TestCase):
  """The COLD twin: a healthy run preempted before its first checkpoint reruns
  from step 0 -- clones the spec, wires NO resume pointer, still bounded by the
  auto_resume budget."""

  def _dead(self, **kw):
    base = dict(
        job_id='h100-8-dead', power='h100-8', archs=('h100',),
        tier='PROD', allowed_metros=['sin', 'cbf'], state=R.JobState.FAILED,
        xid='288098495', auto_resumes=0,
        launch_kwargs={'config': 'cfgX', 'exp_name': 'parcae-dw', 'group': '9'})
    base.update(kw)
    return _entry(**base)

  def test_clones_spec_and_sets_no_resume_pointer(self):
    e = R.build_cold_restart_entry(self._dead(), 'h100-8-new01')
    self.assertEqual(e.job_id, 'h100-8-new01')
    self.assertEqual(e.power, 'h100-8')
    self.assertEqual(e.allowed_archs, ['h100'])
    self.assertEqual(e.tier, 'PROD')
    self.assertEqual(e.allowed_metros, ['sin', 'cbf'])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.launch_kwargs['config'], 'cfgX')  # same run
    # THE POINT: cold means NO resume pointer of any kind.
    self.assertNotIn('load_from', e.launch_kwargs)
    self.assertNotIn('restart_from', e.launch_kwargs)
    self.assertNotIn('restart_step', e.launch_kwargs)

  def test_strips_stale_resume_pointer_from_cloned_spec(self):
    # If the dead row somehow carried a resume pointer, cold must drop it.
    dead = self._dead(launch_kwargs={
        'config': 'cfgX', 'exp_name': 'parcae-dw', 'load_from': '/cns/old/step_1.pt',
        'restart_from': '/cns/old', 'restart_step': '5'})
    e = R.build_cold_restart_entry(dead, 'j2')
    self.assertNotIn('load_from', e.launch_kwargs)
    self.assertNotIn('restart_from', e.launch_kwargs)
    self.assertNotIn('restart_step', e.launch_kwargs)

  def test_increments_auto_resumes_and_names_attempt(self):
    e = R.build_cold_restart_entry(self._dead(auto_resumes=1), 'j2')
    self.assertEqual(e.auto_resumes, 2)
    self.assertEqual(e.launch_kwargs['exp_name'], 'parcae-dw-r2')

  def test_records_prior_xid(self):
    e = R.build_cold_restart_entry(self._dead(), 'j2')
    self.assertIn('288098495', e.prior_xids)

  def test_does_not_mutate_dead_entry(self):
    dead = self._dead()
    R.build_cold_restart_entry(dead, 'j2')
    self.assertEqual(dead.auto_resumes, 0)


class PackageDirTest(unittest.TestCase):
  """package_dir picks the enqueue snapshot over the live workdir, else falls
  back to workdir, else ''."""

  def test_snapshot_wins_over_workdir(self):
    e = _entry(workdir='/live/checkout', snapshot_dir='/snap/j1')
    self.assertEqual(R.package_dir(e), '/snap/j1')

  def test_falls_back_to_workdir_when_no_snapshot(self):
    e = _entry(workdir='/live/checkout')
    self.assertEqual(R.package_dir(e), '/live/checkout')

  def test_empty_when_both_absent(self):
    e = _entry()
    self.assertEqual(R.package_dir(e), '')

  def test_whitespace_is_not_a_path(self):
    e = _entry(workdir='   ', snapshot_dir='   ')
    self.assertEqual(R.package_dir(e), '')

  def test_old_row_without_field_defaults_empty_and_uses_workdir(self):
    # An entry deserialized from a pre-feature queue file has no snapshot_dir
    # key; from_dict defaults it to '' and package_dir must use workdir.
    d = _entry(workdir='/live/checkout').to_dict()
    d.pop('snapshot_dir', None)
    e = R.QueueEntry.from_dict(d)
    self.assertEqual(e.snapshot_dir, '')
    self.assertEqual(R.package_dir(e), '/live/checkout')

  def test_serde_round_trips_snapshot_dir(self):
    e = _entry(workdir='/w', snapshot_dir='/snap/j1')
    e2 = R.QueueEntry.from_dict(e.to_dict())
    self.assertEqual(e2.snapshot_dir, '/snap/j1')


class PartitionForArchiveTest(unittest.TestCase):
  """`tpu clear`'s queue-cascade safety logic: archive only FINISHED rows the
  requested XIDs point at; refuse live ones; match on current OR historical XID."""

  def _row(self, job_id, state, xid=None, prior=()):
    e = _entry(job_id=job_id)
    e.state = state
    e.xid = xid
    e.prior_xids = list(prior)
    return e

  def test_finished_states_is_only_done_and_failed(self):
    # The whole safety story rests on RUNNING being EXCLUDED here.
    self.assertEqual(R.FINISHED_STATES,
                     frozenset({R.JobState.DONE, R.JobState.FAILED}))
    self.assertNotIn(R.JobState.RUNNING, R.FINISHED_STATES)

  def test_done_row_is_archived(self):
    rows = [self._row('j1', R.JobState.DONE, xid='111')]
    part = R.partition_for_archive(rows, {'111'})
    self.assertEqual([e.job_id for e in part.to_archive], ['j1'])
    self.assertEqual(part.refused, [])
    self.assertEqual(part.matched_xids, {'111'})

  def test_failed_row_is_archived(self):
    rows = [self._row('j1', R.JobState.FAILED, xid='111')]
    part = R.partition_for_archive(rows, {'111'})
    self.assertEqual([e.job_id for e in part.to_archive], ['j1'])

  def test_running_row_is_refused_never_archived(self):
    # The core invariant: a live row is the router's handle on a job on the
    # cluster. Archiving it would strand the work -- so it must be REFUSED.
    rows = [self._row('j1', R.JobState.RUNNING, xid='111')]
    part = R.partition_for_archive(rows, {'111'})
    self.assertEqual(part.to_archive, [])
    self.assertEqual(part.refused, [('j1', 'RUNNING', '111')])
    self.assertEqual(part.matched_xids, {'111'})

  def test_every_live_state_is_refused(self):
    live = [R.JobState.QUEUED, R.JobState.BUILD_REQUESTED, R.JobState.BUILDING,
            R.JobState.BUDGET_DEFERRED, R.JobState.SUBMITTED, R.JobState.RUNNING,
            R.JobState.HELD]
    for st in live:
      rows = [self._row('j', st, xid='999')]
      part = R.partition_for_archive(rows, {'999'})
      self.assertEqual(part.to_archive, [], f'{st} must not be archived')
      self.assertEqual(len(part.refused), 1, f'{st} must be refused')

  def test_matches_on_prior_xid_not_only_current(self):
    # A row re-dispatched once carries its old XID in prior_xids; clearing that
    # old XID must still find and archive the (finished) row.
    rows = [self._row('j1', R.JobState.DONE, xid='222', prior=('111',))]
    part = R.partition_for_archive(rows, {'111'})
    self.assertEqual([e.job_id for e in part.to_archive], ['j1'])
    self.assertEqual(part.matched_xids, {'111'})

  def test_unmatched_xid_is_not_reported_as_matched(self):
    rows = [self._row('j1', R.JobState.DONE, xid='111')]
    part = R.partition_for_archive(rows, {'nope'})
    self.assertEqual(part.to_archive, [])
    self.assertEqual(part.refused, [])
    self.assertEqual(part.matched_xids, set())

  def test_unplaced_row_with_no_xid_never_matches(self):
    # xid=None must not collide with another id via a bogus '' match.
    rows = [self._row('j1', R.JobState.QUEUED, xid=None)]
    part = R.partition_for_archive(rows, {''})
    self.assertEqual(part.matched_xids, set())
    self.assertEqual(part.refused, [])

  def test_mixed_batch_splits_correctly(self):
    rows = [
        self._row('done1', R.JobState.DONE, xid='1'),
        self._row('run1', R.JobState.RUNNING, xid='2'),
        self._row('fail1', R.JobState.FAILED, xid='3'),
        self._row('other', R.JobState.DONE, xid='4'),  # not requested
    ]
    part = R.partition_for_archive(rows, {'1', '2', '3'})
    self.assertEqual({e.job_id for e in part.to_archive}, {'done1', 'fail1'})
    self.assertEqual(part.refused, [('run1', 'RUNNING', '2')])
    self.assertEqual(part.matched_xids, {'1', '2', '3'})

  def test_entry_xids_helper(self):
    e = self._row('j', R.JobState.DONE, xid='cur', prior=('a', 'b'))
    self.assertEqual(R.entry_xids(e), {'cur', 'a', 'b'})
    e2 = self._row('j', R.JobState.QUEUED, xid=None)
    self.assertEqual(R.entry_xids(e2), set())


class MintJobIdTest(unittest.TestCase):
  """The opaque job_id minter (§5.1): time-ordered prefix + random suffix, so
  ids sort by creation, are greppable, and do not collide in practice -- which
  is what makes "immutable + never-reused" structural rather than a history
  scan."""

  def test_shape_matches_the_documented_format(self):
    jid = R.mint_job_id(now=1789500000.0)
    self.assertRegex(jid, R._JOB_ID_RE)
    self.assertRegex(jid, r'^\d{8}T\d{6}-[0-9a-f]{10}$')

  def test_time_prefix_is_the_given_instant_and_sorts_by_creation(self):
    early = R.mint_job_id(now=1789500000.0)
    late = R.mint_job_id(now=1789600000.0)
    # lexical sort == chronological, because the prefix is a zero-padded stamp
    self.assertLess(early.split('-')[0], late.split('-')[0])

  def test_suffix_is_random_across_mints_in_the_same_second(self):
    # Same instant -> same prefix, but the 40-bit suffix must differ, or two
    # jobs enqueued in one second would collide. Probability of a dup here is
    # ~n^2/2^41; 200 draws is astronomically safe, so a hit means a real bug.
    ids = {R.mint_job_id(now=1789500000.0) for _ in range(200)}
    self.assertEqual(len(ids), 200)

  def test_not_derived_from_any_name(self):
    # The whole point of the split: the id carries no label, so re-running a
    # job under the same --job_name later yields a brand-new id.
    a = R.mint_job_id(now=1789500000.0)
    b = R.mint_job_id(now=1789500000.0)
    self.assertNotEqual(a, b)


class ResolveJobRefsTest(unittest.TestCase):
  """The CLI inverse of the opaque id: a person types a readable name, a full
  id, or a git-style id SUFFIX, and we map it to row(s) -- refusing an
  AMBIGUOUS token instead of guessing which arm to act on (§5.4 discipline)."""

  def _rows(self):
    return [
        _entry(job_id='20260916T120000-aaaaaaaaaa', name='parcae-torch'),
        _entry(job_id='20260916T130000-bbbbbbbbbb', name='parcae-jax'),
        _entry(job_id='20260916T140000-abcabcabca', name='eqr-run'),
    ]

  def test_exact_id_wins(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['20260916T120000-aaaaaaaaaa'])
    self.assertEqual(matched, {'20260916T120000-aaaaaaaaaa'})
    self.assertEqual(errs, [])

  def test_readable_name_resolves(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['eqr-run'])
    self.assertEqual(matched, {'20260916T140000-abcabcabca'})
    self.assertEqual(errs, [])

  def test_unique_id_suffix_resolves_git_style(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['aaaaaaaaaa'])
    self.assertEqual(matched, {'20260916T120000-aaaaaaaaaa'})
    self.assertEqual(errs, [])

  def test_ambiguous_suffix_fails_closed(self):
    # Two ids ending in the same short tail: resolve NEITHER, and say so. The
    # failure direction that matters -- never dequeue/cancel a guessed arm.
    rows = [
        _entry(job_id='20260916T120000-0000000abc', name='one'),
        _entry(job_id='20260916T130000-1111111abc', name='two'),
    ]
    matched, errs = R.resolve_job_refs(rows, ['abc'])
    self.assertEqual(matched, set())
    self.assertEqual(len(errs), 1)
    self.assertIn('ambiguous', errs[0])

  def test_duplicate_name_across_two_rows_fails_closed(self):
    # Should not happen for LIVE rows (enqueue enforces name-uniqueness), but if
    # it ever does, a name that hits >1 row resolves to none, not a guess.
    rows = [
        _entry(job_id='20260916T120000-aaaaaaaaaa', name='dup'),
        _entry(job_id='20260916T130000-bbbbbbbbbb', name='dup'),
    ]
    matched, errs = R.resolve_job_refs(rows, ['dup'])
    self.assertEqual(matched, set())
    self.assertEqual(len(errs), 1)
    self.assertIn('names 2 rows', errs[0])

  def test_unknown_token_is_reported_not_silently_dropped(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['nope'])
    self.assertEqual(matched, set())
    self.assertEqual(len(errs), 1)
    self.assertIn('matched no job', errs[0])

  def test_name_beats_suffix_when_both_could_match(self):
    # A token that is BOTH a row's exact name and another row's id suffix must
    # resolve by name (tier 2 before tier 3), deterministically.
    rows = [
        _entry(job_id='20260916T120000-000000eqr0', name='other'),
        _entry(job_id='20260916T130000-bbbbbbbbbb', name='eqr0'),
    ]
    matched, errs = R.resolve_job_refs(rows, ['eqr0'])
    self.assertEqual(matched, {'20260916T130000-bbbbbbbbbb'})
    self.assertEqual(errs, [])

  def test_mixed_batch_good_and_bad_tokens(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['parcae-torch', 'nope', 'eqr-run'])
    self.assertEqual(
        matched,
        {'20260916T120000-aaaaaaaaaa', '20260916T140000-abcabcabca'})
    self.assertEqual(len(errs), 1)  # only 'nope'

  def test_empty_and_whitespace_tokens_are_skipped(self):
    rows = self._rows()
    matched, errs = R.resolve_job_refs(rows, ['', '  ', 'eqr-run'])
    self.assertEqual(matched, {'20260916T140000-abcabcabca'})
    self.assertEqual(errs, [])


class ApplyWarmRestartInPlaceTest(unittest.TestCase):
  """The IN-PLACE warm-restart the reroute path uses: the SAME row is kept
  (job_id, backoff counter, cooldowns), and only the resume pointer is wired in,
  layout-correctly (ELT restart_from vs torch load_from). The twin of
  build_warm_restart_entry (which mints a fresh row for the reconcile path); both
  share _apply_resume_pointer, so the silently-destructive layout trap is decided
  in one place."""

  def _row(self, **kw):
    base = dict(
        job_id='j-keep', power='h100-8', archs=('h100',),
        tier='PROD', state=R.JobState.QUEUED, auto_resumes=0,
        launch_kwargs={'config': 'cfgX', 'exp_name': 'parcae-dw'})
    base.update(kw)
    return _entry(**base)

  def test_torch_layout_sets_load_from_and_clears_restart(self):
    # A torch `step_<N>.pt` leaf resumes via $LOAD_FROM; any stale ELT keys must
    # be cleared (both set trips main_eqr's guard).
    e = self._row(launch_kwargs={'config': 'cfgX', 'exp_name': 'dw',
                                 'restart_from': '/old/wd', 'restart_step': '10'})
    R.apply_warm_restart_in_place(e, '/cns/x/steps/step_2048.pt')
    self.assertEqual(e.launch_kwargs['load_from'], '/cns/x/steps/step_2048.pt')
    self.assertNotIn('restart_from', e.launch_kwargs)
    self.assertNotIn('restart_step', e.launch_kwargs)

  def test_elt_layout_sets_restart_from_step_and_clears_load_from(self):
    # NEGATIVE CONTROL for the destructive trap: an ELT checkpoints/<int> leaf
    # must resume via restart_from+restart_step and NEVER via load_from (which
    # would make orbax prune the checkpoints it resumed from).
    e = self._row(launch_kwargs={'config': 'cfgX', 'exp_name': 'dw',
                                 'load_from': '/stale/leaf'})
    R.apply_warm_restart_in_place(e, '/cns/x/wd/checkpoints/1536')
    self.assertEqual(e.launch_kwargs['restart_from'], '/cns/x/wd')
    self.assertEqual(e.launch_kwargs['restart_step'], '1536')
    self.assertNotIn('load_from', e.launch_kwargs)

  def test_keeps_row_identity_and_bumps_auto_resumes(self):
    e = self._row(auto_resumes=1)
    R.apply_warm_restart_in_place(e, '/cns/x/steps/step_1.pt')
    self.assertEqual(e.job_id, 'j-keep')     # SAME row, not a fresh one
    self.assertEqual(e.power, 'h100-8')
    self.assertEqual(e.auto_resumes, 2)      # shared budget bumped

  def test_exp_name_suffix_does_not_stack(self):
    e = self._row(auto_resumes=2, launch_kwargs={'exp_name': 'dw-r2'})
    R.apply_warm_restart_in_place(e, '/cns/x/steps/step_1.pt')
    self.assertEqual(e.launch_kwargs['exp_name'], 'dw-r3')

  def test_exp_name_falls_back_to_name_then_job_id(self):
    e = self._row(job_id='jx', launch_kwargs={})
    e.name = 'readable-name'
    R.apply_warm_restart_in_place(e, '/cns/x/steps/step_1.pt')
    self.assertEqual(e.launch_kwargs['exp_name'], 'readable-name-r1')

  def test_elt_layout_does_not_rollback_higher_restart_step(self):
    e = self._row(launch_kwargs={
        'config': 'cfgX', 'exp_name': 'dw-r3',
        'restart_from': '/cns/newer/wd', 'restart_step': '35000',
    })
    R.apply_warm_restart_in_place(e, '/cns/older/wd/checkpoints/25001')
    self.assertEqual(e.launch_kwargs['restart_from'], '/cns/newer/wd')
    self.assertEqual(e.launch_kwargs['restart_step'], '35000')


class ClassifyFailureTest(unittest.TestCase):
  """Tests for the AUTO-RESUME RULE SET in route_lib.classify_failure."""

  def test_affinity_group_in_use_resumes(self):
    verdict, why = R.classify_failure(
        'FAILED_PRECONDITION: AffinityGroup name: "nk_qiaos.1/qiaos" is still in'
        ' use... { error: AFFINITY_GROUP_IN_USE }'
    )
    self.assertEqual(verdict, R.RESUME_XID)
    self.assertIn('affinity group', why.lower())

  def test_affinity_group_verdict_phrase_resumes(self):
    verdict, why = R.classify_failure(
        'Borg AffinityGroup in use (transient conflict; retry/reroute)'
    )
    self.assertEqual(verdict, R.RESUME_XID)
    self.assertIn('affinity group', why.lower())

  def test_task_action_fish_ici_resumes(self):
    verdict, why = R.classify_failure(
        'Borg task failed: TASK_ACTION_FISH_ICI_SECURITY_SETUP'
    )
    self.assertEqual(verdict, R.RESUME_XID)
    self.assertIn('ici', why.lower())

  def test_ici_setup_failure_phrase_resumes(self):
    verdict, why = R.classify_failure(
        'TPU ICI setup failure (hardware/bad node; reroute)'
    )
    self.assertEqual(verdict, R.RESUME_XID)
    self.assertIn('ici', why.lower())

  def test_zero_work_unit_holds(self):
    verdict, why = R.classify_failure(
        'reconciled: XM resolves the id but reports ZERO work units; experiment'
        ' is gone'
    )
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('work unit', why.lower())

  def test_unrecognised_reason_defaults_to_hold(self):
    verdict, why = R.classify_failure('some random unrecognised error')
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('unrecognised', why.lower())

  def test_empty_reason_defaults_to_hold(self):
    verdict, why = R.classify_failure('')
    self.assertEqual(verdict, R.HOLD)
    self.assertIn('cannot classify', why.lower())


class PlanOneWhyNotTest(unittest.TestCase):
  """plan_one's one-line why-not for a row it cannot place (shown by the
  worker's requeue, `tpu check` and `tpu queue-status`). Diagnostic only: the
  placement decision itself must not change."""

  def test_no_free_slice(self):
    e = _entry(power='v7-32', archs=('v7',))
    self.assertIsNone(R.plan_one(e, {'c7': _avail('c7', 'v7', free=0)},
                                 now=0.0))
    self.assertEqual(e.last_filter_reason, 'v7-32: no free slice')

  def test_oversold_reads_as_no_free_slice(self):
    e = _entry(power='v7-32', archs=('v7',))
    self.assertIsNone(R.plan_one(
        e, {'c7': _avail('c7', 'v7', free=320, oversold=True)}, now=0.0))
    self.assertEqual(e.last_filter_reason, 'v7-32: no free slice')

  def test_metro_filter_is_named(self):
    e = _entry(power='v7-32', archs=('v7',), allowed_metros=['cbf'])
    self.assertIsNone(R.plan_one(
        e, {'c7': _avail('c7', 'v7', free=320, metro='kul')}, now=0.0))
    self.assertEqual(e.last_filter_reason, 'v7-32: no free slice in cbf')

  def test_one_note_per_arch_in_the_order_tried(self):
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    self.assertIsNone(R.plan_one(e, {}, now=0.0))
    self.assertEqual(e.last_filter_reason,
                     'v7-32: no free slice | v6p-32: no free slice')

  def test_arch_with_several_shapes_is_named_once(self):
    e = _entry(power='v7-32', archs=('v7',), power_tolerance=1.0)
    self.assertGreater(len(R.candidate_shapes(e)), 1)      # precondition
    self.assertIsNone(R.plan_one(e, {}, now=0.0))
    self.assertEqual(e.last_filter_reason.count('v7-'), 1)

  def test_over_the_limit_order_cap_is_named(self):
    e = _entry(power='v7-32', archs=('v7',))
    self.assertIsNone(R.plan_one(
        e, {'c7': _avail('c7', 'v7', free=320, price=25.0)}, now=0.0))
    self.assertEqual(e.last_filter_reason,
                     'v7-32: 1 cell(s) over limit-order cap 20')

  def test_no_group_storage_is_named(self):
    e = _entry(power='v7-32', archs=('v7',))
    avail = {f'c{i}': _avail(f'c{i}', 'v7', free=320, metro='nostore')
             for i in range(5)}
    with mock.patch.object(R, '_metro_has_group_storage',
                           lambda m: m != 'nostore'):
      self.assertIsNone(R.plan_one(e, avail, now=0.0))
    self.assertEqual(
        e.last_filter_reason,
        'v7-32: 5 cell(s) in metros without group storage (c0,c1,c2,...)')

  def test_no_accepted_shape_is_named(self):
    # A locked v6p-32 (2x4x4) matches no v6e mesh, so no shape is tried.
    e = _entry(power='v6p-32', archs=('v6e',), topology_locked=True)
    self.assertEqual(R.candidate_shapes(e), [])             # precondition
    self.assertIsNone(R.plan_one(e, {'c': _avail('c', 'v6e', 640)}, now=0.0))
    self.assertTrue(e.last_filter_reason.startswith(
        'no accepted shape: power v6p-32 fits none of archs v6e'),
                    e.last_filter_reason)
    self.assertIn('topology lock', e.last_filter_reason)

  def test_reset_on_every_call_and_empty_on_placement(self):
    e = _entry(power='v7-32', archs=('v7',))
    e.last_filter_reason = 'stale verdict from an earlier pass'
    _ok(R.plan_one(e, {'c7': _avail('c7', 'v7', free=320)}, now=0.0))
    self.assertEqual(e.last_filter_reason, '')
    e.last_filter_reason = 'stale verdict from an earlier pass'
    self.assertIsNone(R.plan_one(e, {}, now=0.0))
    self.assertEqual(e.last_filter_reason, 'v7-32: no free slice')

  def test_cooldown_fallback_placement_leaves_no_reason(self):
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    e.cooldown_pairs = {'c7|v7': {'until': 1e12, 'strikes': 1}}
    p = _ok(R.plan_one(e, {'c7': _avail('c7', 'v7', free=320)}, now=0.0))
    self.assertEqual(p.cell, 'c7')
    self.assertEqual(e.last_filter_reason, '')

  def test_length_is_bounded(self):
    s = R._unplaced_reason(_entry(), ['x' * 100] * 5)
    self.assertEqual(len(s), R._FILTER_REASON_MAX_CHARS)
    self.assertTrue(s.endswith('...'))

  def test_select_and_plan_leaves_a_reason_on_each_unplaced_row(self):
    a = _entry('a', power='v7-32', archs=('v7',))
    b = _entry('b', power='v7-32', archs=('v7',))
    got = R.select_and_plan([a, b], {'c7': _avail('c7', 'v7', free=32)},
                            now=0.0, rng=random.Random(0))
    self.assertEqual(len(got), 1)                 # one slice for two rows
    placed = {got[0].job_id}
    for e in (a, b):
      if e.job_id in placed:
        self.assertEqual(e.last_filter_reason, '')
      else:
        self.assertEqual(e.last_filter_reason, 'v7-32: no free slice')


# ---------------------------------------------------------------------------
# Group cooldown + "can this pool hold the job" (operator 2026-09-23: "和对tpu
# type / cell做冷却时一样的逻辑，再次之外加一个'g5能不能用'的判断，如果不能就不route到g5").


def _gcap(floor=None, used=None, balance=0.0, burn=0.0, caps=None, age=10.0,
          prices=None, group='5'):
  return R.GroupCapacity(group=group, floor=dict(floor or {}),
                         used=dict(used or {}), balance=balance,
                         above_floor_burn=burn, limit_caps=dict(caps or {}),
                         age_s=age, prices=dict(prices or {}))


class GroupCooldownTest(unittest.TestCase):
  """The group cooldown is recorded EXACTLY like the arch cooldown:
  {'until', 'strikes'}, strikes stacking while the window is open and resetting
  once it has fully elapsed."""

  def test_stamp_first_strike(self):
    e = _entry()
    R.stamp_group_cooldown(e, '5', now=100.0, cooldown_s=60.0)
    self.assertEqual(e.cooldown_groups, {'5': {'until': 160.0, 'strikes': 1}})

  def test_strikes_stack_while_the_window_is_open(self):
    e = _entry()
    R.stamp_group_cooldown(e, '5', now=100.0, cooldown_s=60.0)
    R.stamp_group_cooldown(e, '5', now=150.0, cooldown_s=60.0)
    self.assertEqual(e.cooldown_groups['5'], {'until': 210.0, 'strikes': 2})

  def test_strikes_reset_once_the_window_has_elapsed(self):
    e = _entry()
    R.stamp_group_cooldown(e, '5', now=100.0, cooldown_s=60.0)
    R.stamp_group_cooldown(e, '5', now=161.0, cooldown_s=60.0)
    self.assertEqual(e.cooldown_groups['5'], {'until': 221.0, 'strikes': 1})

  def test_no_group_is_a_noop(self):
    e = _entry()
    R.stamp_group_cooldown(e, None, now=100.0, cooldown_s=60.0)
    R.stamp_group_cooldown(e, '  ', now=100.0, cooldown_s=60.0)
    self.assertEqual(e.cooldown_groups, {})

  def test_group_cooling_live_expired_other_group(self):
    e = _entry()
    R.stamp_group_cooldown(e, '5', now=100.0, cooldown_s=60.0)
    self.assertIsNotNone(R.group_cooling(e, '5', now=159.0))
    self.assertIsNone(R.group_cooling(e, '5', now=160.0))
    self.assertIsNone(R.group_cooling(e, '3', now=120.0))

  def test_group_cooling_tolerates_unreadable_records(self):
    e = _entry()
    e.cooldown_groups = {'9': 12345.0, '7': {'until': 'soon'}, '6': {}}
    for g in ('9', '7', '6'):
      self.assertIsNone(R.group_cooling(e, g, now=1.0))

  def test_old_row_without_the_field_is_not_cooling(self):
    e = R.QueueEntry.from_dict(
        {'job_id': 'old', 'power': 'v7-32', 'allowed_archs': ['v7']})
    self.assertEqual(e.cooldown_groups, {})
    self.assertIsNone(R.group_cooling(e, '5', now=1.0))

  def test_serde_roundtrip_keeps_cooldown_groups(self):
    e = _entry()
    R.stamp_group_cooldown(e, '5', now=100.0, cooldown_s=60.0)
    back = R.QueueEntry.from_dict(e.to_dict())
    self.assertEqual(back.cooldown_groups, {'5': {'until': 160.0, 'strikes': 1}})

  def test_mark_reroute_cools_the_group_the_submission_ran_under(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.group = '9'
    e.open_creating('111', cell='yulpptr', arch='v7', chips=32, group='5',
                    now=0.0)
    e.cell, e.arch, e.chips, e.submitted_at = 'yulpptr', 'v7', 32, 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.state, R.JobState.QUEUED)
    # The submission's own group wins over the row's.
    self.assertEqual(e.cooldown_groups, {'5': {'until': 2500.0, 'strikes': 1}})

  def test_mark_reroute_falls_back_to_the_row_group(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell, e.submitted_at = 'yulpptr', 0.0
    e.xid = '12345'          # v1-style submission: carries no group
    e.group = '5'
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.cooldown_groups, {'5': {'until': 2500.0, 'strikes': 1}})

  def test_repeated_reroutes_off_the_same_group_stack(self):
    e = _entry()
    e.group = '5'
    for i, t in enumerate((700.0, 800.0, 900.0)):
      e.state = R.JobState.SUBMITTED
      e.cell, e.submitted_at = f'cell{i}', t - 100.0
      e.xid = str(1000 + i)
      R.mark_reroute(e, now=t, cooldown_s=1800.0)
    self.assertEqual(e.cooldown_groups['5'], {'until': 2700.0, 'strikes': 3})

  def test_mark_reroute_without_any_group_stamps_nothing(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell, e.submitted_at = 'yulpptr', 0.0
    e.xid = '1'
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.cooldown_groups, {})


class GroupCanHoldTest(unittest.TestCase):
  """group_can_hold: can a NON-fallback pool actually hold the job? Fails
  closed at every unknown."""

  def test_no_data_is_no(self):
    v = R.group_can_hold(None, 'h100', 8, 1.0)
    self.assertFalse(v.ok)
    self.assertIn('no capacity data', v.reason)

  def test_stale_data_is_no(self):
    cap = _gcap(floor={'h100': 64}, age=R.GROUP_CAPACITY_MAX_AGE_S + 1.0)
    v = R.group_can_hold(cap, 'h100', 8, 1.0)
    self.assertFalse(v.ok)
    self.assertIn('old', v.reason)

  def test_unknown_shape_is_no(self):
    cap = _gcap(floor={'h100': 64})
    self.assertFalse(R.group_can_hold(cap, None, 8, 1.0).ok)
    self.assertFalse(R.group_can_hold(cap, 'h100', None, 1.0).ok)
    self.assertFalse(R.group_can_hold(cap, 'h100', 0, 1.0).ok)

  def test_floor_room_admits_with_zero_balance_and_no_price(self):
    cap = _gcap(floor={'h100': 48}, used={'h100': 40}, balance=0.0)
    v = R.group_can_hold(cap, 'h100', 8, None)
    self.assertTrue(v.ok)
    self.assertFalse(v.above_floor)

  def test_the_2026_09_23_g5_shape_is_refused(self):
    # g5 at ~22:15Z: H100 floor 48/48 used, B200 24 chips above its floor
    # (~48 cr/hr burn), balance ~24.
    cap = _gcap(floor={'h100': 48}, used={'h100': 48}, balance=24.0,
                burn=48.0, prices={'h100': 0.531})
    v = R.group_can_hold(cap, 'h100', 8, 0.531)
    self.assertFalse(v.ok)
    self.assertIn('no h100 floor room', v.reason)
    self.assertIn('balance 24', v.reason)

  def test_pending_chips_eat_floor_room(self):
    cap = _gcap(floor={'h100': 48}, used={'h100': 32})
    self.assertTrue(R.group_can_hold(cap, 'h100', 16, None).ok)
    self.assertFalse(
        R.group_can_hold(cap, 'h100', 16, None, pending={'h100': 8}).ok)
    self.assertTrue(      # pending chips of ANOTHER family leave h100 alone
        R.group_can_hold(cap, 'h100', 16, None, pending={'v7': 8}).ok)

  def test_balance_buys_above_floor(self):
    # g3: no floors, 56k balance; v6e-16 at 20/chip-hr = 320 cr/hr per hour held.
    cap = _gcap(group='3', balance=55996.0)
    v = R.group_can_hold(cap, 'v6e', 16, 20.0)
    self.assertTrue(v.ok)
    self.assertTrue(v.above_floor)

  def test_balance_threshold_is_reserve_hours_of_total_burn(self):
    # need = reserve hours x (existing burn 100 + this job 8 x 10)
    need = R.GROUP_BALANCE_RESERVE_H * (100.0 + 8 * 10.0)
    self.assertFalse(R.group_can_hold(
        _gcap(balance=need - 1.0, burn=100.0), 'v4', 8, 10.0).ok)
    self.assertTrue(R.group_can_hold(
        _gcap(balance=need, burn=100.0), 'v4', 8, 10.0).ok)

  def test_default_reserve_is_one_hour(self):
    # Operator 2026-09-24 01:09Z: "1改成一个小时吧" (was 6h).
    self.assertEqual(R.GROUP_BALANCE_RESERVE_H, 1.0)

  def test_reserve_hours_can_be_passed_explicitly(self):
    # need = 6h x 180 = 1080 > 1079; 5h x 180 = 900 <= 1079.
    cap = _gcap(balance=1079.0, burn=100.0)
    self.assertFalse(R.group_can_hold(cap, 'v4', 8, 10.0, reserve_h=6.0).ok)
    self.assertTrue(R.group_can_hold(cap, 'v4', 8, 10.0, reserve_h=5.0).ok)

  def test_only_the_chips_over_the_floor_are_costed(self):
    # floor 16, used 8: 8 of a 16-chip job fit the floor, 8 go above.
    # Costing all 16 chips would need twice as much.
    need = R.GROUP_BALANCE_RESERVE_H * 8 * 10.0
    lo = _gcap(floor={'v4': 16}, used={'v4': 8}, balance=need - 1.0)
    hi = _gcap(floor={'v4': 16}, used={'v4': 8}, balance=need)
    self.assertFalse(R.group_can_hold(lo, 'v4', 16, 10.0).ok)
    self.assertTrue(R.group_can_hold(hi, 'v4', 16, 10.0).ok)

  def test_pending_above_floor_is_costed_at_the_market_price(self):
    # 16 pending v7 chips, no v7 floor: 16 x 30 = 480 cr/hr on top of this job.
    prices = {'v7': 30.0, 'h100': 1.0}
    need = R.GROUP_BALANCE_RESERVE_H * (480.0 + 8 * 1.0)
    lo = _gcap(balance=need - 1.0, prices=prices)
    hi = _gcap(balance=need, prices=prices)
    self.assertFalse(
        R.group_can_hold(lo, 'h100', 8, 1.0, pending={'v7': 16}).ok)
    self.assertTrue(
        R.group_can_hold(hi, 'h100', 8, 1.0, pending={'v7': 16}).ok)

  def test_pending_above_floor_without_a_price_fails_closed(self):
    cap = _gcap(balance=1e9, prices={'h100': 1.0})
    self.assertFalse(
        R.group_can_hold(cap, 'h100', 8, 1.0, pending={'v7': 16}).ok)

  def test_group_limit_order_below_the_price_is_no_even_with_room(self):
    # g5 carries its own v6e order (~10): GQM refuses a dearer job outright.
    cap = _gcap(floor={'v6e': 64}, balance=1e9, caps={'v6e': 10.0})
    v = R.group_can_hold(cap, 'v6e', 16, 20.1)
    self.assertFalse(v.ok)
    self.assertIn('limit order', v.reason)
    self.assertTrue(R.group_can_hold(cap, 'v6e', 16, 9.9).ok)

  def test_limit_order_uses_the_market_price_when_no_cell_price(self):
    cap = _gcap(floor={'v6e': 64}, caps={'v6e': 10.0}, prices={'v6e': 20.1})
    self.assertFalse(R.group_can_hold(cap, 'v6e', 16, None).ok)

  def test_limit_order_with_no_price_at_all_fails_closed(self):
    cap = _gcap(floor={'v6e': 64}, caps={'v6e': 10.0})
    v = R.group_can_hold(cap, 'v6e', 16, None)
    self.assertFalse(v.ok)
    self.assertIn('limit order', v.reason)

  def test_no_floor_room_and_no_price_fails_closed(self):
    v = R.group_can_hold(_gcap(balance=1e9), 'b300', 8, None)
    self.assertFalse(v.ok)
    self.assertIn('no price', v.reason)

  def test_family_is_case_insensitive(self):
    cap = _gcap(floor={'h100': 16})
    self.assertTrue(R.group_can_hold(cap, 'H100', 8, None).ok)
    self.assertFalse(
        R.group_can_hold(cap, 'h100', 16, None, pending={'H100': 8}).ok)


if __name__ == '__main__':
  unittest.main()
