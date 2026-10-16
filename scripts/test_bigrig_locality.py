#!/usr/bin/env python3
"""Golden-diff tests for the A100 BIG_RIG locality fix in xm_launcher.py.

The request-building loop is replayed offline by launcher_request_harness.py
(recording stand-ins for xm / xm_abc; nothing contacts XManager or Borg). The
golden file was captured from the launcher BEFORE the fix, so:

  * every non-A100 case (h100-8, b200-8, h100-4, h200-8, TPU, CPU, a Fallback
    pair, eviction-budget and BATCH variants) must serialise IDENTICALLY;
  * a100-8 must equal its pre-fix request plus exactly two fields: the job
    `borg_overrides.device_group_spec` (group + BIG_RIG_8) and the alloc
    `alloc_overrides.device_group_spec` (child_group + BIG_RIG_8).

Run:  python3 ~/work/tpu_cmd/scripts/test_bigrig_locality.py
      XM_LAUNCHER_UNDER_TEST=<path> python3 ...   # test a candidate file
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launcher_request_harness as harness  # pylint: disable=g-import-not-at-top,wrong-import-position


def _bigrig_bcl(field: str, scalar: str, count: int) -> str:
  return ('{\n'
          '  spec1 = {\n'
          f"    {field} = [{{ scalar = [{{ key = '{scalar}', value = {count} }}] }}]\n"
          f"    locality = [{{ type = 'BIG_RIG_{count}', devices = ['{scalar}'] }}]\n"
          '  }\n'
          '}')


def _token(text: str) -> dict:
  return {'__type__': 'RESTRICTED_BorgToken', 'text': text}


class BigRigGoldenTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.golden = json.loads(harness.GOLDEN.read_text())['cases']
    cls.replay = harness.LauncherReplay(harness.DEFAULT_LAUNCHER)
    cls.now = cls.replay.emit_all()

  def test_golden_covers_every_case(self):
    self.assertEqual(sorted(self.golden), sorted(c[0] for c in harness.CASES))

  def test_unaffected_types_are_byte_identical(self):
    for name in sorted(self.golden):
      if name in harness.CHANGED_BY_BIGRIG_FIX:
        continue
      with self.subTest(case=name):
        self.assertEqual(
            json.dumps(self.now[name], sort_keys=True),
            json.dumps(self.golden[name], sort_keys=True))

  def test_the_named_operator_cases_are_unchanged(self):
    # The operator's explicit list: h100-8, b200-8 and a TPU type.
    for name in ('h100-8', 'b200-8', 'v6e-16', 'v6p-32', 'v7-16'):
      with self.subTest(case=name):
        self.assertEqual(self.now[name], self.golden[name])
        borg = self.now[name]['executors'][0]
        self.assertNotIn('alloc_overrides', borg)
        self.assertNotIn('borg_overrides', borg)

  def _expected_with_bigrig(self, name, scalar, count):
    want = copy.deepcopy(self.golden[name])
    borg = want['executors'][0]
    job_ov = borg.get('borg_overrides') or {
        '__type__': 'RESTRICTED_BorgOverrides',
        'scheduling': {'__type__': 'scheduling'}}
    job_ov['device_group_spec'] = _token(_bigrig_bcl('group', scalar, count))
    borg['borg_overrides'] = job_ov
    borg['alloc_overrides'] = {
        '__type__': 'RESTRICTED_BorgOverrides',
        'scheduling': {'__type__': 'scheduling'},
        'device_group_spec': _token(_bigrig_bcl('child_group', scalar, count)),
    }
    return want

  def test_a100_8_is_prefix_request_plus_big_rig_8(self):
    self.assertEqual(self.now['a100-8'],
                     self._expected_with_bigrig('a100-8', 'GPU_TESLA_A100_40GIB', 8))

  def test_other_sub_host_a100_asks(self):
    for name, scalar, count in (
        ('a100-4', 'GPU_TESLA_A100_40GIB', 4),
        ('a100_80gib-4', 'GPU_TESLA_A100_80GIB', 4),
        ('a100-8+batch', 'GPU_TESLA_A100_40GIB', 8)):
      with self.subTest(case=name):
        self.assertEqual(self.now[name],
                         self._expected_with_bigrig(name, scalar, count))

  def test_eviction_budget_and_locality_share_one_override(self):
    got = self.now['a100-8+evict3']
    self.assertEqual(got, self._expected_with_bigrig(
        'a100-8+evict3', 'GPU_TESLA_A100_40GIB', 8))
    job_ov = got['executors'][0]['borg_overrides']
    self.assertEqual(job_ov['scheduling']['max_task_evictions'], 3)
    self.assertIn('BIG_RIG_8', job_ov['device_group_spec']['text'])

  def test_no_rule_where_xmanager_or_borg_already_decides(self):
    # a100-16 / a100_80gib-8 are full host groups: XManager emits the rule.
    # a100-1 needs none. All three must be exactly the pre-fix request.
    for name in ('a100-16', 'a100_80gib-8', 'a100-1'):
      with self.subTest(case=name):
        self.assertEqual(self.now[name], self.golden[name])

  def test_negative_control_harness_sees_the_change(self):
    # If the harness could not observe a difference, every "identical" verdict
    # above would be vacuous. The pre-fix a100-8 request must differ.
    self.assertNotEqual(self.now['a100-8'], self.golden['a100-8'])
    self.assertNotIn('alloc_overrides', self.golden['a100-8']['executors'][0])

  def test_job_half_uses_group_and_alloc_half_child_group(self):
    borg = self.now['a100-8']['executors'][0]
    job = borg['borg_overrides']['device_group_spec']['text']
    alloc = borg['alloc_overrides']['device_group_spec']['text']
    self.assertIn('group = [', job)
    self.assertNotIn('child_group', job)   # Borg: "Jobs cannot specify child groups."
    self.assertIn('child_group = [', alloc)
    for text in (job, alloc):
      self.assertIn("key = 'GPU_TESLA_A100_40GIB', value = 8", text)
      self.assertIn("type = 'BIG_RIG_8', devices = ['GPU_TESLA_A100_40GIB']", text)


if __name__ == '__main__':
  unittest.main(verbosity=2)
