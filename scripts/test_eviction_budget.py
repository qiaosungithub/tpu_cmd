#!/usr/bin/env python3
"""Offline regression tests for the Borg task-eviction launch budget.

The production launcher depends on internal XManager packages that are not in
the workstation Python environment. These tests execute the policy helper from
its AST with a minimal stand-in for the generated override proto, then inspect
the launcher/wrapper AST and case arms for the integration contract. They never
contact XManager or Borg.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "xm_launcher.py"
WRAPPER = ROOT / "tpu_wrapper.sh"


class _Scheduling:
    pass


class _Overrides:

    def __init__(self):
        self.scheduling = _Scheduling()


class _FakeXmAbc:
    calls = 0

    @classmethod
    def RESTRICTED_BorgOverrides(cls):  # pylint: disable=invalid-name
        cls.calls += 1
        return _Overrides()


def _launcher_tree() -> ast.Module:
    return ast.parse(LAUNCHER.read_text(), filename=str(LAUNCHER))


def _load_policy_helper():
    helper = next(
        node
        for node in _launcher_tree().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_borg_overrides_for_eviction_budget"
    )
    helper_module = ast.fix_missing_locations(
        ast.Module(body=[helper], type_ignores=[]))
    namespace = {"xm_abc": _FakeXmAbc}
    exec(compile(helper_module, str(LAUNCHER), "exec"), namespace)  # pylint: disable=exec-used
    return namespace["_borg_overrides_for_eviction_budget"]


class EvictionBudgetPolicyTest(unittest.TestCase):

    def setUp(self):
        _FakeXmAbc.calls = 0
        self.policy = _load_policy_helper()

    def test_default_minus_one_emits_no_override(self):
        self.assertIsNone(self.policy(-1))
        self.assertEqual(_FakeXmAbc.calls, 0)

    def test_zero_sets_only_finite_eviction_budget(self):
        overrides = self.policy(0)
        self.assertEqual(overrides.scheduling.max_task_evictions, 0)
        self.assertEqual(_FakeXmAbc.calls, 1)
        self.assertEqual(vars(overrides), {"scheduling": overrides.scheduling})
        self.assertEqual(vars(overrides.scheduling), {"max_task_evictions": 0})

    def test_positive_budget_is_preserved(self):
        overrides = self.policy(3)
        self.assertEqual(overrides.scheduling.max_task_evictions, 3)

    def test_values_below_minus_one_are_rejected_before_override_creation(self):
        with self.assertRaisesRegex(ValueError, "must be -1"):
            self.policy(-2)
        self.assertEqual(_FakeXmAbc.calls, 0)


class EvictionBudgetWiringTest(unittest.TestCase):

    def test_flag_default_is_unlimited(self):
        assignment = next(
            node
            for node in _launcher_tree().body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_MAX_TASK_EVICTIONS"
                for target in node.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Call)
        self.assertEqual(ast.literal_eval(assignment.value.args[0]),
                         "borg_max_task_evictions")
        self.assertEqual(ast.literal_eval(assignment.value.args[1]), -1)

    def test_registry_and_application_filter_record_and_consume_flag(self):
        source = LAUNCHER.read_text()
        self.assertRegex(
            source,
            re.compile(
                r'["\']max_task_evictions["\']\s*:\s*'
                r'_MAX_TASK_EVICTIONS\.value'))
        self.assertIn("'--borg_max_task_evictions='", source)
        self.assertIn("borg_kwargs['borg_overrides'] = eviction_overrides",
                      source)

    def test_wrapper_accepts_equals_and_separated_forms(self):
        arms = [
            line.strip()
            for line in WRAPPER.read_text().splitlines()
            if line.strip().endswith(")")
            and "--borg_max_task_evictions" in line
        ]
        self.assertTrue(
            any("--borg_max_task_evictions=*" in arm for arm in arms), arms)
        self.assertTrue(
            any(
                re.search(
                    r"(?:^|\|)--borg_max_task_evictions(?:\||\)$)", arm)
                for arm in arms
            ),
            arms,
        )


if __name__ == "__main__":
    unittest.main()
