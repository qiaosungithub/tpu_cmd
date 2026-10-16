#!/usr/bin/env python3
"""Replays xm_launcher.py's executor-construction block offline and records
the exact Borg request it would emit, per `--tpu_type`.

WHY. The launcher imports internal XManager packages that the workstation
Python cannot import, so the request-building loop in `main()` has never been
testable as a unit. This harness lifts that block out of the launcher's AST
(from `executors = []` through the `for tpu_str in tpu_types:` loop, i.e. the
code that builds every `xm.JobRequirements` / `xm_abc.Borg(...)`), runs it
against recording stand-ins for `xm` / `xm_abc`, and serialises everything the
executor would be constructed with. Two launchers that serialise identically
for a `--tpu_type` emit the same request for it.

    python3 launcher_request_harness.py --write-golden   # regenerate golden
    python3 launcher_request_harness.py --show a100-8    # print one request

The golden file (testdata/launcher_request_golden.json) was captured from the
launcher BEFORE the 2026-10-16 BigRig fix; test_bigrig_locality.py diffs the
current launcher against it.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = Path(os.environ.get('XM_LAUNCHER_UNDER_TEST',
                                       str(ROOT / 'xm_launcher.py')))
GOLDEN = Path(__file__).resolve().parent / 'testdata' / 'launcher_request_golden.json'

# Module-level helpers the block calls. Names absent from a given launcher
# (e.g. the BigRig helper in a pre-fix launcher) are simply skipped.
_MODULE_NAMES = (
    '_borg_overrides_for_eviction_budget',
    '_BIGRIG_SUBHOST_GPUS',
    '_BIGRIG_DOMAIN_SIZES',
    '_bigrig_device_group_specs',
    '_apply_bigrig_locality',
)

# (case name, --tpu_type, extra launcher argv, --borg_max_task_evictions)
CASES = (
    ('h100-8', 'h100-8', (), -1),
    ('b200-8', 'b200-8', (), -1),
    ('h100-4', 'h100-4', (), -1),
    ('h200-8', 'h200-8', (), -1),
    ('a100-8', 'a100-8', (), -1),
    ('a100-4', 'a100-4', (), -1),
    ('a100-16', 'a100-16', (), -1),
    ('a100-1', 'a100-1', (), -1),
    ('a100_80gib-8', 'a100_80gib-8', (), -1),
    ('a100_80gib-4', 'a100_80gib-4', (), -1),
    ('v6e-16', 'v6e-16', (), -1),
    ('v6p-32', 'v6p-32', (), -1),
    ('v7-16', 'v7-16', (), -1),
    ('v5p-16', 'v5p-16', (), -1),
    ('cpu', 'cpu=1', (), -1),
    ('h100-8,b200-8', 'h100-8,b200-8', (), -1),
    ('h100-8+evict3', 'h100-8', (), 3),
    ('a100-8+evict3', 'a100-8', (), 3),
    ('b200-8+batch', 'b200-8', ('--tier=BATCH',), -1),
    ('a100-8+batch', 'a100-8', ('--tier=BATCH',), -1),
)

# Cases whose request the BigRig fix is SUPPOSED to change.
CHANGED_BY_BIGRIG_FIX = frozenset({
    'a100-8', 'a100-4', 'a100_80gib-4', 'a100-8+evict3', 'a100-8+batch'})


class _Fake:
  """A recording stand-in: remembers its type name and every attribute."""

  def __init__(self, type_name: str, **kwargs: Any):
    self._type = type_name
    for key, value in kwargs.items():
      setattr(self, key, value)


class _Overrides(_Fake):
  """RESTRICTED_BorgOverrides: attribute bag with an auto `scheduling` node."""

  def __init__(self):
    super().__init__('RESTRICTED_BorgOverrides')
    self.scheduling = _Fake('scheduling')


class _FakeXm:
  GiB = 2**30

  class ServiceTier:  # pylint: disable=too-few-public-methods
    PROD = 'ServiceTier.PROD'
    BATCH = 'ServiceTier.BATCH'

  @staticmethod
  def JobRequirements(**kwargs):  # pylint: disable=invalid-name
    return _Fake('JobRequirements', **kwargs)


class _FakeXmAbc:

  @staticmethod
  def BorgScheduling(**kwargs):  # pylint: disable=invalid-name
    return _Fake('BorgScheduling', **kwargs)

  @staticmethod
  def RESTRICTED_BorgOverrides():  # pylint: disable=invalid-name
    return _Overrides()

  @staticmethod
  def RESTRICTED_BorgToken(text):  # pylint: disable=invalid-name
    return _Fake('RESTRICTED_BorgToken', text=text)

  @staticmethod
  def AutopilotParams(**kwargs):  # pylint: disable=invalid-name
    return _Fake('AutopilotParams', **kwargs)

  @staticmethod
  def Borg(**kwargs):  # pylint: disable=invalid-name
    return _Fake('Borg', **kwargs)

  @staticmethod
  def Gcp(**kwargs):  # pylint: disable=invalid-name
    return _Fake('Gcp', **kwargs)


class _Flag:  # pylint: disable=too-few-public-methods

  def __init__(self, value):
    self.value = value


def _serialise(obj: Any) -> Any:
  if obj is None or isinstance(obj, (str, int, float, bool)):
    return obj
  if isinstance(obj, (list, tuple)):
    return [_serialise(x) for x in obj]
  if isinstance(obj, dict):
    return {str(k): _serialise(v) for k, v in sorted(obj.items())}
  if isinstance(obj, _Fake):
    out = {'__type__': obj._type}  # pylint: disable=protected-access
    for key, value in sorted(vars(obj).items()):
      if key != '_type':
        out[key] = _serialise(value)
    return out
  raise TypeError(f'cannot serialise {type(obj).__name__}: {obj!r}')


def _find_block(tree: ast.Module) -> list[ast.stmt]:
  """Statements from `executors = []` through the `for tpu_str` loop."""
  main = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == 'main')
  for node in ast.walk(main):
    for field in ('body', 'orelse', 'finalbody'):
      stmts = getattr(node, field, None)
      if not isinstance(stmts, list):
        continue
      for i, stmt in enumerate(stmts):
        if (isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name)
            and stmt.target.id == 'tpu_str'):
          start = next(
              j for j in range(i, -1, -1)
              if isinstance(stmts[j], ast.Assign) and any(
                  isinstance(t, ast.Name) and t.id == 'executors'
                  for t in stmts[j].targets))
          return stmts[start:i + 1]
  raise LookupError('executor-construction loop not found in launcher')


def _module_defs(tree: ast.Module) -> list[ast.stmt]:
  out = []
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _MODULE_NAMES:
      out.append(node)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      if any(isinstance(t, ast.Name) and t.id in _MODULE_NAMES for t in targets):
        out.append(node)
  return out


class LauncherReplay:
  """Compiled once per launcher file; `emit()` replays one --tpu_type."""

  def __init__(self, launcher: Path = DEFAULT_LAUNCHER):
    self.launcher = Path(launcher)
    tree = ast.parse(self.launcher.read_text(), filename=str(self.launcher))
    self._module_code = compile(
        ast.fix_missing_locations(
            ast.Module(body=_module_defs(tree), type_ignores=[])),
        str(self.launcher), 'exec')
    self._block_code = compile(
        ast.fix_missing_locations(
            ast.Module(body=_find_block(tree), type_ignores=[])),
        str(self.launcher), 'exec')

  def emit(self, tpu_type: str, extra_argv=(), max_task_evictions: int = -1,
           cell: str = 'mg') -> dict:
    argv = ['xm_launcher.py', f'--tpu_type={tpu_type}',
            '--xm_resource_alloc=group:deepmind-dynamic/'
            'fr-dna-grand-challenge-team-resource']
    if not any(a.startswith('--tier=') for a in extra_argv):
      argv.append('--tier=PROD')
    argv.extend(extra_argv)
    ns: dict[str, Any] = {
        'xm': _FakeXm, 'xm_abc': _FakeXmAbc,
        'argv': argv,
        '_TPU_TYPE': _Flag(tpu_type),
        'tmp_ram_fs_gib': 16.0,
        'package_mode': 'bazel',
        '_RAM_GIB': _Flag(0.0),
        '_REPLICAS': _Flag(1),
        '_CELL': _Flag(cell),
        '_CELL_PREFER': _Flag(''),
        '_preferred_cells': lambda spec: [],
        '_describe_cell_choice': lambda cells: None,
        '_MAX_TASK_FAILURES': _Flag(-1),
        '_MAX_PER_TASK_FAILURES': _Flag(3),
        '_FAILURE_CREDIT_PERIOD': _Flag(7200),
        '_MAX_TASK_EVICTIONS': _Flag(max_task_evictions),
        '_AUTOPILOT': _Flag(None),
        'print': lambda *a, **k: None,
    }
    exec(self._module_code, ns)  # pylint: disable=exec-used
    exec(self._block_code, ns)  # pylint: disable=exec-used
    return {
        'executors': _serialise(ns['executors']),
        'is_tpu_job': ns['is_tpu_job'],
        'tpu_res_names': list(ns['_tpu_res_names']),
    }

  def emit_all(self) -> dict:
    return {name: self.emit(tpu, extra, ev) for name, tpu, extra, ev in CASES}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--launcher', default=str(DEFAULT_LAUNCHER))
  parser.add_argument('--write-golden', action='store_true')
  parser.add_argument('--show')
  args = parser.parse_args()
  replay = LauncherReplay(Path(args.launcher))
  if args.show:
    case = next(c for c in CASES if c[0] == args.show)
    print(json.dumps(replay.emit(case[1], case[2], case[3]), indent=1))
    return 0
  if args.write_golden:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    payload = {'launcher': str(Path(args.launcher).resolve()),
               'cases': replay.emit_all()}
    GOLDEN.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(f'wrote {GOLDEN} ({len(payload["cases"])} cases)')
    return 0
  print(json.dumps(replay.emit_all(), indent=1, sort_keys=True))
  return 0


if __name__ == '__main__':
  sys.exit(main())
