#!/usr/bin/env python3
"""Unit tests for stage_content_hash.hash_tree / hash_manifest.

Self-asserting: exits non-zero on the first failure, prints a summary on
success. Run:  python3 stage_content_hash_test.py

These tests ARE the §8 "hash stability" verification from the design doc:
  - identical tree            -> identical hash
  - one content byte changed  -> different hash
  - mtime / permission change -> SAME hash (no time/inode data in the hash)
  - TARGET_LABEL line change  -> SAME hash (circular line ignored)
  - other config.sh change    -> different hash
  - excluded paths ignored    -> adding a .git/, __pycache__/, *.pt is a no-op
  - file rename (same bytes)  -> different hash (path is part of the record)
  - symlink to a file (-L)    -> hashed by referent content
  - symlink cycle             -> terminates, deterministic
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import stage_content_hash as sch


_failures = []


def _check(cond: bool, msg: str) -> None:
  status = 'ok  ' if cond else 'FAIL'
  print(f'[{status}] {msg}')
  if not cond:
    _failures.append(msg)


def _write(root: str, rel: str, content: bytes) -> None:
  p = os.path.join(root, rel)
  os.makedirs(os.path.dirname(p), exist_ok=True)
  with open(p, 'wb') as f:
    f.write(content)


def _base_tree(root: str) -> None:
  """A representative stage tree: BUILD, entry src, config.sh, configs, vendor."""
  _write(root, 'BUILD', b'pytype_binary(name="main", srcs=["main_eqr.py"])\n')
  _write(root, 'main_eqr.py', b'print("hello")\n')
  _write(root, 'config.sh',
         b'export PROJECT_NAME="elt-dit"\n'
         b'export TARGET_LABEL="//experimental/qiaos/eqr_jax_final_stages/eqr_run_260915_172131_790b9d:main"\n')
  _write(root, 'configs/base.yml', b'lr: 0.001\nsteps: 1000\n')
  _write(root, 'configs/model/dit.py', b'DIM = 768\n')
  _write(root, 'vendor/gldm/unet.py', b'class UNet: pass\n')


def main() -> int:
  tmp = tempfile.mkdtemp(prefix='sch_test_')
  try:
    a = os.path.join(tmp, 'a')
    b = os.path.join(tmp, 'b')
    os.makedirs(a)
    os.makedirs(b)

    # 1. identical trees -> identical hash
    _base_tree(a)
    _base_tree(b)
    ha, hb = sch.hash_tree(a), sch.hash_tree(b)
    _check(ha == hb, f'identical trees hash equal ({ha[:12]} == {hb[:12]})')
    _check(len(ha) == 64, 'hash is a 64-hex-char sha256')

    # 2. mtime change alone -> SAME hash (no time data in the hash)
    old = time.time() - 100000
    os.utime(os.path.join(b, 'main_eqr.py'), (old, old))
    _check(sch.hash_tree(b) == ha, 'mtime change does NOT change the hash')

    # 3. permission change alone -> SAME hash
    os.chmod(os.path.join(b, 'main_eqr.py'), 0o600)
    _check(sch.hash_tree(b) == ha, 'permission change does NOT change the hash')

    # 4. TARGET_LABEL line differs -> SAME hash (circular line ignored)
    _write(b, 'config.sh',
           b'export PROJECT_NAME="elt-dit"\n'
           b'export TARGET_LABEL="//totally/different/path/eqr_run_999999_abcdef:main"\n')
    _check(sch.hash_tree(b) == ha,
           'differing TARGET_LABEL line does NOT change the hash')

    # 4b. TARGET_LABEL without `export ` prefix is also ignored
    _write(b, 'config.sh',
           b'export PROJECT_NAME="elt-dit"\n'
           b'TARGET_LABEL="//x:main"\n')
    _check(sch.hash_tree(b) == ha,
           'bare (no-export) TARGET_LABEL line also ignored')

    # 5. a NON-TARGET_LABEL change in config.sh -> DIFFERENT hash
    _write(b, 'config.sh',
           b'export PROJECT_NAME="SOMETHING-ELSE"\n'
           b'export TARGET_LABEL="//x:main"\n')
    _check(sch.hash_tree(b) != ha,
           'a real config.sh change (PROJECT_NAME) DOES change the hash')
    _write(b, 'config.sh',  # restore
           b'export PROJECT_NAME="elt-dit"\n'
           b'export TARGET_LABEL="//x:main"\n')
    _check(sch.hash_tree(b) == ha, 'restoring config.sh returns to base hash')

    # 6. one content byte in a real input -> DIFFERENT hash
    _write(b, 'configs/base.yml', b'lr: 0.002\nsteps: 1000\n')
    _check(sch.hash_tree(b) != ha, 'a one-byte config yml change changes the hash')
    _write(b, 'configs/base.yml', b'lr: 0.001\nsteps: 1000\n')  # restore
    _check(sch.hash_tree(b) == ha, 'restoring the yml returns to base hash')

    # 7. excluded paths are ignored (adding them is a no-op)
    _write(b, '.git/config', b'[core]\n')
    _write(b, '__pycache__/x.pyc', b'\x00\x01\x02')
    _write(b, 'vendor/gldm/__pycache__/unet.cpython-311.pyc', b'\xff')
    _write(b, 'weights.pt', b'BIGWEIGHTS' * 1000)
    _write(b, 'ckpt/model.safetensors', b'X' * 100)
    _write(b, 'data/train.npy', b'Y' * 100)
    _write(b, 'logs/run.log', b'log line\n')
    _write(b, 'wandb/latest', b'w')
    os.makedirs(os.path.join(b, 'bazel-out'), exist_ok=True)
    _write(b, 'bazel-out/thing', b'z')
    _check(sch.hash_tree(b) == ha,
           'adding excluded paths (.git,__pycache__,*.pt,*.safetensors,data,logs,wandb,bazel-*) is a no-op')

    # 8. file rename with identical bytes -> DIFFERENT hash (path is a record)
    c = os.path.join(tmp, 'c')
    os.makedirs(c)
    _base_tree(c)
    os.rename(os.path.join(c, 'configs/base.yml'),
              os.path.join(c, 'configs/base_renamed.yml'))
    _check(sch.hash_tree(c) != ha, 'renaming a file (same bytes) changes the hash')

    # 9. a NEW real file -> DIFFERENT hash
    d = os.path.join(tmp, 'd')
    os.makedirs(d)
    _base_tree(d)
    _write(d, 'configs/extra.yml', b'new: true\n')
    _check(sch.hash_tree(d) != ha, 'adding a real config file changes the hash')

    # 10. symlink to a regular file (-L): hashed by referent content
    e = os.path.join(tmp, 'e')
    f = os.path.join(tmp, 'f')
    os.makedirs(e)
    os.makedirs(f)
    _base_tree(e)
    _base_tree(f)
    # In f, replace vendor/gldm/unet.py with a symlink to an identical-content
    # file elsewhere in the tree.
    _write(f, 'vendor/gldm/_real_unet.py', b'class UNet: pass\n')
    os.remove(os.path.join(f, 'vendor/gldm/unet.py'))
    os.symlink('_real_unet.py', os.path.join(f, 'vendor/gldm/unet.py'))
    # e must match f only if we ALSO add the same _real_unet.py to e (so the
    # file SETS match); the point of this test is that the symlink is followed
    # to content, not recorded as a link. Add the real file to e too:
    _write(e, 'vendor/gldm/_real_unet.py', b'class UNet: pass\n')
    _check(sch.hash_tree(e) == sch.hash_tree(f),
           'a -L symlink is hashed by referent content, not as a link')

    # 11. symlink cycle terminates deterministically
    g = os.path.join(tmp, 'g')
    os.makedirs(g)
    _base_tree(g)
    os.symlink('.', os.path.join(g, 'selfloop'))  # dir cycle
    try:
      hg1 = sch.hash_tree(g)
      hg2 = sch.hash_tree(g)
      _check(hg1 == hg2, 'symlink cycle: hash terminates and is deterministic')
    except RecursionError:
      _check(False, 'symlink cycle caused RecursionError (guard failed)')

    # 12. non-directory root -> ValueError
    try:
      sch.hash_tree(os.path.join(a, 'BUILD'))
      _check(False, 'hashing a non-directory should raise ValueError')
    except ValueError:
      _check(True, 'hashing a non-directory raises ValueError')

    # 13. manifest agrees with the file set and is sorted
    man = sch.hash_manifest(a)
    rels = [r for r, _ in man]
    _check(rels == sorted(rels), 'manifest is sorted by relpath')
    _check('config.sh' in rels and 'BUILD' in rels and 'configs/base.yml' in rels,
           'manifest covers the expected staged files')
    _check(all('.git' not in r and '__pycache__' not in r for r in rels),
           'manifest excludes the excluded paths')

  finally:
    shutil.rmtree(tmp, ignore_errors=True)

  print()
  if _failures:
    print(f'FAILED: {len(_failures)} check(s) failed:')
    for m in _failures:
      print(f'  - {m}')
    return 1
  print('ALL CHECKS PASSED')
  return 0


if __name__ == '__main__':
  sys.exit(main())
