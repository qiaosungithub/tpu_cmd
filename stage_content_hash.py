#!/usr/bin/env python3
"""Deterministic content hash of a stage source tree.

This is the *content-addressing* primitive for the launcher's stagedirs
(design: ~/work/stagedir_content_addressing_design.md). It answers one
question and nothing else:

    "What is the sha256 of the exact bytes that would be staged from this
     source tree, ignoring anything that does not change the built .par?"

Two source trees that stage to byte-identical inputs MUST hash the same, so
that the content-addressed build target `//...:eqr_<sha>:main` is stable and
blaze's action cache hits instead of relinking the ~7 GB .par from scratch.
Two trees that differ in any build-relevant byte MUST hash differently, so a
distinct code can never collide with another distinct code's stagedir.

Design constraints honored here (see §3.4 of the design doc):

  * Mirror the launcher's own rsync selection EXACTLY. The staging rsync is
        rsync -aL --exclude={bazel-*,.citc,.git,.jj,.venv,__pycache__,
                             *.npy,*.npz,*.ckpt,*.pth,*.pt,*.safetensors,
                             data,logs,wandb} ./ dst/
    so the hash covers exactly the set that gets staged. `_STAGE_EXCLUDES`
    below is the single source of truth for that mirror; if the rsync line in
    tpu_wrapper.sh ever changes, change this together (LINT.IfChange guard).

  * NO mtime / inode / ownership / permission data. Those differ across
    byte-identical trees and would defeat the cache. Only path + content.

  * IGNORE the `TARGET_LABEL=...` line of the top-level `config.sh`. That line
    is DERIVED from the stagedir name, so hashing it would be circular (the
    name depends on the hash depends on the name). It is also not a .par input
    (top-level config.sh is read by the launcher at runtime, not in the data
    glob), so ignoring it cannot change the built artifact. Every other byte of
    config.sh is hashed normally.

  * `-L` (follow symlinks) is mirrored: a symlink to a regular file is hashed
    by its referent's content, exactly as rsync would materialize it. Symlink
    cycles are guarded so a pathological tree cannot hang the hasher.

The output is a lowercase hex sha256. It is intended to be used as
`eqr_<sha12>` (the first 12 hex chars are ample: 48 bits, collision-free for
any realistic number of distinct codes) but the caller decides truncation.

CLI:
    python3 stage_content_hash.py <root_dir>            # prints full hex sha
    python3 stage_content_hash.py <root_dir> --short=12 # prints first 12 chars
    python3 stage_content_hash.py <root_dir> --manifest # prints per-file lines
                                                        #   (debugging only)
Exit codes: 0 ok; 2 root missing / not a dir.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from typing import Iterable, Iterator, List, Tuple

# LINT.IfChange(_STAGE_EXCLUDES)
# EXACT mirror of the rsync --exclude={...} set in
# ~/work/tpu_cmd/tpu_wrapper.sh (the `timeout ... rsync -aL --exclude={...}`
# line). rsync matches a pattern with no internal slash against the BASENAME
# at every directory level, and prunes a matched directory entirely. We do the
# same below. Keep these two lists identical.
_STAGE_EXCLUDES: Tuple[str, ...] = (
    'bazel-*',
    '.citc',
    '.git',
    '.jj',
    '.venv',
    '__pycache__',
    '*.npy',
    '*.npz',
    '*.ckpt',
    '*.pth',
    '*.pt',
    '*.safetensors',
    'data',
    'logs',
    'wandb',
)
# LINT.ThenChange(//depot/google3/../../../work/tpu_cmd/tpu_wrapper.sh)

# The one file whose TARGET_LABEL line is ignored (circular; not a .par input).
_CONFIG_SH_RELPATH = 'config.sh'
_TARGET_LABEL_PREFIXES = ('export TARGET_LABEL=', 'TARGET_LABEL=')

_CHUNK = 1 << 20  # 1 MiB streaming read; never load a whole file into memory.


def _excluded(name: str) -> bool:
  """True iff a basename matches any rsync exclude pattern (case-sensitive)."""
  for pat in _STAGE_EXCLUDES:
    if fnmatch.fnmatchcase(name, pat):
      return True
  return False


def _iter_files(root: str) -> Iterator[str]:
  """Yield relpaths of files that rsync -aL would stage, sorted per directory.

  Mirrors rsync: prune excluded directories, skip excluded files, follow
  symlinks (-L) while guarding against cycles by real inode identity.
  """
  root = os.path.abspath(root)
  seen_dirs = set()  # real (st_dev, st_ino) of directories already entered

  def _walk(rel: str) -> Iterator[str]:
    abs = os.path.join(root, rel) if rel else root
    try:
      # Sort entries for deterministic order regardless of FS enumeration.
      entries = sorted(os.listdir(abs))
    except OSError:
      return
    for name in entries:
      if _excluded(name):
        continue
      child_rel = os.path.join(rel, name) if rel else name
      child_abs = os.path.join(abs, name)
      # Resolve through symlinks (-L). Use stat (follows) to classify; on a
      # broken symlink stat raises and rsync -aL would also skip it.
      try:
        st = os.stat(child_abs)  # follows symlinks
      except OSError:
        continue
      if os.path.isdir(child_abs):  # follows symlinks
        key = (st.st_dev, st.st_ino)
        if key in seen_dirs:
          continue  # symlink cycle or hardlinked dir; do not recurse twice
        seen_dirs.add(key)
        yield from _walk(child_rel)
      elif os.path.isfile(child_abs):  # follows symlinks
        yield child_rel
      # other (fifo/socket/device): rsync -a would copy specials, but they
      # never occur in a stage source and carry no build-relevant content;
      # ignore deterministically.

  yield from _walk('')


def _normalized_config_sh(abs_path: str) -> bytes:
  """config.sh bytes with any TARGET_LABEL line removed (circular; ignored).

  Read as bytes, split on newline, drop lines whose stripped form starts with
  a TARGET_LABEL assignment, rejoin. Byte-oriented so it never chokes on odd
  encodings; TARGET_LABEL lines are pure ASCII.
  """
  with open(abs_path, 'rb') as f:
    data = f.read()
  out_lines: List[bytes] = []
  for line in data.split(b'\n'):
    stripped = line.lstrip()
    if any(stripped.startswith(p.encode('ascii')) for p in _TARGET_LABEL_PREFIXES):
      continue
    out_lines.append(line)
  return b'\n'.join(out_lines)


def hash_manifest(root: str) -> List[Tuple[str, str]]:
  """Return [(relpath, per_file_sha256_hex), ...] in stable sorted order.

  Exposed for debugging / provenance and for the unit tests to pinpoint which
  file made two trees diverge. The overall tree hash is derived from this.
  """
  files = sorted(_iter_files(root))
  out: List[Tuple[str, str]] = []
  for rel in files:
    abs_path = os.path.join(os.path.abspath(root), rel)
    h = hashlib.sha256()
    if rel == _CONFIG_SH_RELPATH:
      h.update(_normalized_config_sh(abs_path))
    else:
      try:
        with open(abs_path, 'rb') as f:
          while True:
            chunk = f.read(_CHUNK)
            if not chunk:
              break
            h.update(chunk)
      except OSError:
        # A file that vanished mid-walk (rare). Hash a stable sentinel so the
        # result is deterministic rather than raising; callers verify
        # completeness separately.
        h.update(b'\0<UNREADABLE>\0')
    out.append((rel, h.hexdigest()))
  return out


def hash_tree(root: str) -> str:
  """Deterministic sha256 hex over the staged content of `root`.

  Algorithm (design §3.4): for each staged file in sorted path order, feed
  `relpath\\0` then the file's content bytes into one sha256. config.sh has its
  TARGET_LABEL line removed first. No mtimes, no inode data, no permissions.
  """
  if not os.path.isdir(root):
    raise ValueError(f'stage source root is not a directory: {root!r}')
  h = hashlib.sha256()
  for rel in sorted(_iter_files(root)):
    abs_path = os.path.join(os.path.abspath(root), rel)
    # The relpath, NUL-delimited, so file boundaries are unambiguous and a
    # rename (same content, new path) changes the hash.
    h.update(rel.encode('utf-8'))
    h.update(b'\0')
    if rel == _CONFIG_SH_RELPATH:
      h.update(_normalized_config_sh(abs_path))
    else:
      try:
        with open(abs_path, 'rb') as f:
          while True:
            chunk = f.read(_CHUNK)
            if not chunk:
              break
            h.update(chunk)
      except OSError:
        h.update(b'\0<UNREADABLE>\0')
    h.update(b'\0')  # explicit terminator between records
  return h.hexdigest()


def _main(argv: Iterable[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('root', help='stage source directory to hash')
  ap.add_argument('--short', type=int, default=0,
                  help='print only the first N hex chars (0 = full)')
  ap.add_argument('--manifest', action='store_true',
                  help='print per-file "sha  relpath" lines instead of the tree hash')
  args = ap.parse_args(list(argv))

  if not os.path.isdir(args.root):
    print(f'stage_content_hash: not a directory: {args.root}', file=sys.stderr)
    return 2

  if args.manifest:
    for rel, sha in hash_manifest(args.root):
      print(f'{sha}  {rel}')
    return 0

  sha = hash_tree(args.root)
  if args.short > 0:
    sha = sha[:args.short]
  print(sha)
  return 0


if __name__ == '__main__':
  sys.exit(_main(sys.argv[1:]))
