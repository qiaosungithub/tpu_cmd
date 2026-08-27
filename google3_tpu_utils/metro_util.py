"""Cell -> metro resolution. Dependency-free leaf, so every layer agrees.

A borg cell name usually encodes its metro as ``yu<metro><suffix>``
(``yutulpz`` -> ``tul``, ``yucbfiv`` -> ``cbf``). Short / legacy names do not,
so an override table catches those. This is the SINGLE owner of that mapping:
both routers depend on it (the smart-cell default via ``avail_provider`` and the
``--power`` router via ``preflight.router``), and if the two disagreed on which
metro a cell is in, ``--metro`` would filter differently depending on which path
picked the cell -- exactly the class of silent drift this module removes.

Precision only matters when a job opts into an allowed-metros filter; with no
filter (the default) the metro is never consulted.
"""

# Cells whose metro is NOT recoverable from the ``yu<metro>...`` pattern.
# Seeded from xm_launcher._CELL_BUCKETS (the launcher's cell -> same-metro
# bucket map is the authority on which cells share a storage metro).
METRO_OVERRIDES: dict[str, str] = {
    'dl': 'las',   # las -> dl-d (2nd v4 cell)
    'je': 'cbf',   # cbf neighbour
    'nl': 'tul',   # tul neighbour
    'nk': 'tul',   # tul neighbour
    'el': 'grq',
    'mb': 'ckv',
    'sk': 'sin', 'sn': 'sin', 'so': 'sin',
}


def metro_of(cell: str) -> str:
  """Best-effort metro token for a borg cell name.

  Falls back to the override table, then to the ``yu<metro>`` prefix, then to
  the cell name itself (so an unknown cell is its own metro rather than a crash).
  """
  c = (cell or '').lower()
  if c in METRO_OVERRIDES:
    return METRO_OVERRIDES[c]
  if c.startswith('yu') and len(c) >= 5:
    return c[2:5]
  return c
