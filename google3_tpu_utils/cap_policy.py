"""Single source of truth for our per-accelerator price caps (limit orders).

Operator directive (2026-08-25): every job this workstation launches is capped
at a FIXED price per chip-hour by accelerator family -- no external override.
The same table is what `tpu money` displays and colours against, so what we
enforce and what we show can never drift apart.

Cap is an ABSOLUTE price in credits per CHIP-hour, PROD tier. BATCH clears at
~0 and its admission never reads a price, so a cap there is inert -- we do not
apply caps to BATCH jobs (money still displays the policy for reference).

Keep in sync with the launcher's hardcoded table:
LINT.IfChange
"""

# family -> credits/chip-hour. The one place these numbers live.
CAP_POLICY = {
    "v7": 20,
    "v6p": 20,
    "v6e": 10,
    "v5p": 5,
    "v4": 5,
}
# LINT.ThenChange(//depot/google3/experimental/users/qiaos/tpu_cmd/tpu_wrapper.sh)

# GQM keys accelerators by numeric product id; map id -> family so a caller
# holding a market.json / TARGET_CARDS id can resolve the policy. Ids per the
# authoritative table in wiki_agents/tpu_reference.md: VIPERFISH(59)=v5p,
# VIPERLITE(60)=v5e (NOT the swapped pair some older tools carry), GHOSTFISH(92)
# =v6p, GHOSTFISHLITE(101)=v7. v6e also appears as 63 in one pool cache.
_FAMILY_BY_ID = {34: "v4", 59: "v5p", 60: "v5e", 63: "v6e", 76: "v6e",
                 92: "v6p", 101: "v7"}


def cap_for_family(family):
  """Cap for a family name like 'v7', or None if we have no policy for it."""
  return CAP_POLICY.get(family)


def cap_for_type_id(type_id):
  """Cap for a numeric GQM product id, or None if unknown / no policy."""
  fam = _FAMILY_BY_ID.get(int(type_id)) if type_id is not None else None
  return CAP_POLICY.get(fam) if fam else None
