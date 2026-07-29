# `tpu` — TPU job submission wrapper for qiaos

A thin, opinionated CLI around `xmanager launch` that adds:

- Client-side **preflight** checks (topology + capacity) *before* the
  ~5 min bazel packaging so you don't burn time on requests that will be
  immediately rejected by the allocator.
- A **local router** that turns a desired compute level
  (e.g. `v5p-32`, `v6e-16`, `v4-32` — all equivalent) into a concrete
  `(group, tpu_type)` recommendation.
- Zero-latency cached views of `tpu quota`, `tpu money`, `tpu check`
  (refreshed every 60s by a background `tmux` daemon).

Everything else (packaging, `xmanager launch`, background daemon,
auto-retry on PROD/`Rejected by Allocator/Borg`) works the same as
before.

**Prerequisite**: source the wrapper in your shell (already in
`.bashrc` for interactive shells):

```bash
source ~/work/tpu_cmd/tpu_wrapper.sh
```

---

## Command reference

### `tpu queue` / `tpu q` — submit a job

```
tpu queue --tpu_type=v6e-16 --group=5 [--tier=PROD] [passthrough flags...]
tpu queue --power=v5p-32 [--tier=PROD] [passthrough flags...]   # router mode
```

Flags:

| Flag | Meaning |
|---|---|
| `--tpu_type=<arch>-<chips>` | e.g. `v6e-16`, `v5p-32`, `v4-64`. Comma-list allowed for XM Fallback. |
| `--group=<N>` | group id (1..9) or full `group:...` alloc string. Comma-list allowed. |
| `--power=<spec>` | Instead of `--tpu_type + --group`: let the router pick. `<spec>` is `v5p-32`, `v6e-16`, `v4-32` (equivalent), or a bare integer in v5p-chip units. Mutually exclusive with `--tpu_type` / `--group`. |
| `--tier=PROD\|BATCH` | Defaults to empty; g5 auto-set to PROD for legacy reason. |
| `--force` / `-f` | Submit even if preflight verdict is RED. |
| `--skip-preflight` / `--no-preflight` | Bypass preflight entirely. |
| `--exp_name=...`, `-n ...`, `--config=...`, `--bucket=...`, `--workdir=...`, `--resume_xid=...`, `--config.*=...` | Passed through to `xm_launcher.py`. |

Behaviour:

1. If `--power`, run the router → pick top-1 → set `group` + `tpu_type`.
2. Run preflight on the resolved `(group, tpu_type, tier)`:
   - **RED**: refuse to submit (unless `--force`).
   - **YELLOW**: print warnings, proceed.
   - **GREEN**: proceed silently.
3. Snapshot code to CitC, run `xmanager launch xm_launcher.py`.
4. Register the XID in `~/.tpu_jobs.json` for `tpu check` to pick up.

### `tpu preflight` / `tpu pf` — run just the check

```
tpu preflight --tpu_type=v6e-16 --group=5 --tier=PROD [--json] [--offline]
```

Exit code: 0 for GREEN/YELLOW, 1 for RED, 2 for internal error.

Three layers of check, all client-side:

- **L1 (µs, in-process)**: is the topology legal for this arch? Does the
  alloc's PROD tier require a bigger min slice than you asked for?
- **L2 (~1s, one RPC)**: does *any* cell in this alloc+tier have ≥ N
  obtainable chips right now? (via `GoodputService.GetCellAvailability`)
- **L2.5 (heuristic)**: PROD quota headroom check —
  `remaining_quota < 2 × request` → YELLOW warning.

Sample output (GREEN):

```
✅ preflight: GREEN
    · alloc quota: 14966 chips (used=0, remaining=14966)
    · candidate cells (chips obtainable): yucbfrl(29578), yucbflq(8696), ...
```

Sample output (RED, L1 policy):

```
❌ preflight: RED
    · Allocator 'group:deepmind-dynamic/vqfree-xm' at tier PROD enforces a
      min slice of v6e-16, but you requested v6e-8. Requesting below the
      minimum is immediately rejected by the allocator.
```

**What preflight does NOT catch** — the "no contiguous 4x4 slice"
fragmentation case. That requires `BorgMaster.ProbeSliceAvailability`,
which has no usable Python stubby wrapper at the moment. Such rejections
still happen post-packaging and are handled by the daemon retry loop.

### `tpu route` / `tpu r` — recommend (group, tpu_type)

```
tpu route --power=v5p-32 [--tier=PROD] [--groups=1,3,5] [--top=3] [--verbose] [--json]
```

Given a target compute level, fan out preflight checks over all
(candidate arch × candidate group) combinations in parallel (8 workers)
and rank the survivors.

Power equivalence heuristic (v5p-chip units):

- 1 v4 chip = 1 v5p chip = 1 unit
- 1 v6e chip = 2 v5p chips
- 1 v6p chip = 2 v5p chips
- 1 v5e chip = 0.5 v5p chip

So `--power=v5p-32` = `--power=v6e-16` = `--power=v4-32` = `--power=32`.

Ranking:

1. Verdict status (GREEN > YELLOW; RED filtered out).
2. `remaining_quota / requested_chips` (headroom ratio, bigger better).
3. `max_cell_obtainable / requested_chips`.
4. Arch preference: v6e > v6p > v5p > v4 > v5e (prefer newer).

Sample:

```
Router recommendations for power=v5p-32 @ PROD

  rank  group tpu_type     status   quota      headroom     reasons
  -----------------------------------------------------------------
  1     g3    v6e-16       GREEN    15364      15364/16=960x -
  2     g5    v6e-16       GREEN    14948      14948/16=934x -
  3     g3    v5p-32       GREEN    20905      20905/32=653x -
```

### `tpu check` / `tpu c` — job status board

```
tpu check              # active + pending + last 10 done
tpu check -a           # all done
tpu check -f           # do not truncate long names
tpu check -d 30        # last 30 done
```

Reads two caches:
- `~/.tpu_jobs.json` (wrapper's own registration; includes tier,
  alloc, launch log path).
- `~/.tpu_check_cache.txt` (background daemon's parsed
  `infra_check` output; ANSI table format).

Merged into three sections: **active**, **pending**, **recent done**.
`WHY` column classifies failures per `xmanager.md`.

### `tpu quota` — guaranteed PROD/BATCH quotas

```
tpu quota          # aggregated across all my groups
tpu quota -l       # per-group breakdown, all groups
tpu quota -g 5     # just G5
```

Backed by `~/.tpu_quota_cache_dir/*.txt` (updated every 60s by daemon).
Complains loudly if cache is >120s stale (auth failure or dead daemon).

**Empty tiers are hidden.** PROD is always printed. BATCH and SPOT are
suppressed when every row has quota 0 *and* usage 0 — which is the normal
state for these allocs, where BATCH floors are ~0 and SPOT has no floor by
construction. A one-line footnote records which tiers were hidden, and they
reappear automatically the moment a floor is granted or a job actually runs
there. Note this hides only the *quota floor* view: BATCH capacity is still
reachable via the free-pool auction, so use `tpu money` and `tpu preflight`
— not this table — to judge whether a BATCH submit can land.

### `tpu money` / `tpu m` / `tpu price` — GQM bidding power + prices

```
tpu money
```

Shows:

1. **MDB Groups Money Table**: bidding power (credits/hr) per group +
   current PROD/BATCH chip usage.
2. **Clearing Prices in Your Pools**: BATCH + PROD clearing prices for
   v4/v5p/v6e/v6p, **filtered to the pools you actually participate in**
   (deepmind-dynamic-pool etc), min–max + median, plus a few sample
   cells. Free pool (0.00) is annotated explicitly. Card types are
   separated by horizontal rules, since each contributes a PROD and a
   BATCH row and the wrapped price/cell text otherwise runs together.

The **Sample cells** column is a price-stratified sample — 2 dearest,
2 median, 2 cheapest, one labelled line each — rather than the top-N by
price. Prices cluster hard: v6p typically shows 2 cells at ~25000, 17 at
~59 and 12 at 0.00, so a dearest-only sample filled all four slots with
five-figure outliers and hid the free cells entirely. The cheap band is
the one that answers "where can this actually land". With fewer than 6
priced cells the column degrades to a plain cheapest-first list instead
of repeating a cell across bands.

Each cell is coloured individually against the limit order: **green** =
clears at or below the cap, so a job there can be admitted; **red** =
clears above the cap and would be stranded in `TRIGGERED_LIMIT_ORDER`.
With no cap in force every cell is green. The band labels are dim on
purpose so the only colour in the column carries the reachable/blocked
signal. Both this and the `Limit order` verdict read the cap through one
helper (`_resolve_cap`), so they cannot disagree.

Groups on a static (non-GQM) pool render as a bare `0.0 (Static Pool)`
rather than `0.0 Credits/hr (Static Pool)`: the long form wrapped onto a
second line for every such group and doubled the table height for no
added signal.

Understanding the output:

- **PROD price ≠ admission cost**. PROD is quota-gated; the "PROD price"
  is a shadow signal of how contested the (cell, accelerator) is.
- **The `Limit order` column is a price cap, and it bites before
  scheduling.** A job whose cell clears above the cap is moved to
  `BUCKET_ID_TRIGGERED_LIMIT_ORDER`, which bypasses the main scheduling
  process — it is pulled from the queue *before* any capacity check, so
  free quota and free chips do not help. Three states: `N ok` (cap above
  every observed price), `N blocks dear cells` (cap sits inside the price
  range — the cheap cells still clear), `N BLOCKS ALL` (cap below even the
  cheapest cell — nothing can land). The name in parentheses is whoever
  set it: caps are **MDB-scoped**, so a teammate's cap silently applies to
  everyone in the group (resolution order SCU > XID > MDB).
- **BATCH admission is combinatorial**, not `bp ≥ price`. `bp ≥ price ×
  chips` is a necessary-but-not-sufficient sanity check. See
  `wiki_agents/xmanager.md § GQM Bidding Power`.
- **Free pool BATCH is still preemptible**. Only Churn Protection (~8h
  window) provides any stability shield.

### `tpu monitor` / `tpu watch` — live-refresh `tpu check`

```
tpu monitor              # loops every 5s, forwards args to `tpu check`
```

Ctrl-C to quit.

---

## File layout

```
~/work/tpu_cmd/
├── README.md              (this file)
├── tpu_wrapper.sh         (all subcommand routing + `tpu queue` orchestrator)
└── xm_launcher.py         (XM entry: parses --tpu_type, --tier, --config, etc,
                            builds JobRequirements, calls experiment.package())

~/work/tpu_check_daemon.sh (60s poll loop: infra_check + quota_check +
                            money_check + PROD auto-retry driver)

/google/src/cloud/qiaos/xm_test/google3/experimental/users/qiaos/tpu_utils/
├── BUILD
├── group_utils.py         (group_id ↔ alloc string mappings)
├── infra_check.py         (parses XManager API → tpu check tables)
├── quota_check.py         (list_resources → tpu quota cache files)
├── money_check.py         (Spanner ResourcePrices → tpu money cache)
├── inspect_gqm.py         (ad-hoc GQM debugging)
└── preflight/
    ├── BUILD
    ├── topology.py        (L1: legal topology table + min-slice policy)
    ├── capacity.py        (L2: GetCellAvailability + list_resources)
    ├── preflight.py       (orchestrator + Verdict type)
    ├── preflight_cli.py   (`tpu preflight` entry)
    ├── router.py          (power class → (group, tpu_type) ranking)
    └── router_cli.py      (`tpu route` entry)

~/.tpu_jobs.json           (per-XID: tpu_type, tier, alloc, logdir, stagedir,
                            launch_log, exp_name, status, error, retry_count)
~/.tpu_quota_cache_dir/*   (quota + money cached tables, updated every 60s)
~/.tpu_check_cache.txt     (infra_check tables, updated every 60s)
```

Blaze binaries are looked up at `blaze-bin/experimental/users/qiaos/tpu_utils/`
under the CitC workspace `/google/src/cloud/qiaos/xm_test`. Rebuild after
editing:

```bash
cd /google/src/cloud/qiaos/xm_test/google3
blaze build experimental/users/qiaos/tpu_utils/preflight:preflight_cli
blaze build experimental/users/qiaos/tpu_utils/preflight:router_cli
blaze build experimental/users/qiaos/tpu_utils:money_check
blaze build experimental/users/qiaos/tpu_utils:quota_check
blaze build experimental/users/qiaos/tpu_utils:infra_check
```

---

## Common workflows

### "I have a small model, ~half a v5p rack of compute"

```bash
tpu route --power=v5p-32 --tier=PROD          # inspect the ranking
tpu queue --power=v5p-32 --tier=PROD --config=my_config
```

### "I know exactly what I want"

```bash
tpu queue --tpu_type=v6e-16 --group=5 --tier=PROD --config=my_config
```

Preflight runs automatically. On RED it refuses; on YELLOW it proceeds
with a warning; on GREEN it goes straight through.

### "Preflight is wrong / I want to try anyway"

```bash
tpu queue ... --force              # ignore RED verdict
tpu queue ... --skip-preflight     # don't even run the check
```

### "Someone rejected my job; is my quota OK?"

```bash
tpu quota -g <N>         # per-group table
tpu preflight ...        # will report headroom + candidate cells
tpu money                # in-pool clearing prices for context
```

### "Something's stuck / cache is stale"

The `tpu-daemon` tmux session runs the 60s poll. Restart if quota /
money tables are >120s stale:

```bash
tmux kill-session -t tpu-daemon
tmux new-session -d -s tpu-daemon 'bash -c "while true; do ~/work/tpu_check_daemon.sh; sleep 5; done"'
```

`tpu quota` / `tpu money` auto-restart the daemon when cache is stale,
so this is usually only needed when `gcert` credentials just expired.

---

## Known issues / gaps

- **L3 topology fragmentation not covered**. Preflight can say "enough
  chips are obtainable in cell X" but not "a contiguous 4×4 v6e slice
  actually exists in cell X". If your PROD v6e-16 job passes preflight
  and still fails with `Rejected by Allocator/Borg`, this is almost
  certainly the reason. The daemon auto-retries these up to 5×.
- **`~/.tpu_jobs.json` merge bug**: `xm_launcher.py::update_mapping()`
  overwrites the entry instead of merging, so `tier`/`alloc` written by
  the shell wrapper get clobbered. Symptom: `tpu check` shows `TIER=-`
  and `GROUP=-` for jobs submitted via `tpu queue`. Not yet fixed.
- **rich `Console(width=...)` needs `height=` too.** `Console.size`
  short-circuits to a hard-coded 80×25 on a dumb terminal (`TERM=dumb`,
  i.e. any non-tty daemon or agent shell) unless BOTH width and height
  are set. Symptom: tables silently render 80 cols wide and wrap, but
  only when regenerated outside tmux. Both `quota_check.py` and
  `money_check.py` now pass `height=200`.
- **`tpu quota` BATCH `Obtainable` disagrees with preflight.** The quota
  table can report five figures of obtainable BATCH chips for an alloc
  where `tpu preflight --tier=BATCH` finds 0 eligible cells; the two read
  different APIs (`get_forecast_info` with global-batch availability vs
  `GoodputService.GetCellAvailability`). Trust preflight before
  submitting.
- **`money_check.py`** now filters to your own pools, but the pool
  discovery via `resource_service.get_resource_alloc` is best-effort;
  static/legacy pools you belong to may be missing.
- **PROD attribution prompt**: first-time submissions to a new alloc
  trigger a Research Hub attribution prompt from `xmanager` itself; that
  is unrelated to preflight and cannot be side-stepped here. Set
  `attribution_urls=` in `xm_launcher.py` to skip.

---

## Under the hood

Read `~/work/wiki_agents/xmanager.md` for:

- The full PROD vs BATCH admission model
- GQM market-cycle mechanics (1-minute uniform clearing auction,
  free-pool semantics, bidding-power formula)
- The XM allocator reject taxonomy (`FLEX_CEILING_EXCEEDED`,
  `DEFICIT_IN_PARENT_POOLS`, `SLICE_DEFRAGMENTATION`, etc.)
- Why preflight is fundamentally best-effort and daemon retry is the
  real backstop

Research notes on how we arrived at the current design:

- `$AMPLY_ARTIFACT_DIR/research_allocator.md` — XM/Borg admission pipeline
- `$AMPLY_ARTIFACT_DIR/research_topology.md` — TPU topology / cell-level APIs
- `$AMPLY_ARTIFACT_DIR/research_gqm.md` — GQM bidding power + auction
