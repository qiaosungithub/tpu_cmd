#!/usr/bin/env bash
# The check-board daemon for lyy's registry (the `npu` half of tpu_wrapper.sh).
#
#   tmux new-session -d -s npu-daemon -c ~/work/tpu_cmd \
#     'while true; do bash run-npu-daemon.sh; echo "restarting in 5s"; sleep 5; done'
#
# `npu check` renders from $TPU_CHECK_CACHE_FILE. Without this process nothing
# ever writes that file, so every one of lyy's jobs reads SUBMITTED forever no
# matter what it is really doing — the board looks alive and is not.
#
# This is the SAME script as sqa's daemon, pointed at lyy's three files. Do not
# copy the daemon; a second copy is a second thing to keep in sync.
set -euo pipefail
cd "$(dirname "$0")"

export TPU_JOBS_FILE="${NPU_JOBS_FILE:-$HOME/lyy-work/.npu_jobs.json}"
export TPU_CHECK_CACHE_FILE="${NPU_CHECK_CACHE_FILE:-$HOME/lyy-work/.npu_check_cache.txt}"
export TPU_CHECK_TIME_FILE="${NPU_CHECK_TIME_FILE:-$HOME/lyy-work/.npu_check_time.txt}"
# The ARCHIVE ledger, and the one file of this set that was missing (added
# 2026-09-03). Unset, the daemon inherits the wrapper default and writes lyy's
# finished jobs into sqa's ~/.tpu_jobs_legacy.json -- silently, since appending
# to the wrong ledger looks exactly like appending to the right one.
# npu_dispatch_worker.sh:40 has always set it; the daemon did not, so which
# ledger a job landed in depended on which process archived it.
export TPU_JOBS_LEGACY_FILE="${NPU_JOBS_LEGACY_FILE:-$HOME/lyy-work/.npu_jobs_legacy.json}"

# CRITICAL SCOPING (2026-08-24): this daemon must NOT touch the shared default
# local queue ~/.tpu_local_queue.json. Without these two lines it inherited the
# tpu_check_daemon.sh default (TPU_LOCAL_QUEUE_FILE:=~/.tpu_local_queue.json)
# and its LIVE route lane (TPU_ROUTE_ENABLED:=1) ran place/reroute on the SAME
# file sqa's daemon owns -- two route writers whole-queue-overwriting each other,
# which silently clobbered rows other lines had just `tpu enqueue`d (paligemma
# enc10410 vanished 3x). lyy's daemon only needs the CHECK board (infra/money/
# quota); routing the shared queue is exclusively sqa's tmux tpu-daemon.
#   (1) give npu its own queue file (isolated even if the lane is ever enabled);
#   (2) hard-disable the route lane here so there is a SINGLE route writer.
export TPU_LOCAL_QUEUE_FILE="${NPU_LOCAL_QUEUE_FILE:-$HOME/lyy-work/.npu_local_queue.json}"
# ═══ 【2026-08-26】lyy 明确要求打开这条 lane;(2) 的前提已经不成立 ═══════════
#
# 上面 (2) 关掉它,是因为当时两个 daemon 共用 ~/.tpu_local_queue.json,两个 route
# writer 整队覆写、抹掉别人刚 enqueue 的行(paligemma enc10410 消失 3 次)。
# 而 (1) 已经把 npu 的队列隔离到 ~/lyy-work/.npu_local_queue.json —— 原注释自己
# 就写着「isolated even if the lane is ever enabled」,预留了这一天。
#
# 08-26 实测两侧确实分开(读 /proc/*/environ):
#   npu daemon  queue=~/lyy-work/.npu_local_queue.json  route=0
#   sqa daemon  queue=<默认 ~/.tpu_local_queue.json>     route=<默认 1>
# 打开后仍是「每个 daemon 只路由自己的队列」,单文件单 writer 的性质不变。
#
# 为什么要打开:sqa 的 budget_enforcer.py 取消一条臂后打印
# 「cancelled + re-queued as resume」并把 resume 写进 npu 的队列 —— 那句话在 sqa
# 侧是真的(他的 lane 开着),在 npu 侧只有前半句是真的。队列因此只进不出
# (08-26 实测 2 条,attempts=0、xid=None,从未被尝试),而**没有任何东西会报错**。
# 被 enforcer 取消 = 永久停止。这正是本线反复吃亏的「输出说做了、底下没发生」。
# 两个组件各自都按设计工作,合起来才是死的。
#
# 回退:把下面这行改回 0。lane 的其余默认值未动(DRYRUN=0 live、GROUP=9)。
export TPU_ROUTE_ENABLED=1
# Standalone npu-reroute (tmux) owns reconcile+reroute. In-lane --reroute
# never ran --reconcile, which is how 59 XM-COMPLETED rows sat SUBMITTED for days.
export TPU_ROUTE_INLANE_REROUTE=0

exec bash "$PWD/tpu_check_daemon.sh"
