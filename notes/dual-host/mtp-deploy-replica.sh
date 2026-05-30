#!/bin/bash
# mtp-deploy-replica.sh — copy ds4-mtp-replica + Metal kernels to the MacBook
# replica and start the off-host MTP drafter under nohup over SSH.
#
# Run on the HOST (Mac Mini). The replica (MacBook) must:
#   - already accept the host SSH key (see setup-replica.sh)
#   - have the base GGUF + MTP GGUF present (see REPL_MODEL / REPL_MTP below);
#     they are NOT shipped by this script (86 GiB base is the user's call)
#   - have the chosen port free on the bridge interface
#
# IRON RULE: this script never deletes anything on the replica. mkdir -p makes
# the dir if missing; scp overwrites the binary + kernels; we never rm. Stopping
# the drafter is a process kill (pkill), which is not a file deletion.
#
# MEMORY-SAFETY RULE: the replica only fits 8 GiB if Metal view-shrink wires just
# the embd + output views (~5.1 GiB), NOT the whole 86 GiB base. This script will
# NOT let the box thrash/panic on a regression: it refuses to start if the remote
# RAM can't hold the working set, and arms a detached RSS watchdog that SIGKILLs
# the replica the instant resident size crosses MEM_CEIL_MIB. See the memory note
# feedback_memory_safety_gate. Tune via MEM_NEED_MIB / MEM_CEIL_MIB below.
#
# Usage:
#   ./notes/dual-host/mtp-deploy-replica.sh
#   REPL_USER=foo REPL_HOST=10.0.0.5 REPL_PORT=17502 \
#     REPL_MODEL=~/m/base.gguf REPL_MTP=~/m/mtp.gguf \
#     ./notes/dual-host/mtp-deploy-replica.sh
#
# After this exits, pass the printed value to ds4-server / ds4:
#   --mtp-remote HOST:PORT --mtp-draft 2

set -u

REPL_USER="${REPL_USER:-fodelf}"
REPL_HOST="${REPL_HOST:-192.168.1.2}"
REPL_PORT="${REPL_PORT:-17502}"
REPL_DIR="${REPL_DIR:-/Users/${REPL_USER}/ds4-mtp-replica}"
REPL_BIND="${REPL_BIND:-${REPL_HOST}}"
REPL_CTX="${REPL_CTX:-8192}"
# Model paths AS SEEN ON THE REPLICA. The base GGUF was already copied for M3a
# (e.g. ~/ds4-main/ds4flash.gguf); the MTP GGUF must be present too.
REPL_MODEL="${REPL_MODEL:-/Users/${REPL_USER}/ds4-main/ds4flash.gguf}"
REPL_MTP="${REPL_MTP:-/Users/${REPL_USER}/ds4-main/ds4flash-mtp.gguf}"
LOCAL_BIN="${LOCAL_BIN:-./ds4-mtp-replica}"

# Memory-safety budget (see header). The replica must stay near the view-shrunk
# working set: embd ~1.06 + output ~0.56 + MTP ~3.5 GiB. If RSS climbs toward the
# full base, view-shrink regressed — kill it before it OOMs the box.
MEM_NEED_MIB="${MEM_NEED_MIB:-5222}"   # expected resident ~5.1 GiB
MEM_CEIL_MIB="${MEM_CEIL_MIB:-6656}"   # hard RSS ceiling 6.5 GiB -> SIGKILL above

echo "mtp-deploy: replica = ${REPL_USER}@${REPL_HOST}:${REPL_PORT} (bind ${REPL_BIND})"
echo "mtp-deploy: dir     = ${REPL_DIR}"
echo "mtp-deploy: base    = ${REPL_MODEL}"
echo "mtp-deploy: mtp     = ${REPL_MTP}"
echo "mtp-deploy: ctx     = ${REPL_CTX}"
echo

if [ ! -x "$LOCAL_BIN" ]; then
    echo "mtp-deploy: ERROR ${LOCAL_BIN} not found; run 'make ds4-mtp-replica' first" >&2
    exit 2
fi

echo "[1/5] Pinging replica via SSH"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${REPL_USER}@${REPL_HOST}" 'uname -ms' ; then
    echo "mtp-deploy: ERROR cannot SSH to ${REPL_USER}@${REPL_HOST}" >&2
    exit 3
fi
echo

echo "[2/5] Ensuring ${REPL_DIR} and ${REPL_DIR}/metal exist (mkdir -p, never rm)"
ssh "${REPL_USER}@${REPL_HOST}" "mkdir -p '${REPL_DIR}/metal'"
echo

echo "[3/5] scp binary + Metal kernels (overwrite, never rm)"
scp -p "$LOCAL_BIN" "${REPL_USER}@${REPL_HOST}:${REPL_DIR}/ds4-mtp-replica"
scp -p metal/*.metal "${REPL_USER}@${REPL_HOST}:${REPL_DIR}/metal/"
echo

echo "[4/5] Checking model files exist on replica (no transfer — too large to assume)"
ssh "${REPL_USER}@${REPL_HOST}" "
    for f in '${REPL_MODEL}' '${REPL_MTP}'; do
        if [ ! -f \"\$f\" ]; then
            echo \"mtp-deploy: ERROR missing on replica: \$f\" >&2
            exit 5
        fi
    done
    echo 'OK: base + MTP GGUF present on replica'
" || exit 5
echo

echo "[5/5] Memory preflight, then stop prior replica (pkill — no file deletion) and start guarded"
# pkill the PROCESS only; consistent with feedback-no-remote-file-delete.
# Metal kernels are loaded relative to the binary's cwd, so cd into REPL_DIR.
#
# Memory safety (feedback_memory_safety_gate): before launching, refuse if the
# replica box can't hold the view-shrunk working set (MEM_NEED_MIB + margin).
# After launching, arm a DETACHED RSS watchdog that polls resident size and
# SIGKILLs the replica the instant it crosses MEM_CEIL_MIB — this is the brake
# that stops a view-shrink regression (IOGPU wiring the full 86 GiB base) from
# thrashing swap or panicking the box. The watchdog runs under setsid/nohup so
# it survives this SSH session closing.
ssh "${REPL_USER}@${REPL_HOST}" "
    # --- RAM preflight: abort if the box is too small for the working set ---
    page=\$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
    membytes=\$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    mem_total_mib=\$(( membytes / 1048576 ))
    need_with_margin=\$(( ${MEM_NEED_MIB} + 1024 ))
    echo \"mem: replica box has \${mem_total_mib} MiB; need \${need_with_margin} MiB (working set ${MEM_NEED_MIB} + 1024 margin); kill ceiling ${MEM_CEIL_MIB} MiB\"
    if [ \"\${mem_total_mib}\" -gt 0 ] && [ \"\${mem_total_mib}\" -lt \"\${need_with_margin}\" ]; then
        echo \"mtp-deploy: ERROR replica RAM \${mem_total_mib} MiB < required \${need_with_margin} MiB — refusing to launch (memory-safety gate)\" >&2
        exit 6
    fi

    pkill -f 'ds4-mtp-replica listen' 2>/dev/null
    sleep 0.5
    cd '${REPL_DIR}'
    DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_MODEL_MAX_VIEW_BYTES=2147483648 \
    DS4_METAL_NO_RESIDENCY=1 DS4_METAL_NO_MODEL_WARMUP=1 DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
    nohup ./ds4-mtp-replica listen ${REPL_PORT} ${REPL_BIND} \
        -m '${REPL_MODEL}' --mtp '${REPL_MTP}' -c ${REPL_CTX} \
        > '${REPL_DIR}/replica.log' 2>&1 &
    rpid=\$!
    echo \$rpid > '${REPL_DIR}/replica.pid'

    # --- detached RSS watchdog: SIGKILL the instant resident > ceiling ---
    # ps RSS is in KiB on macOS; compare in MiB. Poll every 0.5s. Exits when the
    # replica is gone. Logs the kill into replica.log so the cause is visible.
    nohup sh -c '
        rp='\"\$rpid\"'
        ceil='\"${MEM_CEIL_MIB}\"'
        while kill -0 \$rp 2>/dev/null; do
            rss_kib=\$(ps -o rss= -p \$rp 2>/dev/null | tr -d \" \")
            [ -z \"\$rss_kib\" ] && break
            rss_mib=\$(( rss_kib / 1024 ))
            if [ \$rss_mib -gt \$ceil ]; then
                echo \"mtp-watchdog: RSS \${rss_mib} MiB > ceiling \${ceil} MiB — SIGKILL pid \$rp (view-shrink regression?)\" >> '${REPL_DIR}/replica.log'
                kill -9 \$rp 2>/dev/null
                break
            fi
            sleep 0.5
        done
    ' >> '${REPL_DIR}/replica.log' 2>&1 &
    echo \$! > '${REPL_DIR}/replica.watchdog.pid'
    echo \"mem: RSS watchdog armed (pid \$(cat '${REPL_DIR}/replica.watchdog.pid'), ceiling ${MEM_CEIL_MIB} MiB)\"

    sleep 3
    if kill -0 \$rpid 2>/dev/null; then
        echo \"OK: ds4-mtp-replica pid=\$rpid (model still loading — watch the log)\"
        tail -5 '${REPL_DIR}/replica.log' 2>/dev/null || true
    else
        echo 'FAIL: ds4-mtp-replica died at startup (OOM watchdog or load error), last 20 log lines:' >&2
        tail -20 '${REPL_DIR}/replica.log' >&2
        exit 4
    fi
"
echo

echo "============================================================"
echo "Replica drafter starting. The base GGUF mmap + Metal init can take a"
echo "while; wait for a 'listening on' line in the log before connecting:"
echo "    ssh ${REPL_USER}@${REPL_HOST} tail -f ${REPL_DIR}/replica.log"
echo
echo "Then pass to ds4-server / ds4:"
echo "    --mtp-remote ${REPL_HOST}:${REPL_PORT} --mtp-draft 2"
echo
echo "Stop replica (process only, no file delete):"
echo "    ssh ${REPL_USER}@${REPL_HOST} \"kill \\\$(cat ${REPL_DIR}/replica.pid)\""
echo "============================================================"
