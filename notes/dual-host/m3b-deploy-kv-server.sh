#!/bin/bash
# m3b-deploy-kv-server.sh — copy ds4-kv-server to the MacBook replica and
# start it under nohup over SSH. Print the listen address for the host side.
#
# This script is run on the HOST (Mac Mini). The replica (MacBook) is
# assumed to:
#   - already accept the host SSH key (see setup-replica.sh)
#   - have port 17501 free on the bridge interface
#   - have nothing destructive in ~/ds4-kv-demo (the dir is created if missing
#     and only the binary is dropped; no existing files are removed)
#
# IRON RULE: this script never deletes anything on the replica. If the binary
# at the destination is already present, scp overwrites it; we never rm.
#
# Usage:
#   ./notes/dual-host/m3b-deploy-kv-server.sh             # default user=fodelf, host=192.168.1.2
#   REPL_USER=foo REPL_HOST=10.0.0.5 REPL_PORT=18000 ./notes/dual-host/m3b-deploy-kv-server.sh
#
# After this script exits, take the printed --kv-offload value and pass it
# to ds4-server. Stop the remote process with:
#   ssh fodelf@192.168.1.2 'pkill -f ds4-kv-server'
# (or read /tmp/m3b-kv-server.pid on the replica and kill that PID)

set -u

REPL_USER="${REPL_USER:-fodelf}"
REPL_HOST="${REPL_HOST:-192.168.1.2}"
REPL_PORT="${REPL_PORT:-17501}"
REPL_DIR="${REPL_DIR:-/Users/${REPL_USER}/ds4-kv-demo}"
REPL_BIND="${REPL_BIND:-${REPL_HOST}}"
REPL_MEM_GIB="${REPL_MEM_GIB:-3}"
LOCAL_BIN="${LOCAL_BIN:-./ds4-kv-server}"

echo "m3b-deploy: replica = ${REPL_USER}@${REPL_HOST}:${REPL_PORT} (bind ${REPL_BIND})"
echo "m3b-deploy: dir     = ${REPL_DIR}"
echo "m3b-deploy: cap     = ${REPL_MEM_GIB} GiB"
echo

if [ ! -x "$LOCAL_BIN" ]; then
    echo "m3b-deploy: ERROR ${LOCAL_BIN} not found; run 'make ds4-kv-server' first" >&2
    exit 2
fi

echo "[1/4] Pinging replica via SSH"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${REPL_USER}@${REPL_HOST}" 'uname -ms' ; then
    echo "m3b-deploy: ERROR cannot SSH to ${REPL_USER}@${REPL_HOST}" >&2
    exit 3
fi
echo

echo "[2/4] Ensuring ${REPL_DIR} exists (mkdir -p, never rm)"
ssh "${REPL_USER}@${REPL_HOST}" "mkdir -p '${REPL_DIR}'"
echo

echo "[3/4] scp ${LOCAL_BIN} -> ${REPL_USER}@${REPL_HOST}:${REPL_DIR}/ds4-kv-server"
scp -p "$LOCAL_BIN" "${REPL_USER}@${REPL_HOST}:${REPL_DIR}/ds4-kv-server"
echo

echo "[4/4] Stopping any prior ds4-kv-server (pkill — no file deletion) and starting fresh"
# We pkill the PROCESS only; this does not touch any files on disk and is
# consistent with feedback-no-remote-file-delete (process kill is allowed).
ssh "${REPL_USER}@${REPL_HOST}" "
    pkill -f 'ds4-kv-server listen' 2>/dev/null
    sleep 0.5
    cd '${REPL_DIR}'
    nohup ./ds4-kv-server listen ${REPL_PORT} ${REPL_BIND} --mem-limit-gib ${REPL_MEM_GIB} \
        > '${REPL_DIR}/server.log' 2>&1 &
    echo \$! > '${REPL_DIR}/server.pid'
    sleep 0.5
    if kill -0 \$(cat '${REPL_DIR}/server.pid') 2>/dev/null; then
        echo \"OK: ds4-kv-server pid=\$(cat '${REPL_DIR}/server.pid')\"
        head -3 '${REPL_DIR}/server.log' 2>/dev/null || true
    else
        echo 'FAIL: ds4-kv-server died at startup, last 20 log lines:' >&2
        tail -20 '${REPL_DIR}/server.log' >&2
        exit 4
    fi
"
echo

echo "============================================================"
echo "Replica is up. Pass to ds4-server:"
echo "    --kv-offload ${REPL_HOST}:${REPL_PORT} --kv-offload-rows-per-token 1"
echo
echo "Tail replica log live:"
echo "    ssh ${REPL_USER}@${REPL_HOST} tail -f ${REPL_DIR}/server.log"
echo
echo "Stop replica (process only, no file delete):"
echo "    ssh ${REPL_USER}@${REPL_HOST} \"kill \\\$(cat ${REPL_DIR}/server.pid)\""
echo "============================================================"
