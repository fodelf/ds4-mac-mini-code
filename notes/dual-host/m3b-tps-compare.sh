#!/bin/bash
# m3b-tps-compare.sh — drive a running ds4-server and record per-token wire
# impact of --kv-offload-rows-per-token. The script does NOT start ds4-server
# itself (that's a heavy 86 GiB GGUF load — the user controls when to run it).
#
# Two-phase workflow:
#
#   PHASE A: baseline (no offload pushes)
#     Terminal 1:
#         ./ds4-server -m ds4flash.gguf -c 32768 --port 8000
#     Terminal 2:
#         ./notes/dual-host/m3b-tps-compare.sh baseline
#
#   PHASE B: offload (per-token wire push)
#     Terminal 1 (Ctrl+C the baseline first, then):
#         ./notes/dual-host/m3b-deploy-kv-server.sh    # one-time setup
#         ./ds4-server -m ds4flash.gguf -c 32768 --port 8000 \
#             --kv-offload 192.168.1.2:17501 --kv-offload-rows-per-token 1
#     Terminal 2:
#         ./notes/dual-host/m3b-tps-compare.sh offload
#
# After both runs the script prints the t/s delta and per-token push estimate.

set -u

LABEL="${1:-baseline}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
N_REQUESTS="${N_REQUESTS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT="${PROMPT:-Write a short paragraph about Thunderbolt 4 bandwidth.}"
OUT_DIR="${OUT_DIR:-/tmp}"
RESULTS_FILE="${OUT_DIR}/m3b-${LABEL}.txt"

echo "m3b-tps-compare: label=${LABEL}  server=${SERVER_URL}  N=${N_REQUESTS}  max_tokens=${MAX_TOKENS}"
echo "m3b-tps-compare: results -> ${RESULTS_FILE}"
echo

if ! curl -fsS -m 2 "${SERVER_URL}/v1/models" >/dev/null 2>&1; then
    echo "m3b-tps-compare: ERROR ds4-server not reachable at ${SERVER_URL}" >&2
    echo "                 start it first (see header of this script)" >&2
    exit 2
fi

: > "$RESULTS_FILE"
echo "# m3b-tps-compare $LABEL @ $(date '+%F %T')" >> "$RESULTS_FILE"
echo "# prompt=$(printf %s "$PROMPT" | wc -c) bytes  max_tokens=${MAX_TOKENS}  N=${N_REQUESTS}" >> "$RESULTS_FILE"

ESCAPED_PROMPT=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")
PAYLOAD=$(cat <<EOF
{
  "model": "ds4flash",
  "messages": [{"role": "user", "content": ${ESCAPED_PROMPT}}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0,
  "stream": false
}
EOF
)

declare -a WALLS
declare -a TOKEN_COUNTS
total_tokens=0
total_wall=0

for i in $(seq 1 "$N_REQUESTS"); do
    echo "[run $i/$N_REQUESTS] sending request ..."
    start_ns=$(python3 -c 'import time; print(time.monotonic_ns())')
    resp=$(curl -fsS -X POST "${SERVER_URL}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d "$PAYLOAD")
    rc=$?
    end_ns=$(python3 -c 'import time; print(time.monotonic_ns())')
    if [ "$rc" -ne 0 ] || [ -z "$resp" ]; then
        echo "[run $i] curl failed rc=$rc"
        echo "[run $i] curl failed rc=$rc" >> "$RESULTS_FILE"
        continue
    fi
    wall_ms=$(python3 -c "print((${end_ns}-${start_ns})/1e6)")
    # Token count comes from the OpenAI usage block when present.
    n_tok=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); u=d.get("usage",{}); print(u.get("completion_tokens",0))' <<<"$resp")
    if [ -z "$n_tok" ] || [ "$n_tok" = "0" ]; then
        # Fallback: count whitespace-separated words in the content.
        n_tok=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(len(d["choices"][0]["message"]["content"].split()))' <<<"$resp")
    fi
    tps=$(python3 -c "n=$n_tok; w=$wall_ms; print(f'{(n*1000.0/w if w>0 else 0):.3f}')")
    echo "[run $i] wall=${wall_ms} ms  tokens=${n_tok}  tps=${tps}"
    echo "run=$i wall_ms=${wall_ms} tokens=${n_tok} tps=${tps}" >> "$RESULTS_FILE"
    WALLS+=("$wall_ms")
    TOKEN_COUNTS+=("$n_tok")
    total_tokens=$((total_tokens + n_tok))
    total_wall=$(python3 -c "print($total_wall + $wall_ms)")
done

avg_tps=$(python3 -c "n=$total_tokens; w=$total_wall; print(f'{(n*1000.0/w if w>0 else 0):.3f}')")
echo
echo "============================================================"
echo "${LABEL} summary: total_tokens=${total_tokens}  total_wall_ms=${total_wall}  avg_tps=${avg_tps}"
echo "============================================================"
echo "summary total_tokens=${total_tokens} total_wall_ms=${total_wall} avg_tps=${avg_tps}" >> "$RESULTS_FILE"

# If both phase files exist, print a diff.
if [ -f "${OUT_DIR}/m3b-baseline.txt" ] && [ -f "${OUT_DIR}/m3b-offload.txt" ]; then
    base_tps=$(awk -F'avg_tps=' '/^summary/ {print $2; exit}' "${OUT_DIR}/m3b-baseline.txt")
    off_tps=$(awk -F'avg_tps=' '/^summary/ {print $2; exit}' "${OUT_DIR}/m3b-offload.txt")
    echo
    echo "============================================================"
    echo "DELTA: baseline=${base_tps} t/s   offload=${off_tps} t/s"
    python3 -c "b=float('${base_tps}' or 0); o=float('${off_tps}' or 0); print(f'wire overhead = {(b-o):+.3f} t/s ({(100*(o-b)/b if b>0 else 0):+.1f}%)')"
    echo
    echo "Per-token push at rows=1: 43 layers x 608 B = 25.5 KiB / token"
    echo "Expected wire byte rate at offload tps: $(python3 -c "import sys; print(f'{float(\"$off_tps\")*43*608/1024:.1f} KiB/s')")"
    echo "============================================================"
fi
