#!/bin/bash
# test-api-smoke.sh — Phase 1 of the Claude-Code-on-ds4 plumbing validation.
#
# Run AFTER ds4-server is up (./serve-32k.sh in another terminal).
# Total wall time: ~3-5 min (prefill ~50 tok @ ~1 t/s + decode 32 tok @ 0.09 t/s).
#
# Validates, in order:
#   1. /v1/models returns deepseek-v4-flash             (~instant)
#   2. /v1/messages non-stream: small Claude-style msg  (~3 min, decode-bound)
#   3. /v1/messages stream (SSE): same request streaming
#
# Stops at the first failure with the curl exit code.  If any of the 3 pass,
# the API surface Claude Code uses is confirmed wired.

set -u

HOST="${DS4_SERVER_HOST:-127.0.0.1}"
PORT="${DS4_SERVER_PORT:-8000}"
BASE="http://$HOST:$PORT"

echo "=== test-api-smoke: target $BASE ==="

# ---- Test 1: GET /v1/models ----
echo
echo "--- Test 1: GET /v1/models ---"
if ! curl -sS --max-time 5 "$BASE/v1/models" -o /tmp/ds4-api-test1.json; then
    echo "FAIL: curl /v1/models failed.  Is ds4-server running on $BASE?"
    exit 1
fi
cat /tmp/ds4-api-test1.json
echo
if ! grep -q "deepseek-v4-flash" /tmp/ds4-api-test1.json; then
    echo "FAIL: response did not contain deepseek-v4-flash."
    exit 1
fi
echo "PASS: /v1/models returned expected model id"

# ---- Test 2: POST /v1/messages non-stream ----
echo
echo "--- Test 2: POST /v1/messages (non-stream, decode 16 tok, ~2 min) ---"
T2_REQ='{
  "model": "claude-sonnet-4-5",
  "max_tokens": 16,
  "system": "You are a terse assistant.",
  "messages": [
    {"role": "user", "content": "Reply with the single word PONG and nothing else."}
  ],
  "thinking": {"type": "disabled"}
}'
echo "request body:"
echo "$T2_REQ"
echo
echo "sending (this may take ~2 min for the model to decode 16 tokens)..."
START=$(date +%s)
if ! curl -sS --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST "$BASE/v1/messages" \
        -d "$T2_REQ" \
        -o /tmp/ds4-api-test2.json; then
    echo "FAIL: curl /v1/messages (non-stream) failed."
    exit 2
fi
END=$(date +%s)
echo "done in $((END - START)) s"
echo "response body:"
cat /tmp/ds4-api-test2.json
echo
# Anthropic Messages API response shape: {"id":...,"type":"message","role":"assistant",
# "content":[{"type":"text","text":"..."}],"model":"...","stop_reason":"...","usage":{...}}
if ! grep -q '"type":"message"' /tmp/ds4-api-test2.json; then
    echo "FAIL: response missing \"type\":\"message\" — wrong shape."
    exit 2
fi
if ! grep -qE '"content":\[' /tmp/ds4-api-test2.json; then
    echo "FAIL: response missing content array."
    exit 2
fi
echo "PASS: /v1/messages returned valid Anthropic-shape response in $((END - START)) s"

# ---- Test 3: POST /v1/messages stream (SSE) ----
echo
echo "--- Test 3: POST /v1/messages (stream=true, decode 16 tok, ~2 min) ---"
T3_REQ='{
  "model": "claude-sonnet-4-5",
  "max_tokens": 16,
  "stream": true,
  "system": "You are a terse assistant.",
  "messages": [
    {"role": "user", "content": "Reply with the single word PONG and nothing else."}
  ],
  "thinking": {"type": "disabled"}
}'
echo "request body:"
echo "$T3_REQ"
echo
echo "streaming response (each line is one SSE event):"
START=$(date +%s)
# -N disables curl's output buffering so SSE events appear live.
if ! curl -sS -N --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST "$BASE/v1/messages" \
        -d "$T3_REQ" \
        | tee /tmp/ds4-api-test3.sse \
        | head -200; then
    echo "FAIL: curl /v1/messages (stream) failed."
    exit 3
fi
END=$(date +%s)
echo
echo "done in $((END - START)) s"
# SSE for Anthropic: lines like "event: message_start\ndata: {...}\n\n", and
# events include message_start / content_block_start / content_block_delta / .
if ! grep -q "event: message_start" /tmp/ds4-api-test3.sse; then
    echo "FAIL: SSE stream missing message_start event."
    exit 3
fi
if ! grep -q "event: content_block_delta" /tmp/ds4-api-test3.sse; then
    echo "FAIL: SSE stream missing content_block_delta event."
    exit 3
fi
if ! grep -q "event: message_stop" /tmp/ds4-api-test3.sse; then
    echo "FAIL: SSE stream missing message_stop event."
    exit 3
fi
echo "PASS: SSE stream contained message_start, content_block_delta, message_stop"

echo
echo "=== ALL 3 API SMOKE TESTS PASSED ==="
echo "ds4-server is serving the Anthropic /v1/messages surface Claude Code expects."
echo "Saved artifacts:"
echo "  /tmp/ds4-api-test1.json  (/v1/models)"
echo "  /tmp/ds4-api-test2.json  (/v1/messages non-stream response)"
echo "  /tmp/ds4-api-test3.sse   (/v1/messages SSE event log)"
echo
echo "Next: Phase 2 — point Claude Code at ds4-server."
echo "  ANTHROPIC_BASE_URL=$BASE ANTHROPIC_API_KEY=dummy claude"
echo "  (warning: first Claude Code turn = ~1 hour, decode = 0.09 t/s)"
