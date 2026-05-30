/* ds4-kv-server — replica-side KV byte store.
 *
 * Minimal sibling of ds4-replica: accepts a TCP connection from the host,
 * receives FP8 attn KV chunks (608 B/row), keeps them in an in-memory
 * chunk table, serves them back on request. Does NOT load the model,
 * does NOT touch Metal/IOGPU, does NOT mmap the GGUF — so it fits in the
 * MacBook's 8 GiB user budget without the host-side ds4 footprint that
 * blew up M3a (see notes/execution-log.md 2026-05-27 M3a entry).
 *
 * Wire protocol: shared with ds4-replica (ds4_replica.{h,c}). KV row
 * bytes pinned to DS4_REPLICA_KV_ROW_BYTES (608), byte-exact compatible
 * with the host's g->layer_attn_comp_cache[il] layout (Lever A, ds4.c:121).
 *
 * Usage:
 *   ds4-kv-server listen <port> [bind_addr] [--mem-limit-gib N]
 *
 * Memory policy: a hard cap (default 3 GiB) on total bytes held; PUTs
 * past the cap are rejected with a BYE-error response (host falls back
 * to keeping the row local, no correctness regression).
 */

#include "ds4_replica.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t layer_idx;
    uint32_t row_start;
    uint32_t row_count;
    uint8_t *bytes;          /* malloc'd row_count * DS4_REPLICA_KV_ROW_BYTES */
} kv_chunk;

typedef struct {
    kv_chunk *items;
    size_t    n;
    size_t    cap;
    size_t    total_bytes;
    size_t    mem_limit_bytes;
    /* lightweight stats for the per-connection summary line. */
    uint64_t  put_count;
    uint64_t  put_bytes;
    uint64_t  put_rejected;
    uint64_t  pull_count;
    uint64_t  pull_bytes;
    uint64_t  pull_missed;
} kv_store;

static void kv_store_init(kv_store *s, size_t mem_limit_bytes) {
    memset(s, 0, sizeof(*s));
    s->mem_limit_bytes = mem_limit_bytes;
}

static void kv_store_free(kv_store *s) {
    for (size_t i = 0; i < s->n; i++) free(s->items[i].bytes);
    free(s->items);
    memset(s, 0, sizeof(*s));
}

/* Find a chunk that exactly matches (layer, row_start, row_count). Returns
 * index or -1. Linear scan; with ~43 layers * tens of chunks the table stays
 * well under 1000 entries — fine for demo. */
static ssize_t kv_store_find(const kv_store *s,
                             uint32_t layer_idx,
                             uint32_t row_start,
                             uint32_t row_count) {
    for (size_t i = 0; i < s->n; i++) {
        const kv_chunk *c = &s->items[i];
        if (c->layer_idx == layer_idx &&
            c->row_start == row_start &&
            c->row_count == row_count) {
            return (ssize_t)i;
        }
    }
    return -1;
}

/* Put: copies bytes into the store. If the same (layer, row_start, row_count)
 * already exists, the old buffer is replaced (overwrite). Returns 0 on
 * success, -1 if the put would exceed the memory cap (caller informs host). */
static int kv_store_put(kv_store *s,
                        uint32_t layer_idx,
                        uint32_t row_start,
                        uint32_t row_count,
                        const uint8_t *src,
                        size_t src_bytes) {
    if (src_bytes != (size_t)row_count * DS4_REPLICA_KV_ROW_BYTES) {
        errno = EINVAL;
        return -1;
    }

    ssize_t existing = kv_store_find(s, layer_idx, row_start, row_count);
    if (existing >= 0) {
        memcpy(s->items[existing].bytes, src, src_bytes);
        s->put_count++;
        s->put_bytes += src_bytes;
        return 0;
    }

    /* Cap check on the new allocation only — overwrites don't change total. */
    if (s->total_bytes + src_bytes > s->mem_limit_bytes) {
        s->put_rejected++;
        errno = ENOSPC;
        return -1;
    }

    if (s->n == s->cap) {
        size_t new_cap = s->cap ? s->cap * 2 : 64;
        kv_chunk *grown = realloc(s->items, new_cap * sizeof(kv_chunk));
        if (!grown) return -1;
        s->items = grown;
        s->cap = new_cap;
    }

    uint8_t *buf = malloc(src_bytes);
    if (!buf) return -1;
    memcpy(buf, src, src_bytes);

    s->items[s->n].layer_idx = layer_idx;
    s->items[s->n].row_start = row_start;
    s->items[s->n].row_count = row_count;
    s->items[s->n].bytes = buf;
    s->n++;
    s->total_bytes += src_bytes;
    s->put_count++;
    s->put_bytes += src_bytes;
    return 0;
}

/* Get-copy: writes bytes for the requested (layer, row_start, row_count) into
 * out. Returns 0 on hit, -1 with errno=ENOENT on miss. The host pulls when
 * a decode step needs an evicted row and the lossy-skip path is disabled. */
static int kv_store_get(kv_store *s,
                        uint32_t layer_idx,
                        uint32_t row_start,
                        uint32_t row_count,
                        uint8_t *out,
                        size_t out_bytes) {
    ssize_t idx = kv_store_find(s, layer_idx, row_start, row_count);
    if (idx < 0) {
        s->pull_missed++;
        errno = ENOENT;
        return -1;
    }
    size_t need = (size_t)row_count * DS4_REPLICA_KV_ROW_BYTES;
    if (out_bytes < need) { errno = EMSGSIZE; return -1; }
    memcpy(out, s->items[idx].bytes, need);
    s->pull_count++;
    s->pull_bytes += need;
    return 0;
}

/* Receive a KV_STREAM message and stash it. The wire header has already been
 * read by the caller (msg_type matched). */
static int handle_kv_stream(int fd, kv_store *s, const ds4_replica_wire_header *wh) {
    if (wh->payload_bytes < sizeof(ds4_replica_kv_stream_hdr)) {
        errno = EBADMSG; return -1;
    }
    ds4_replica_kv_stream_hdr hdr;
    if (!ds4_replica_recv_payload(fd, &hdr, sizeof(hdr), 30000)) return -1;
    if (hdr.row_bytes != DS4_REPLICA_KV_ROW_BYTES) { errno = EBADMSG; return -1; }
    size_t rows_bytes = (size_t)hdr.row_count * DS4_REPLICA_KV_ROW_BYTES;
    if (rows_bytes != wh->payload_bytes - sizeof(hdr)) { errno = EBADMSG; return -1; }

    /* Bounded staging buf — up to a single chunk at a time. */
    uint8_t *staging = malloc(rows_bytes);
    if (!staging) return -1;
    if (!ds4_replica_recv_payload(fd, staging, rows_bytes, 30000)) {
        free(staging); return -1;
    }
    int rc = kv_store_put(s, hdr.layer_idx, hdr.row_start, hdr.row_count,
                          staging, rows_bytes);
    free(staging);
    return rc;
}

/* Pull request: host sends KV_HANDOVER carrying a kv_stream_hdr (no row
 * payload). We reply with KV_STREAM if hit, or with a zero-payload
 * KV_HANDOVER_ACK if miss (host treats miss as "stay local / sink"). */
static int handle_kv_handover(int fd, kv_store *s, const ds4_replica_wire_header *wh) {
    if (wh->payload_bytes != sizeof(ds4_replica_kv_stream_hdr)) {
        errno = EBADMSG; return -1;
    }
    ds4_replica_kv_stream_hdr hdr;
    if (!ds4_replica_recv_payload(fd, &hdr, sizeof(hdr), 5000)) return -1;
    if (hdr.row_bytes != DS4_REPLICA_KV_ROW_BYTES) { errno = EBADMSG; return -1; }

    size_t bytes = (size_t)hdr.row_count * DS4_REPLICA_KV_ROW_BYTES;
    uint8_t *out = malloc(bytes);
    if (!out) return -1;
    int rc = kv_store_get(s, hdr.layer_idx, hdr.row_start, hdr.row_count,
                          out, bytes);
    if (rc < 0) {
        /* Miss — empty ACK signals "not held". */
        free(out);
        return ds4_replica_send_msg(fd, DS4_REPLICA_MSG_KV_HANDOVER_ACK, 0,
                                    wh->seq, NULL, 0) ? 0 : -1;
    }

    int ok = ds4_replica_kv_stream_send(fd, wh->seq, hdr.layer_idx,
                                        hdr.row_start, hdr.row_count,
                                        hdr.total_rows, out);
    free(out);
    return ok ? 0 : -1;
}

static int drain_payload(int fd, uint32_t bytes) {
    uint8_t scratch[4096];
    while (bytes) {
        size_t n = bytes < sizeof(scratch) ? bytes : sizeof(scratch);
        if (!ds4_replica_recv_payload(fd, scratch, n, 5000)) return -1;
        bytes -= (uint32_t)n;
    }
    return 0;
}

static int run_listen(uint16_t port, const char *bind_addr, size_t mem_limit_bytes) {
    int lfd = ds4_replica_listen(port, bind_addr);
    if (lfd < 0) { perror("listen"); return 1; }
    fprintf(stdout, "[ds4-kv-server] listening on %s:%u  mem_limit=%.2f GiB\n",
        bind_addr && bind_addr[0] ? bind_addr : "*", (unsigned)port,
        (double)mem_limit_bytes / (1024.0 * 1024.0 * 1024.0));

    /* Single-connection loop for now — host owns one ds4-server per replica. */
    for (;;) {
        int fd = ds4_replica_accept(lfd, -1);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept"); break;
        }
        fprintf(stdout, "[ds4-kv-server] accepted connection\n");

        kv_store store;
        kv_store_init(&store, mem_limit_bytes);

        ds4_replica_handshake self, peer;
        ds4_replica_self_describe(&self, DS4_REPLICA_ROLE_REPLICA);
        if (!ds4_replica_do_handshake(fd, &self, &peer, 10000)) {
            perror("handshake");
            ds4_replica_close(fd);
            kv_store_free(&store);
            continue;
        }
        fprintf(stdout,
            "[ds4-kv-server] handshake ok: peer host=%s ver=%s layers=%u row_bytes=%u\n",
            peer.hostname, peer.ds4_version, peer.n_layer, peer.kv_row_bytes);
        if (peer.kv_row_bytes != DS4_REPLICA_KV_ROW_BYTES) {
            fprintf(stderr,
                "[ds4-kv-server] WARN peer kv_row_bytes=%u != %u; aborting\n",
                peer.kv_row_bytes, DS4_REPLICA_KV_ROW_BYTES);
            ds4_replica_close(fd);
            kv_store_free(&store);
            continue;
        }

        for (;;) {
            ds4_replica_wire_header wh;
            if (!ds4_replica_recv_header(fd, &wh, -1)) {
                if (errno == ECONNRESET || errno == 0) break;
                perror("recv_header"); break;
            }
            int rc = 0;
            switch (wh.msg_type) {
                case DS4_REPLICA_MSG_KV_STREAM:
                    rc = handle_kv_stream(fd, &store, &wh);
                    if (rc < 0 && errno == ENOSPC) {
                        /* Out of budget — tell host so it stops pushing. */
                        ds4_replica_send_msg(fd, DS4_REPLICA_MSG_KV_HANDOVER_ACK,
                                             1u /* flag: rejected */, wh.seq,
                                             NULL, 0);
                        rc = 0;
                    }
                    break;
                case DS4_REPLICA_MSG_KV_HANDOVER:
                    rc = handle_kv_handover(fd, &store, &wh);
                    break;
                case DS4_REPLICA_MSG_PING:
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    if (rc == 0 &&
                        !ds4_replica_send_msg(fd, DS4_REPLICA_MSG_PONG, 0,
                                              wh.seq, NULL, 0)) rc = -1;
                    break;
                case DS4_REPLICA_MSG_BYE:
                    if (wh.payload_bytes) drain_payload(fd, wh.payload_bytes);
                    fprintf(stdout, "[ds4-kv-server] BYE\n");
                    goto done;
                default:
                    fprintf(stderr,
                        "[ds4-kv-server] unhandled msg_type=0x%x payload=%u (drained)\n",
                        wh.msg_type, wh.payload_bytes);
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    break;
            }
            if (rc < 0) { perror("handler"); break; }
        }
        done:
        fprintf(stdout,
            "[ds4-kv-server] session done: chunks=%zu held=%.2f MiB  "
            "puts=%" PRIu64 "/%.2f MiB rej=%" PRIu64 "  "
            "pulls=%" PRIu64 "/%.2f MiB miss=%" PRIu64 "\n",
            store.n,
            (double)store.total_bytes / (1024.0 * 1024.0),
            store.put_count,
            (double)store.put_bytes / (1024.0 * 1024.0),
            store.put_rejected,
            store.pull_count,
            (double)store.pull_bytes / (1024.0 * 1024.0),
            store.pull_missed);
        ds4_replica_close(fd);
        kv_store_free(&store);
    }

    ds4_replica_close(lfd);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "Usage: ds4-kv-server listen <port> [bind_addr] [--mem-limit-gib N]\n"
        "  default --mem-limit-gib 3   (caps in-memory KV byte store)\n");
}

int main(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[1], "listen") != 0) { usage(); return 2; }

    uint16_t port = (uint16_t)atoi(argv[2]);
    const char *bind_addr = NULL;
    double mem_limit_gib = 3.0;

    /* Optional positional bind_addr (must not start with '-'), then flags. */
    int i = 3;
    if (i < argc && argv[i][0] != '-') { bind_addr = argv[i++]; }
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--mem-limit-gib") && i + 1 < argc) {
            mem_limit_gib = atof(argv[++i]);
            if (mem_limit_gib <= 0) { usage(); return 2; }
        } else {
            usage(); return 2;
        }
    }

    size_t mem_limit_bytes = (size_t)(mem_limit_gib * 1024.0 * 1024.0 * 1024.0);
    return run_listen(port, bind_addr, mem_limit_bytes);
}
