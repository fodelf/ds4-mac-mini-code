/* ds4-expert-replica — off-host routed-expert cold tier (path B, no MTP).
 *
 * The host (ds4 / ds4-server) pins the dense weights in RAM locally
 * (DS4_DENSE_RESIDENT) and keeps a small routed-expert LRU cache.  On a decode
 * layer whose active experts miss that local cache, the host asks THIS process
 * for those experts' raw bytes in one batched round-trip over Thunderbolt
 * (~2.5 GB/s) instead of re-faulting them from its own SSD (~0.4 GB/s, the
 * remaining decode bottleneck after dense residency).
 *
 * This process loads ONLY the base GGUF, with backend=CPU so nothing but the
 * model is mapped (no Metal graph, no KV, no inference — serving is pure CPU
 * memcpy from the mmap).  The 82 GiB base is mmap-backed and lazy: only the
 * experts actually requested fault into RAM, and the OS page cache naturally
 * keeps the hot subset resident within the machine's carve (no explicit pool,
 * no anon allocation → file-backed, evictable, swap-safe).  On an 8 GiB carve
 * the page cache holds ~1100 experts (~25/layer); the host's local cache + this
 * tier together cover the hot routing set, leaving only rare cold experts to
 * this machine's SSD.
 *
 * Wire protocol: EXPERT_REQ/EXPERT_RESP in ds4_replica.{h,c}.  Single host
 * connection at a time (one decode timeline).
 *
 * Usage:
 *   ds4-expert-replica listen <port> [bind_addr] -m <base.gguf>
 *
 * MEMORY SAFETY: this never wires GPU buffers and never allocates the model in
 * anon RAM; resident cost = page cache for served experts, bounded by the OS to
 * available RAM.  The deploy script adds an RSS watchdog as a backstop.
 */

#include "ds4.h"
#include "ds4_replica.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Host sends at most one decode layer's active-expert misses per request
 * (n_expert_used = 6 for DSv4); 32 gives ample headroom for any batching. */
#define EXPERT_REPLICA_MAX_IDS 32

static int drain_payload(int fd, uint32_t n) {
    uint8_t scratch[512];
    uint32_t left = n;
    while (left) {
        uint32_t want = left < sizeof(scratch) ? left : (uint32_t)sizeof(scratch);
        if (!ds4_replica_recv_payload(fd, scratch, want, -1)) return -1;
        left -= want;
    }
    return 0;
}

static int run_listen(uint16_t port, const char *bind_addr, ds4_engine *engine) {
    uint64_t model_size = 0;
    const uint8_t *map = (const uint8_t *)ds4_engine_model_map_ptr(engine, &model_size);
    uint32_t n_layer = 0, n_expert = 0;
    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!map ||
        ds4_engine_expert_layout(engine, &n_layer, &n_expert, &gate_bytes, &down_bytes) != 0 ||
        gate_bytes == 0 || down_bytes == 0) {
        fprintf(stderr, "[ds4-expert-replica] could not read expert layout from engine\n");
        return 1;
    }
    const uint64_t per_expert = 2u * gate_bytes + down_bytes;  /* gate || up || down */
    uint8_t *resp_buf = malloc((size_t)EXPERT_REPLICA_MAX_IDS * per_expert);
    if (!resp_buf) { perror("malloc resp_buf"); return 1; }

    int lfd = ds4_replica_listen(port, bind_addr);
    if (lfd < 0) { perror("listen"); free(resp_buf); return 1; }
    fprintf(stdout,
            "[ds4-expert-replica] listening on %s:%u  n_layer=%u n_expert=%u "
            "gate_bytes=%" PRIu64 " down_bytes=%" PRIu64 " (per-expert %.2f MiB)\n",
            bind_addr && bind_addr[0] ? bind_addr : "*", (unsigned)port,
            n_layer, n_expert, gate_bytes, down_bytes, (double)per_expert / 1048576.0);
    fflush(stdout);

    for (;;) {
        int fd = ds4_replica_accept(lfd, -1);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept"); break;
        }
        fprintf(stdout, "[ds4-expert-replica] accepted connection\n");
        fflush(stdout);

        ds4_replica_handshake self, peer;
        ds4_replica_self_describe(&self, DS4_REPLICA_ROLE_REPLICA);
        if (!ds4_replica_do_handshake(fd, &self, &peer, 10000)) {
            perror("handshake");
            ds4_replica_close(fd);
            continue;
        }
        fprintf(stdout, "[ds4-expert-replica] handshake ok: peer host=%s ver=%s\n",
                peer.hostname, peer.ds4_version);
        fflush(stdout);

        uint64_t reqs = 0, experts_served = 0;
        for (;;) {
            ds4_replica_wire_header wh;
            if (!ds4_replica_recv_header(fd, &wh, -1)) {
                if (errno == ECONNRESET || errno == 0) break;
                perror("recv_header"); break;
            }
            int rc = 0;
            switch (wh.msg_type) {
                case DS4_REPLICA_MSG_EXPERT_REQ: {
                    ds4_replica_expert_req_hdr rh;
                    if (wh.payload_bytes < sizeof(rh)) { rc = -1; errno = EBADMSG; break; }
                    if (!ds4_replica_recv_payload(fd, &rh, sizeof(rh), -1)) { rc = -1; break; }
                    size_t id_bytes = (size_t)rh.n_ids * sizeof(int32_t);
                    if (id_bytes != wh.payload_bytes - sizeof(rh) ||
                        rh.n_ids > EXPERT_REPLICA_MAX_IDS) { rc = -1; errno = EBADMSG; break; }
                    int32_t ids[EXPERT_REPLICA_MAX_IDS];
                    if (id_bytes && !ds4_replica_recv_payload(fd, ids, id_bytes, -1)) { rc = -1; break; }

                    /* Validate the host's expert byte sizes match ours (same GGUF). */
                    if (rh.gate_expert_bytes != (uint32_t)gate_bytes ||
                        rh.down_expert_bytes != (uint32_t)down_bytes) {
                        fprintf(stderr, "[ds4-expert-replica] expert size mismatch "
                                "(host gate=%u down=%u, ours %" PRIu64 "/%" PRIu64 ")\n",
                                rh.gate_expert_bytes, rh.down_expert_bytes, gate_bytes, down_bytes);
                        if (!ds4_replica_send_expert_resp(fd, wh.seq, 1u, 0,
                                (uint32_t)gate_bytes, (uint32_t)down_bytes, NULL, 0)) rc = -1;
                        break;
                    }
                    uint64_t gate_off = 0, up_off = 0, down_off = 0;
                    if (ds4_engine_expert_offsets(engine, rh.layer_idx,
                                                  &gate_off, &up_off, &down_off) != 0) {
                        if (!ds4_replica_send_expert_resp(fd, wh.seq, 2u, 0,
                                (uint32_t)gate_bytes, (uint32_t)down_bytes, NULL, 0)) rc = -1;
                        break;
                    }
                    /* Gather gate||up||down for each id from the mmap (page cache
                     * serves the hot ones; cold ones fault from this SSD once). */
                    uint8_t *cur = resp_buf;
                    int oob = 0;
                    for (uint32_t k = 0; k < rh.n_ids; k++) {
                        uint32_t id = (uint32_t)ids[k];
                        if (id >= n_expert) id = 0;
                        const uint64_t g = gate_off + (uint64_t)id * gate_bytes;
                        const uint64_t u = up_off   + (uint64_t)id * gate_bytes;
                        const uint64_t d = down_off + (uint64_t)id * down_bytes;
                        if (g + gate_bytes > model_size || u + gate_bytes > model_size ||
                            d + down_bytes > model_size) { oob = 1; break; }
                        memcpy(cur, map + g, (size_t)gate_bytes); cur += gate_bytes;
                        memcpy(cur, map + u, (size_t)gate_bytes); cur += gate_bytes;
                        memcpy(cur, map + d, (size_t)down_bytes); cur += down_bytes;
                    }
                    if (oob) {
                        if (!ds4_replica_send_expert_resp(fd, wh.seq, 3u, 0,
                                (uint32_t)gate_bytes, (uint32_t)down_bytes, NULL, 0)) rc = -1;
                        break;
                    }
                    size_t total = (size_t)rh.n_ids * per_expert;
                    if (!ds4_replica_send_expert_resp(fd, wh.seq, 0u, rh.n_ids,
                            (uint32_t)gate_bytes, (uint32_t)down_bytes, resp_buf, total)) { rc = -1; break; }
                    reqs++;
                    experts_served += rh.n_ids;
                    break;
                }
                case DS4_REPLICA_MSG_PING:
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    if (rc == 0 &&
                        !ds4_replica_send_msg(fd, DS4_REPLICA_MSG_PONG, 0, wh.seq, NULL, 0)) rc = -1;
                    break;
                case DS4_REPLICA_MSG_BYE:
                    if (wh.payload_bytes) drain_payload(fd, wh.payload_bytes);
                    fprintf(stdout, "[ds4-expert-replica] BYE\n");
                    goto done;
                default:
                    fprintf(stderr, "[ds4-expert-replica] unhandled msg_type=0x%x (drained)\n",
                            wh.msg_type);
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    break;
            }
            if (rc < 0) { perror("handler"); break; }
        }
        done:
        fprintf(stdout,
                "[ds4-expert-replica] session done: reqs=%" PRIu64 " experts_served=%" PRIu64 "\n",
                reqs, experts_served);
        fflush(stdout);
        ds4_replica_close(fd);
    }

    ds4_replica_close(lfd);
    free(resp_buf);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "Usage: ds4-expert-replica listen <port> [bind_addr] -m <base.gguf>\n"
        "  Loads the base GGUF (CPU backend, mmap only) and serves routed-expert\n"
        "  bytes to a host over Thunderbolt. The OS page cache holds the hot set.\n");
}

int main(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[1], "listen") != 0) { usage(); return 2; }

    uint16_t port = (uint16_t)atoi(argv[2]);
    const char *bind_addr = NULL;
    const char *model_path = NULL;

    int i = 3;
    if (i < argc && argv[i][0] != '-') { bind_addr = argv[i++]; }
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            usage(); return 2;
        }
    }
    if (!model_path) {
        fprintf(stderr, "[ds4-expert-replica] -m <base.gguf> is required\n");
        usage();
        return 2;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CPU;  /* model mapped only; no graph, no inference */
    opt.expert_server_mode = true;  /* lazy mmap: NO full-model WILLNEED prefetch
                                     * (that prefetch OOMs the host — see ds4.h) */

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "[ds4-expert-replica] failed to open engine\n");
        return 1;
    }

    int rc = run_listen(port, bind_addr, engine);

    ds4_engine_close(engine);
    return rc;
}
