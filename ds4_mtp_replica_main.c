/* ds4-mtp-replica — off-host MTP speculative drafter.
 *
 * The host (ds4-server / ds4) runs the DeepSeek V4 Flash target model and keeps
 * the decode hot path local.  This binary owns the MTP drafter on a second
 * machine: it loads the base GGUF (mmap; only token_embd + the output head are
 * ever faulted — the 43 routed-expert layers are never bound to the GPU, so the
 * 86 GiB model costs ~1.6 GiB resident here) plus the MTP support GGUF
 * (~3.5 GiB).  Total resident ~5 GiB, fitting the MacBook's ~8 GiB carve.
 *
 * Per decode cycle the host ships its just-committed cur_hc (n_hc*n_embd f32)
 * with the committed token + position; this process reseeds the MTP drafter and
 * runs draft_cap recursive steps, returning the proposed token ids.  The host
 * verifies them against the target model (one batched forward) and accepts the
 * longest correct prefix — the speculative speedup, with the drafter's compute
 * and memory moved off the host.  The MTP SWA raw cache persists across bursts;
 * prev_accepted in each request rolls it back to the accepted frontier.
 *
 * Wire protocol: shared with ds4-replica (ds4_replica.{h,c}).  Single host
 * connection at a time (the host runs one decode timeline).
 *
 * Usage:
 *   ds4-mtp-replica listen <port> [bind_addr] -m <base.gguf> --mtp <mtp.gguf>
 *                   [-c ctx_size]
 *
 * RESIDENCY RISK (validate on hardware): this relies on the view-shrink path
 * (DS4_METAL_EXPERT_OFFLOAD + DS4_METAL_MODEL_MAX_VIEW_BYTES) so IOGPU wires only the
 * embd + output views, not the whole base buffer.  If IOGPU over-wires, the
 * base map will not fit 8 GiB — see notes/execution-log.md M3a entry.
 */

#include "ds4.h"
#include "ds4_replica.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MTP_REPLICA_MAX_DRAFT 16

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

static int run_listen(uint16_t port, const char *bind_addr,
                      ds4_engine *engine, ds4_session *session) {
    const int hc_floats = ds4_engine_mtp_hc_floats(engine);
    float *hc_buf = malloc((size_t)hc_floats * sizeof(float));
    if (!hc_buf) { perror("malloc hc"); return 1; }

    int lfd = ds4_replica_listen(port, bind_addr);
    if (lfd < 0) { perror("listen"); free(hc_buf); return 1; }
    fprintf(stdout, "[ds4-mtp-replica] listening on %s:%u  hc_floats=%d\n",
            bind_addr && bind_addr[0] ? bind_addr : "*", (unsigned)port, hc_floats);
    fflush(stdout);

    for (;;) {
        int fd = ds4_replica_accept(lfd, -1);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept"); break;
        }
        fprintf(stdout, "[ds4-mtp-replica] accepted connection\n");
        fflush(stdout);

        ds4_replica_handshake self, peer;
        ds4_replica_self_describe(&self, DS4_REPLICA_ROLE_MTP_REPLICA);
        if (!ds4_replica_do_handshake(fd, &self, &peer, 10000)) {
            perror("handshake");
            ds4_replica_close(fd);
            continue;
        }
        fprintf(stdout,
                "[ds4-mtp-replica] handshake ok: peer host=%s ver=%s\n",
                peer.hostname, peer.ds4_version);
        fflush(stdout);

        /* Each new connection is a fresh decode timeline — clear MTP history. */
        ds4_engine_mtp_draft_reset(session);

        uint64_t bursts = 0, drafts_total = 0;
        for (;;) {
            ds4_replica_wire_header wh;
            if (!ds4_replica_recv_header(fd, &wh, -1)) {
                if (errno == ECONNRESET || errno == 0) break;
                perror("recv_header"); break;
            }
            int rc = 0;
            switch (wh.msg_type) {
                case DS4_REPLICA_MSG_MTP_DRAFT_REQ: {
                    /* The header was already consumed by recv_header; we need the
                     * full message, so reconstruct via the typed recv on the
                     * remaining payload.  recv_mtp_req expects to read the wire
                     * header itself, so instead drain+parse inline here. */
                    ds4_replica_mtp_req_hdr rh;
                    if (wh.payload_bytes < sizeof(rh)) { rc = -1; errno = EBADMSG; break; }
                    if (!ds4_replica_recv_payload(fd, &rh, sizeof(rh), -1)) { rc = -1; break; }
                    size_t hc_bytes = (size_t)rh.hc_floats * sizeof(float);
                    if (hc_bytes != wh.payload_bytes - sizeof(rh) ||
                        rh.hc_floats != (uint32_t)hc_floats) {
                        rc = -1; errno = EBADMSG; break;
                    }
                    if (hc_bytes && !ds4_replica_recv_payload(fd, hc_buf, hc_bytes, -1)) { rc = -1; break; }

                    int drafts[MTP_REPLICA_MAX_DRAFT];
                    char err[160] = {0};
                    int cap = (int)rh.draft_cap;
                    if (cap > MTP_REPLICA_MAX_DRAFT) cap = MTP_REPLICA_MAX_DRAFT;
                    int n = ds4_engine_mtp_draft_burst(session,
                                                       hc_buf, hc_floats,
                                                       rh.seed_token, rh.seed_pos,
                                                       cap, (int)rh.prev_accepted,
                                                       (int)rh.eos_token,
                                                       drafts, MTP_REPLICA_MAX_DRAFT,
                                                       err, sizeof(err));
                    if (n < 0) {
                        fprintf(stderr, "[ds4-mtp-replica] draft failed: %s\n", err);
                        if (!ds4_replica_send_mtp_resp(fd, wh.seq, 1u, 0, NULL)) rc = -1;
                        break;
                    }
                    int32_t out[MTP_REPLICA_MAX_DRAFT];
                    for (int k = 0; k < n; k++) out[k] = (int32_t)drafts[k];
                    if (!ds4_replica_send_mtp_resp(fd, wh.seq, 0u, (uint32_t)n, out)) { rc = -1; break; }
                    bursts++;
                    drafts_total += (uint64_t)n;
                    break;
                }
                case DS4_REPLICA_MSG_MTP_INVALIDATE:
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    ds4_engine_mtp_draft_reset(session);
                    break;
                case DS4_REPLICA_MSG_PING:
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    if (rc == 0 &&
                        !ds4_replica_send_msg(fd, DS4_REPLICA_MSG_PONG, 0, wh.seq, NULL, 0)) rc = -1;
                    break;
                case DS4_REPLICA_MSG_BYE:
                    if (wh.payload_bytes) drain_payload(fd, wh.payload_bytes);
                    fprintf(stdout, "[ds4-mtp-replica] BYE\n");
                    goto done;
                default:
                    fprintf(stderr, "[ds4-mtp-replica] unhandled msg_type=0x%x (drained)\n",
                            wh.msg_type);
                    if (wh.payload_bytes) rc = drain_payload(fd, wh.payload_bytes);
                    break;
            }
            if (rc < 0) { perror("handler"); break; }
        }
        done:
        fprintf(stdout,
                "[ds4-mtp-replica] session done: bursts=%" PRIu64 " drafts=%" PRIu64 "\n",
                bursts, drafts_total);
        fflush(stdout);
        ds4_replica_close(fd);
    }

    ds4_replica_close(lfd);
    free(hc_buf);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "Usage: ds4-mtp-replica listen <port> [bind_addr] -m <base.gguf> --mtp <mtp.gguf> [-c ctx]\n"
        "  Loads the base + MTP GGUFs and serves off-host MTP draft bursts.\n");
}

int main(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[1], "listen") != 0) { usage(); return 2; }

    uint16_t port = (uint16_t)atoi(argv[2]);
    const char *bind_addr = NULL;
    const char *model_path = NULL;
    const char *mtp_path = NULL;
    int ctx_size = 8192;

    int i = 3;
    if (i < argc && argv[i][0] != '-') { bind_addr = argv[i++]; }
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--mtp") && i + 1 < argc) {
            mtp_path = argv[++i];
        } else if (!strcmp(argv[i], "-c") && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
            if (ctx_size <= 0) { usage(); return 2; }
        } else {
            usage(); return 2;
        }
    }
    if (!model_path || !mtp_path) {
        fprintf(stderr, "[ds4-mtp-replica] -m <base.gguf> and --mtp <mtp.gguf> are required\n");
        usage();
        return 2;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.mtp_path = mtp_path;
#ifdef __APPLE__
    opt.backend = DS4_BACKEND_METAL;
#else
    opt.backend = DS4_BACKEND_CUDA;
#endif
    opt.mtp_draft_tokens = MTP_REPLICA_MAX_DRAFT;
    opt.mtp_replica_mode = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "[ds4-mtp-replica] failed to open engine\n");
        return 1;
    }
    if (!ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "[ds4-mtp-replica] MTP not ready after load — wrong --mtp file?\n");
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "[ds4-mtp-replica] failed to create session\n");
        ds4_engine_close(engine);
        return 1;
    }

    int rc = run_listen(port, bind_addr, engine, session);

    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
