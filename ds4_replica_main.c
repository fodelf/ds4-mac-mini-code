#include "ds4_replica.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Minimal driver for the dual-host wire protocol.  This binary is the
 * future ds4-replica process: it will eventually load the engine and run
 * prefill + KV streaming + cold expert serving for the host.  At M2' it
 * only exercises the wire protocol so we can validate framing, handshake,
 * and RTT across a real Thunderbolt link before wiring the engine.
 *
 * Usage:
 *   ds4-replica listen  <port> [bind_addr]
 *   ds4-replica connect <host> <port> [ping_count]
 */

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  ds4-replica listen  <port> [bind_addr]\n"
        "  ds4-replica connect <host> <port> [ping_count]\n");
}

static void print_handshake(const char *label, const ds4_replica_handshake *h) {
    fprintf(stdout,
        "[ds4-replica] %s peer: role=%u backend=%u host=%s ver=%s "
        "layer=%u expert/layer=%u used=%u row_bytes=%u "
        "ram_total=%" PRIu64 " free=%" PRIu64 "\n",
        label,
        h->role, h->backend, h->hostname, h->ds4_version,
        h->n_layer, h->n_expert_per_layer, h->n_expert_used, h->kv_row_bytes,
        h->total_ram_bytes, h->free_ram_bytes);
}

static int run_listen(uint16_t port, const char *bind_addr) {
    int lfd = ds4_replica_listen(port, bind_addr);
    if (lfd < 0) { perror("listen"); return 1; }
    fprintf(stdout, "[ds4-replica] listening on %s:%u\n",
        bind_addr && bind_addr[0] ? bind_addr : "*", (unsigned)port);

    int fd = ds4_replica_accept(lfd, -1);
    if (fd < 0) { perror("accept"); close(lfd); return 1; }
    fprintf(stdout, "[ds4-replica] accepted connection\n");

    ds4_replica_handshake self, peer;
    ds4_replica_self_describe(&self, DS4_REPLICA_ROLE_REPLICA);
    if (!ds4_replica_do_handshake(fd, &self, &peer, 5000)) {
        perror("handshake");
        ds4_replica_close(fd); close(lfd);
        return 1;
    }
    print_handshake("HANDSHAKE", &peer);

    /* Drain ping/pong + BYE in a small server loop. */
    for (;;) {
        ds4_replica_wire_header h;
        if (!ds4_replica_recv_header(fd, &h, -1)) {
            if (errno == ECONNRESET || errno == 0) break;
            perror("recv_header"); break;
        }
        if (h.payload_bytes) {
            /* Drain — none of the v1 messages handled here carry payload yet. */
            uint8_t scratch[4096];
            size_t left = h.payload_bytes;
            while (left) {
                size_t n = left < sizeof(scratch) ? left : sizeof(scratch);
                if (!ds4_replica_recv_payload(fd, scratch, n, 5000)) break;
                left -= n;
            }
            if (left) { perror("drain"); break; }
        }
        if (h.msg_type == DS4_REPLICA_MSG_PING) {
            if (!ds4_replica_send_msg(fd, DS4_REPLICA_MSG_PONG, 0, h.seq, NULL, 0)) {
                perror("pong"); break;
            }
        } else if (h.msg_type == DS4_REPLICA_MSG_BYE) {
            fprintf(stdout, "[ds4-replica] BYE received, closing\n");
            break;
        } else {
            fprintf(stderr, "[ds4-replica] unhandled msg_type=0x%x (skipped)\n",
                h.msg_type);
        }
    }

    ds4_replica_close(fd);
    ds4_replica_close(lfd);
    return 0;
}

static int run_connect(const char *host, uint16_t port, int ping_count) {
    int fd = ds4_replica_connect(host, port, 5000);
    if (fd < 0) { perror("connect"); return 1; }
    fprintf(stdout, "[ds4-replica] connected to %s:%u\n", host, (unsigned)port);

    ds4_replica_handshake self, peer;
    ds4_replica_self_describe(&self, DS4_REPLICA_ROLE_HOST);
    if (!ds4_replica_do_handshake(fd, &self, &peer, 5000)) {
        perror("handshake"); ds4_replica_close(fd); return 1;
    }
    print_handshake("HANDSHAKE_ACK", &peer);

    int64_t total = 0, mn = INT64_MAX, mx = 0;
    int ok = 0;
    for (int i = 0; i < ping_count; i++) {
        int64_t rtt = ds4_replica_ping_rtt_us(fd, 5000);
        if (rtt < 0) { perror("ping"); break; }
        fprintf(stdout, "[ds4-replica] ping[%d] rtt=%" PRId64 " us\n", i, rtt);
        total += rtt;
        if (rtt < mn) mn = rtt;
        if (rtt > mx) mx = rtt;
        ok++;
    }
    if (ok > 0) {
        fprintf(stdout, "[ds4-replica] rtt avg=%.1f us min=%" PRId64 " max=%" PRId64 " (n=%d)\n",
            (double)total / ok, mn, mx, ok);
    }

    ds4_replica_send_msg(fd, DS4_REPLICA_MSG_BYE, 0, 0, NULL, 0);
    ds4_replica_close(fd);
    return ok == ping_count ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(); return 2; }

    if (strcmp(argv[1], "listen") == 0) {
        uint16_t port = (uint16_t)atoi(argv[2]);
        const char *bind_addr = (argc >= 4) ? argv[3] : NULL;
        return run_listen(port, bind_addr);
    }
    if (strcmp(argv[1], "connect") == 0) {
        if (argc < 4) { usage(); return 2; }
        uint16_t port = (uint16_t)atoi(argv[3]);
        int ping_count = (argc >= 5) ? atoi(argv[4]) : 8;
        if (ping_count < 1) ping_count = 1;
        return run_connect(argv[2], port, ping_count);
    }
    usage();
    return 2;
}
