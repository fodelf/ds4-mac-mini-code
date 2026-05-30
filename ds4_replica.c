#include "ds4_replica.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Verify pack(1) didn't drift on this compiler. */
_Static_assert(sizeof(ds4_replica_wire_header)    == 32,  "wire header must be 32 bytes");
_Static_assert(sizeof(ds4_replica_kv_stream_hdr)  == 24,  "kv stream hdr must be 24 bytes");
_Static_assert(sizeof(ds4_replica_handshake)      == 136, "handshake struct must be 136 bytes");
_Static_assert(sizeof(ds4_replica_mtp_req_hdr)    == 24,  "mtp req hdr must be 24 bytes");
_Static_assert(sizeof(ds4_replica_mtp_resp_hdr)   == 16,  "mtp resp hdr must be 16 bytes");

static int64_t now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
}

static int set_blocking(int fd, bool blocking) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (blocking) flags &= ~O_NONBLOCK; else flags |= O_NONBLOCK;
    return fcntl(fd, F_SETFL, flags);
}

void ds4_replica_tune_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    /* Big enough to hold a multi-MiB KV chunk without head-of-line stalls. */
    int buf = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

int ds4_replica_listen(uint16_t port, const char *bind_addr) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (bind_addr && bind_addr[0]) {
        if (inet_pton(AF_INET, bind_addr, &sa.sin_addr) != 1) {
            close(fd); errno = EINVAL; return -1;
        }
    } else {
        sa.sin_addr.s_addr = htonl(INADDR_ANY);
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        int e = errno; close(fd); errno = e; return -1;
    }
    if (listen(fd, 4) < 0) {
        int e = errno; close(fd); errno = e; return -1;
    }
    return fd;
}

int ds4_replica_accept(int listen_fd, int timeout_ms) {
    if (timeout_ms >= 0) {
        struct pollfd p = { .fd = listen_fd, .events = POLLIN };
        int r = poll(&p, 1, timeout_ms);
        if (r <= 0) { if (r == 0) errno = ETIMEDOUT; return -1; }
    }
    struct sockaddr_in peer;
    socklen_t plen = sizeof(peer);
    int fd = accept(listen_fd, (struct sockaddr *)&peer, &plen);
    if (fd < 0) return -1;
    ds4_replica_tune_socket(fd);
    return fd;
}

int ds4_replica_connect(const char *host, uint16_t port, int timeout_ms) {
    if (!host || !host[0]) { errno = EINVAL; return -1; }

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_s[16];
    snprintf(port_s, sizeof(port_s), "%u", (unsigned)port);
    if (getaddrinfo(host, port_s, &hints, &res) != 0 || !res) {
        errno = EHOSTUNREACH; return -1;
    }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    set_blocking(fd, false);
    int rc = connect(fd, res->ai_addr, res->ai_addrlen);
    int err = errno;
    if (rc < 0 && err != EINPROGRESS) {
        close(fd); freeaddrinfo(res); errno = err; return -1;
    }
    if (rc < 0) {
        struct pollfd p = { .fd = fd, .events = POLLOUT };
        int r = poll(&p, 1, timeout_ms < 0 ? -1 : timeout_ms);
        if (r <= 0) { close(fd); freeaddrinfo(res); errno = (r == 0) ? ETIMEDOUT : errno; return -1; }
        int so_err = 0; socklen_t slen = sizeof(so_err);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &slen) < 0 || so_err != 0) {
            close(fd); freeaddrinfo(res); errno = so_err ? so_err : EIO; return -1;
        }
    }
    freeaddrinfo(res);
    set_blocking(fd, true);
    ds4_replica_tune_socket(fd);
    return fd;
}

void ds4_replica_close(int fd) {
    if (fd >= 0) close(fd);
}

static bool send_all(int fd, const void *p, size_t n) {
    const uint8_t *s = (const uint8_t *)p;
    while (n) {
        ssize_t w = send(fd, s, n, 0);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        s += w; n -= (size_t)w;
    }
    return true;
}

static bool recv_all(int fd, void *p, size_t n, int timeout_ms) {
    uint8_t *s = (uint8_t *)p;
    int64_t deadline = (timeout_ms >= 0) ? (now_us() + (int64_t)timeout_ms * 1000LL) : -1;
    while (n) {
        if (deadline >= 0) {
            int64_t left_us = deadline - now_us();
            if (left_us <= 0) { errno = ETIMEDOUT; return false; }
            struct pollfd pfd = { .fd = fd, .events = POLLIN };
            int pr = poll(&pfd, 1, (int)(left_us / 1000));
            if (pr <= 0) { if (pr == 0) errno = ETIMEDOUT; return false; }
        }
        ssize_t r = recv(fd, s, n, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) { errno = ECONNRESET; return false; }
        s += r; n -= (size_t)r;
    }
    return true;
}

bool ds4_replica_send_msg(int fd,
                          ds4_replica_msg_type type,
                          uint32_t flags,
                          uint64_t seq,
                          const void *payload,
                          size_t payload_bytes) {
    if (payload_bytes > 0xFFFFFFFFu) { errno = EMSGSIZE; return false; }
    ds4_replica_wire_header h;
    memset(&h, 0, sizeof(h));
    h.magic = DS4_REPLICA_PROTO_MAGIC;
    h.version = (uint16_t)DS4_REPLICA_PROTO_VERSION;
    h.msg_type = (uint16_t)type;
    h.payload_bytes = (uint32_t)payload_bytes;
    h.flags = flags;
    h.seq = seq;
    if (!send_all(fd, &h, sizeof(h))) return false;
    if (payload_bytes && !send_all(fd, payload, payload_bytes)) return false;
    return true;
}

bool ds4_replica_recv_header(int fd,
                             ds4_replica_wire_header *out_hdr,
                             int timeout_ms) {
    if (!recv_all(fd, out_hdr, sizeof(*out_hdr), timeout_ms)) return false;
    if (out_hdr->magic != DS4_REPLICA_PROTO_MAGIC) { errno = EBADMSG; return false; }
    if (out_hdr->version != DS4_REPLICA_PROTO_VERSION) { errno = ENOTSUP; return false; }
    return true;
}

bool ds4_replica_recv_payload(int fd, void *buf, size_t len, int timeout_ms) {
    if (len == 0) return true;
    return recv_all(fd, buf, len, timeout_ms);
}

bool ds4_replica_do_handshake(int fd,
                              const ds4_replica_handshake *self,
                              ds4_replica_handshake *peer_out,
                              int timeout_ms) {
    /* Symmetric 4-message handshake: each side sends HANDSHAKE, receives
     * peer's HANDSHAKE, sends ACK, receives peer's ACK.  Total 4 messages
     * on the wire (2 per side).  Done this way so neither side leaves a
     * stray ACK queued in the recv buffer for the next caller. */
    if (!ds4_replica_send_msg(fd, DS4_REPLICA_MSG_HANDSHAKE, 0, 0,
                              self, sizeof(*self))) return false;

    ds4_replica_wire_header h;
    if (!ds4_replica_recv_header(fd, &h, timeout_ms)) return false;
    if (h.msg_type != DS4_REPLICA_MSG_HANDSHAKE) { errno = EPROTO; return false; }
    if (h.payload_bytes != sizeof(*peer_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, peer_out, sizeof(*peer_out), timeout_ms)) return false;

    if (!ds4_replica_send_msg(fd, DS4_REPLICA_MSG_HANDSHAKE_ACK, 0, 1, NULL, 0)) return false;

    if (!ds4_replica_recv_header(fd, &h, timeout_ms)) return false;
    if (h.msg_type != DS4_REPLICA_MSG_HANDSHAKE_ACK) { errno = EPROTO; return false; }
    if (h.payload_bytes) {
        /* No payload expected in v1; drain defensively so we don't strand bytes. */
        uint8_t scratch[256];
        size_t left = h.payload_bytes;
        while (left) {
            size_t n = left < sizeof(scratch) ? left : sizeof(scratch);
            if (!ds4_replica_recv_payload(fd, scratch, n, timeout_ms)) return false;
            left -= n;
        }
    }
    return true;
}

bool ds4_replica_kv_stream_send(int fd,
                                uint64_t seq,
                                uint32_t layer_idx,
                                uint32_t row_start,
                                uint32_t row_count,
                                uint32_t total_rows,
                                const void *rows) {
    ds4_replica_kv_stream_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.layer_idx = layer_idx;
    hdr.row_start = row_start;
    hdr.row_count = row_count;
    hdr.row_bytes = DS4_REPLICA_KV_ROW_BYTES;
    hdr.total_rows = total_rows;

    size_t row_bytes_total = (size_t)row_count * DS4_REPLICA_KV_ROW_BYTES;
    size_t payload_bytes = sizeof(hdr) + row_bytes_total;

    ds4_replica_wire_header wh;
    memset(&wh, 0, sizeof(wh));
    wh.magic = DS4_REPLICA_PROTO_MAGIC;
    wh.version = DS4_REPLICA_PROTO_VERSION;
    wh.msg_type = DS4_REPLICA_MSG_KV_STREAM;
    wh.payload_bytes = (uint32_t)payload_bytes;
    wh.flags = 0;
    wh.seq = seq;

    if (!send_all(fd, &wh, sizeof(wh))) return false;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (row_bytes_total && !send_all(fd, rows, row_bytes_total)) return false;
    return true;
}

bool ds4_replica_kv_stream_recv(int fd,
                                ds4_replica_kv_stream_hdr *hdr_out,
                                void *rows_buf,
                                size_t rows_buf_bytes,
                                int timeout_ms) {
    ds4_replica_wire_header wh;
    if (!ds4_replica_recv_header(fd, &wh, timeout_ms)) return false;
    if (wh.msg_type != DS4_REPLICA_MSG_KV_STREAM) { errno = EPROTO; return false; }
    if (wh.payload_bytes < sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, hdr_out, sizeof(*hdr_out), timeout_ms)) return false;
    if (hdr_out->row_bytes != DS4_REPLICA_KV_ROW_BYTES) { errno = EBADMSG; return false; }
    size_t row_bytes_total = (size_t)hdr_out->row_count * DS4_REPLICA_KV_ROW_BYTES;
    if (row_bytes_total != wh.payload_bytes - sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (row_bytes_total > rows_buf_bytes) { errno = EMSGSIZE; return false; }
    if (!ds4_replica_recv_payload(fd, rows_buf, row_bytes_total, timeout_ms)) return false;
    return true;
}

bool ds4_replica_send_mtp_req(int fd,
                              uint64_t seq,
                              int32_t seed_token,
                              uint32_t seed_pos,
                              uint32_t hc_floats,
                              uint32_t draft_cap,
                              uint32_t prev_accepted,
                              uint32_t eos_token,
                              const float *hc) {
    ds4_replica_mtp_req_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.seed_token = seed_token;
    hdr.seed_pos = seed_pos;
    hdr.hc_floats = hc_floats;
    hdr.draft_cap = draft_cap;
    hdr.prev_accepted = prev_accepted;
    hdr.eos_token = eos_token;

    size_t hc_bytes = (size_t)hc_floats * sizeof(float);
    size_t payload_bytes = sizeof(hdr) + hc_bytes;

    ds4_replica_wire_header wh;
    memset(&wh, 0, sizeof(wh));
    wh.magic = DS4_REPLICA_PROTO_MAGIC;
    wh.version = DS4_REPLICA_PROTO_VERSION;
    wh.msg_type = DS4_REPLICA_MSG_MTP_DRAFT_REQ;
    wh.payload_bytes = (uint32_t)payload_bytes;
    wh.seq = seq;

    if (!send_all(fd, &wh, sizeof(wh))) return false;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (hc_bytes && !send_all(fd, hc, hc_bytes)) return false;
    return true;
}

bool ds4_replica_recv_mtp_req(int fd,
                              ds4_replica_mtp_req_hdr *hdr_out,
                              float *hc_buf,
                              uint32_t hc_buf_floats,
                              int timeout_ms) {
    ds4_replica_wire_header wh;
    if (!ds4_replica_recv_header(fd, &wh, timeout_ms)) return false;
    if (wh.msg_type != DS4_REPLICA_MSG_MTP_DRAFT_REQ) { errno = EPROTO; return false; }
    if (wh.payload_bytes < sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, hdr_out, sizeof(*hdr_out), timeout_ms)) return false;
    size_t hc_bytes = (size_t)hdr_out->hc_floats * sizeof(float);
    if (hc_bytes != wh.payload_bytes - sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (hdr_out->hc_floats > hc_buf_floats) { errno = EMSGSIZE; return false; }
    if (hc_bytes && !ds4_replica_recv_payload(fd, hc_buf, hc_bytes, timeout_ms)) return false;
    return true;
}

bool ds4_replica_send_mtp_resp(int fd,
                               uint64_t seq,
                               uint32_t status,
                               uint32_t n_draft,
                               const int32_t *drafts) {
    ds4_replica_mtp_resp_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.status = status;
    hdr.n_draft = n_draft;

    size_t draft_bytes = (size_t)n_draft * sizeof(int32_t);
    size_t payload_bytes = sizeof(hdr) + draft_bytes;

    ds4_replica_wire_header wh;
    memset(&wh, 0, sizeof(wh));
    wh.magic = DS4_REPLICA_PROTO_MAGIC;
    wh.version = DS4_REPLICA_PROTO_VERSION;
    wh.msg_type = DS4_REPLICA_MSG_MTP_DRAFT_RESP;
    wh.payload_bytes = (uint32_t)payload_bytes;
    wh.seq = seq;

    if (!send_all(fd, &wh, sizeof(wh))) return false;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (draft_bytes && !send_all(fd, drafts, draft_bytes)) return false;
    return true;
}

bool ds4_replica_recv_mtp_resp(int fd,
                               ds4_replica_mtp_resp_hdr *hdr_out,
                               int32_t *drafts_buf,
                               uint32_t drafts_buf_cap,
                               int timeout_ms) {
    ds4_replica_wire_header wh;
    if (!ds4_replica_recv_header(fd, &wh, timeout_ms)) return false;
    if (wh.msg_type != DS4_REPLICA_MSG_MTP_DRAFT_RESP) { errno = EPROTO; return false; }
    if (wh.payload_bytes < sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, hdr_out, sizeof(*hdr_out), timeout_ms)) return false;
    size_t draft_bytes = (size_t)hdr_out->n_draft * sizeof(int32_t);
    if (draft_bytes != wh.payload_bytes - sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (hdr_out->n_draft > drafts_buf_cap) { errno = EMSGSIZE; return false; }
    if (draft_bytes && !ds4_replica_recv_payload(fd, drafts_buf, draft_bytes, timeout_ms)) return false;
    return true;
}

bool ds4_replica_send_expert_req(int fd,
                                 uint64_t seq,
                                 uint32_t layer_idx,
                                 uint32_t n_ids,
                                 const int32_t *ids,
                                 uint32_t gate_expert_bytes,
                                 uint32_t down_expert_bytes) {
    ds4_replica_expert_req_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.layer_idx = layer_idx;
    hdr.n_ids = n_ids;
    hdr.gate_expert_bytes = gate_expert_bytes;
    hdr.down_expert_bytes = down_expert_bytes;

    size_t id_bytes = (size_t)n_ids * sizeof(int32_t);
    size_t payload_bytes = sizeof(hdr) + id_bytes;

    ds4_replica_wire_header wh;
    memset(&wh, 0, sizeof(wh));
    wh.magic = DS4_REPLICA_PROTO_MAGIC;
    wh.version = DS4_REPLICA_PROTO_VERSION;
    wh.msg_type = DS4_REPLICA_MSG_EXPERT_REQ;
    wh.payload_bytes = (uint32_t)payload_bytes;
    wh.seq = seq;

    if (!send_all(fd, &wh, sizeof(wh))) return false;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (id_bytes && !send_all(fd, ids, id_bytes)) return false;
    return true;
}

bool ds4_replica_recv_expert_req(int fd,
                                 ds4_replica_expert_req_hdr *hdr_out,
                                 int32_t *ids_buf,
                                 uint32_t ids_buf_cap,
                                 int timeout_ms) {
    ds4_replica_wire_header wh;
    if (!ds4_replica_recv_header(fd, &wh, timeout_ms)) return false;
    if (wh.msg_type != DS4_REPLICA_MSG_EXPERT_REQ) { errno = EPROTO; return false; }
    if (wh.payload_bytes < sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, hdr_out, sizeof(*hdr_out), timeout_ms)) return false;
    size_t id_bytes = (size_t)hdr_out->n_ids * sizeof(int32_t);
    if (id_bytes != wh.payload_bytes - sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (hdr_out->n_ids > ids_buf_cap) { errno = EMSGSIZE; return false; }
    if (id_bytes && !ds4_replica_recv_payload(fd, ids_buf, id_bytes, timeout_ms)) return false;
    return true;
}

bool ds4_replica_send_expert_resp(int fd,
                                  uint64_t seq,
                                  uint32_t status,
                                  uint32_t n_ids,
                                  uint32_t gate_expert_bytes,
                                  uint32_t down_expert_bytes,
                                  const void *blocks,
                                  size_t blocks_bytes) {
    ds4_replica_expert_resp_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.status = status;
    hdr.n_ids = n_ids;
    hdr.gate_expert_bytes = gate_expert_bytes;
    hdr.down_expert_bytes = down_expert_bytes;

    size_t payload_bytes = sizeof(hdr) + blocks_bytes;
    if (payload_bytes > 0xFFFFFFFFu) { errno = EMSGSIZE; return false; }

    ds4_replica_wire_header wh;
    memset(&wh, 0, sizeof(wh));
    wh.magic = DS4_REPLICA_PROTO_MAGIC;
    wh.version = DS4_REPLICA_PROTO_VERSION;
    wh.msg_type = DS4_REPLICA_MSG_EXPERT_RESP;
    wh.payload_bytes = (uint32_t)payload_bytes;
    wh.seq = seq;

    if (!send_all(fd, &wh, sizeof(wh))) return false;
    if (!send_all(fd, &hdr, sizeof(hdr))) return false;
    if (blocks_bytes && !send_all(fd, blocks, blocks_bytes)) return false;
    return true;
}

bool ds4_replica_recv_expert_resp(int fd,
                                  ds4_replica_expert_resp_hdr *hdr_out,
                                  void *blocks_buf,
                                  size_t blocks_buf_bytes,
                                  int timeout_ms) {
    ds4_replica_wire_header wh;
    if (!ds4_replica_recv_header(fd, &wh, timeout_ms)) return false;
    if (wh.msg_type != DS4_REPLICA_MSG_EXPERT_RESP) { errno = EPROTO; return false; }
    if (wh.payload_bytes < sizeof(*hdr_out)) { errno = EBADMSG; return false; }
    if (!ds4_replica_recv_payload(fd, hdr_out, sizeof(*hdr_out), timeout_ms)) return false;
    size_t blocks_bytes = (size_t)wh.payload_bytes - sizeof(*hdr_out);
    if (blocks_bytes > blocks_buf_bytes) { errno = EMSGSIZE; return false; }
    if (blocks_bytes && !ds4_replica_recv_payload(fd, blocks_buf, blocks_bytes, timeout_ms)) return false;
    return true;
}

int64_t ds4_replica_ping_rtt_us(int fd, int timeout_ms) {
    uint64_t t0 = (uint64_t)now_us();
    if (!ds4_replica_send_msg(fd, DS4_REPLICA_MSG_PING, 0, t0, NULL, 0)) return -1;
    ds4_replica_wire_header h;
    if (!ds4_replica_recv_header(fd, &h, timeout_ms)) return -1;
    if (h.msg_type != DS4_REPLICA_MSG_PONG) { errno = EPROTO; return -1; }
    if (h.payload_bytes) {
        /* PONG carries no payload in v1; drain anyway. */
        uint8_t scratch[64];
        size_t left = h.payload_bytes;
        while (left) {
            size_t n = left < sizeof(scratch) ? left : sizeof(scratch);
            if (!ds4_replica_recv_payload(fd, scratch, n, timeout_ms)) return -1;
            left -= n;
        }
    }
    return now_us() - (int64_t)t0;
}

void ds4_replica_self_describe(ds4_replica_handshake *out,
                               ds4_replica_role role) {
    memset(out, 0, sizeof(*out));
    out->role = (uint32_t)role;
    out->backend = 0;  /* filled by engine when wired; 0 = unspecified */
    out->n_layer = 43;
    out->n_expert_per_layer = 256;
    out->n_expert_used = 6;
    out->kv_row_bytes = DS4_REPLICA_KV_ROW_BYTES;
    if (gethostname(out->hostname, sizeof(out->hostname) - 1) != 0) {
        snprintf(out->hostname, sizeof(out->hostname), "(unknown)");
    }
    out->hostname[sizeof(out->hostname) - 1] = '\0';
    snprintf(out->ds4_version, sizeof(out->ds4_version), "ds4-replica/v1");
}
