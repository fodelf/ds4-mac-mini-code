#ifndef DS4_REPLICA_H
#define DS4_REPLICA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Dual-host disaggregated inference wire protocol (path B).
 *
 * Host = Mac Mini M4 16G, runs ds4-server + decode hot path.
 * Replica = MacBook carved ~8G, runs prefill + KV cold tier + cold expert tier.
 * Link = Thunderbolt direct, theoretical ceiling 40 Gbps (~5 GB/s) single dir.
 *
 * This header pins the wire format. KV stream chunks are byte-for-byte
 * compatible with the host's in-memory FP8 attn_comp_kv layout (Lever A,
 * 608 B/row, see ds4.c:121).  That keeps the receive path zero-copy:
 * memcpy straight into g->layer_attn_comp_cache[il]+row_start*608.
 *
 * All multi-byte fields are little-endian. Both sides are Apple Silicon,
 * so this is a no-op today; the constraint is recorded so we never lose it.
 */

#define DS4_REPLICA_PROTO_MAGIC   0x44533452u  /* 'DS4R' */
#define DS4_REPLICA_PROTO_VERSION 1u

/* Must match DSV4_FP8_ATTN_ROW_BYTES in ds4.c:121 (Lever A landed). */
#define DS4_REPLICA_KV_ROW_BYTES  608u

typedef enum {
    DS4_REPLICA_ROLE_HOST        = 1,
    DS4_REPLICA_ROLE_REPLICA     = 2,
    DS4_REPLICA_ROLE_MTP_REPLICA = 3,
} ds4_replica_role;

typedef enum {
    DS4_REPLICA_MSG_HANDSHAKE       = 0x01,
    DS4_REPLICA_MSG_HANDSHAKE_ACK   = 0x02,
    DS4_REPLICA_MSG_KV_STREAM       = 0x10,
    DS4_REPLICA_MSG_KV_HANDOVER     = 0x11,
    DS4_REPLICA_MSG_KV_HANDOVER_ACK = 0x12,
    DS4_REPLICA_MSG_EXPERT_REQ      = 0x20,
    DS4_REPLICA_MSG_EXPERT_RESP     = 0x21,
    DS4_REPLICA_MSG_PREFILL_REQ     = 0x30,
    DS4_REPLICA_MSG_PREFILL_DONE    = 0x31,
    /* MTP off-host drafting (host stays on the target model; replica owns
     * the speculative drafter). Per-token wire = prev_hc (n_hc*n_embd f32)
     * + token id + pos. Response = draft token id + (optional) new hc state. */
    DS4_REPLICA_MSG_MTP_DRAFT_REQ   = 0x40,
    DS4_REPLICA_MSG_MTP_DRAFT_RESP  = 0x41,
    DS4_REPLICA_MSG_MTP_INVALIDATE  = 0x42,
    DS4_REPLICA_MSG_PING            = 0x80,
    DS4_REPLICA_MSG_PONG            = 0x81,
    DS4_REPLICA_MSG_BYE             = 0xFF,
} ds4_replica_msg_type;

#pragma pack(push, 1)

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t msg_type;
    uint32_t payload_bytes;
    uint32_t flags;
    uint64_t seq;
    uint64_t reserved;
} ds4_replica_wire_header;  /* exactly 32 bytes */

typedef struct {
    uint32_t layer_idx;
    uint32_t row_start;
    uint32_t row_count;
    uint32_t row_bytes;   /* must == DS4_REPLICA_KV_ROW_BYTES */
    uint32_t total_rows;
    uint32_t pad0;
} ds4_replica_kv_stream_hdr;  /* 24 bytes; rows follow */

typedef struct {
    uint32_t role;
    uint32_t backend;
    uint64_t total_ram_bytes;
    uint64_t free_ram_bytes;
    uint32_t n_layer;
    uint32_t n_expert_per_layer;
    uint32_t n_expert_used;
    uint32_t kv_row_bytes;
    char     hostname[64];
    char     ds4_version[32];
} ds4_replica_handshake;  /* 136 bytes */

/* MTP draft request header. hc payload follows: hc_floats * 4 bytes.
 *
 * The host ships the target model's just-computed cur_hc (the hyper-connection
 * state after committing `seed_token`).  The replica reseeds its MTP drafter
 * from this state and runs `draft_cap` recursive steps, returning the proposed
 * token ids.  The replica keeps the MTP SWA raw cache internally across bursts;
 * `prev_accepted` from the last cycle is folded into the request so the replica
 * can roll the cache back to base+accepted before drafting — this avoids a
 * separate invalidate round-trip on the hot path.  seed_pos = absolute target
 * position of seed_token (drives MTP SWA addressing on the replica side). */
typedef struct {
    int32_t  seed_token;     /* the just-committed target token */
    uint32_t seed_pos;       /* absolute target position of seed_token */
    uint32_t hc_floats;      /* must equal n_hc * n_embd */
    uint32_t draft_cap;      /* number of recursive draft steps requested */
    uint32_t prev_accepted;  /* drafts accepted from the previous burst */
    uint32_t eos_token;      /* stop drafting after proposing this id */
} ds4_replica_mtp_req_hdr;   /* 24 bytes; hc payload follows */

typedef struct {
    uint32_t status;         /* 0 = ok, non-zero = replica error */
    uint32_t n_draft;        /* number of draft token ids that follow */
    uint32_t reserved0;
    uint32_t reserved1;
} ds4_replica_mtp_resp_hdr;  /* 16 bytes; n_draft * int32 draft ids follow */

/* Routed-expert cold tier (path B without MTP): the host pins dense weights in
 * RAM locally and keeps a small routed-expert LRU; on a decode-layer miss it
 * asks the replica (which holds a larger RAM-pinned routed pool) for that
 * layer's missed experts in one batched round-trip.  Request = layer_idx +
 * n_ids + the per-expert byte sizes (so the replica validates against its own
 * GGUF) followed by n_ids int32 expert ids.  Response = for each id, the
 * gate||up||down bytes concatenated (gate and up share gate_expert_bytes). */
typedef struct {
    uint32_t layer_idx;
    uint32_t n_ids;
    uint32_t gate_expert_bytes;  /* per-expert gate (== up) byte size */
    uint32_t down_expert_bytes;  /* per-expert down byte size */
} ds4_replica_expert_req_hdr;    /* 16 bytes; n_ids * int32 ids follow */

typedef struct {
    uint32_t status;             /* 0 = ok, non-zero = replica error */
    uint32_t n_ids;              /* that many expert blocks follow */
    uint32_t gate_expert_bytes;
    uint32_t down_expert_bytes;
} ds4_replica_expert_resp_hdr;   /* 16 bytes; per id: gate||up||down bytes follow */

#pragma pack(pop)

/* Transport setup ---------------------------------------------------------- */

/* Host calls listen+accept on the replica side; replica calls connect from host.
 * Both return a connected socket fd, or -1 on error (errno set). */
int  ds4_replica_listen(uint16_t port, const char *bind_addr);
int  ds4_replica_accept(int listen_fd, int timeout_ms);
int  ds4_replica_connect(const char *host, uint16_t port, int timeout_ms);
void ds4_replica_close(int fd);

/* Tune socket for low-latency small-message exchange (TCP_NODELAY etc). */
void ds4_replica_tune_socket(int fd);

/* Framed send/recv -------------------------------------------------------- */

bool ds4_replica_send_msg(int fd,
                          ds4_replica_msg_type type,
                          uint32_t flags,
                          uint64_t seq,
                          const void *payload,
                          size_t payload_bytes);

/* recv_header reads exactly 32 bytes and validates magic/version. Returns
 * true and fills out_hdr on success. Spurious bytes terminate the connection. */
bool ds4_replica_recv_header(int fd,
                             ds4_replica_wire_header *out_hdr,
                             int timeout_ms);

bool ds4_replica_recv_payload(int fd, void *buf, size_t len, int timeout_ms);

/* High-level helpers ------------------------------------------------------ */

/* Handshake: each side sends its DS4_REPLICA_MSG_HANDSHAKE first, then awaits
 * the peer's HANDSHAKE_ACK. Either side may initiate.  Returns true and
 * fills peer_out on success. */
bool ds4_replica_do_handshake(int fd,
                              const ds4_replica_handshake *self,
                              ds4_replica_handshake *peer_out,
                              int timeout_ms);

/* KV stream: replica side calls _send for each chunk during prefill;
 * host side calls _recv to drain into its attn_comp_kv cache.
 * rows points to row_count * DS4_REPLICA_KV_ROW_BYTES contiguous bytes
 * laid out exactly like g->layer_attn_comp_cache[layer_idx] starting at
 * row_start. */
bool ds4_replica_kv_stream_send(int fd,
                                uint64_t seq,
                                uint32_t layer_idx,
                                uint32_t row_start,
                                uint32_t row_count,
                                uint32_t total_rows,
                                const void *rows);

/* Receive the next KV stream chunk header into hdr_out, then the row bytes
 * into rows_buf (must hold hdr.row_count * hdr.row_bytes).  Caller is
 * responsible for stitching multiple chunks into layer-major order. */
bool ds4_replica_kv_stream_recv(int fd,
                                ds4_replica_kv_stream_hdr *hdr_out,
                                void *rows_buf,
                                size_t rows_buf_bytes,
                                int timeout_ms);

/* MTP off-host drafting -----------------------------------------------------
 *
 * Host calls _send_mtp_req after committing a target token: it ships the hc
 * state row block (hc_floats * sizeof(float)) so the replica can run one MTP
 * drafter step.  Replica calls _recv_mtp_req to drain the header+payload, runs
 * the drafter, and replies with _send_mtp_resp (draft id + new hc state, and
 * optionally the mtp logits).  Host drains the reply with _recv_mtp_resp.
 *
 * hc_buf on both sides must hold hc_floats floats; logits_buf (host side) must
 * hold the model vocab size when request_logits was set. */
bool ds4_replica_send_mtp_req(int fd,
                              uint64_t seq,
                              int32_t seed_token,
                              uint32_t seed_pos,
                              uint32_t hc_floats,
                              uint32_t draft_cap,
                              uint32_t prev_accepted,
                              uint32_t eos_token,
                              const float *hc);

bool ds4_replica_recv_mtp_req(int fd,
                              ds4_replica_mtp_req_hdr *hdr_out,
                              float *hc_buf,
                              uint32_t hc_buf_floats,
                              int timeout_ms);

bool ds4_replica_send_mtp_resp(int fd,
                               uint64_t seq,
                               uint32_t status,
                               uint32_t n_draft,
                               const int32_t *drafts);

bool ds4_replica_recv_mtp_resp(int fd,
                               ds4_replica_mtp_resp_hdr *hdr_out,
                               int32_t *drafts_buf,
                               uint32_t drafts_buf_cap,
                               int timeout_ms);

/* Routed-expert cold tier ---------------------------------------------------
 *
 * Host calls _send_expert_req with the layer's missed expert ids, then drains
 * the reply with _recv_expert_resp into a flat buffer (n_ids blocks, each
 * 2*gate_expert_bytes + down_expert_bytes).  Replica calls _recv_expert_req,
 * gathers the bytes from its RAM-pinned pool (or mmap), and replies with
 * _send_expert_resp. */
bool ds4_replica_send_expert_req(int fd,
                                 uint64_t seq,
                                 uint32_t layer_idx,
                                 uint32_t n_ids,
                                 const int32_t *ids,
                                 uint32_t gate_expert_bytes,
                                 uint32_t down_expert_bytes);

bool ds4_replica_recv_expert_req(int fd,
                                 ds4_replica_expert_req_hdr *hdr_out,
                                 int32_t *ids_buf,
                                 uint32_t ids_buf_cap,
                                 int timeout_ms);

bool ds4_replica_send_expert_resp(int fd,
                                  uint64_t seq,
                                  uint32_t status,
                                  uint32_t n_ids,
                                  uint32_t gate_expert_bytes,
                                  uint32_t down_expert_bytes,
                                  const void *blocks,
                                  size_t blocks_bytes);

bool ds4_replica_recv_expert_resp(int fd,
                                  ds4_replica_expert_resp_hdr *hdr_out,
                                  void *blocks_buf,
                                  size_t blocks_buf_bytes,
                                  int timeout_ms);

/* ping/pong RTT probe — used by M1 RDMA-over-TB spike + ongoing health
 * check.  Returns measured RTT in microseconds, or -1 on failure. */
int64_t ds4_replica_ping_rtt_us(int fd, int timeout_ms);

/* Build a self-description for handshake. fills .role, .backend, .n_layer
 * etc.; caller fills hostname and ds4_version. */
void ds4_replica_self_describe(ds4_replica_handshake *out,
                               ds4_replica_role role);

#endif  /* DS4_REPLICA_H */
