/* e0_pingpong.c — Task 04 / E0: Thunderbolt all-reduce latency gate (TP go/no-go).
 *
 * Minimal TCP ping-pong, deliberately standalone: links nothing from the ds4
 * core, loads no model, allocates only KB-level buffers (memory-safe by
 * construction). Measures round-trip latency + jitter over a thunderbolt
 * direct link to decide whether tensor parallelism is worth doing.
 *
 * TP costs ~86 layer syncs/token (2 all-reduce x 43 layers); RTT decides the
 * whole approach:
 *   RTT <= ~50us  -> ~2.6-4.3 ms/token sync  -> TP viable (continue #05)
 *   RTT >= ~150us -> ~13 ms/token+           -> TP no-go (fall back to #08)
 *
 * Build:  make e0    (or: cc -O3 -std=c99 -o e0-pingpong tools/e0_pingpong.c -lm)
 * Server: ./e0-pingpong --listen 169.254.188.38:5599 --size 32768 --iters 10000
 * Client: ./e0-pingpong --connect 169.254.188.38:5599 --size 32768 --iters 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/socket.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Read exactly n bytes or fail (TCP recv can return short). */
static int read_full(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r == 0) { errno = ECONNRESET; return -1; }
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n > 0) {
        ssize_t w = send(fd, p, n, 0);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 0;
}

static void set_nodelay(int fd) {
    int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0)
        fprintf(stderr, "warning: TCP_NODELAY failed: %s\n", strerror(errno));
}

static int parse_hostport(const char *s, char *host, size_t hostlen, int *port) {
    const char *colon = strrchr(s, ':');
    if (!colon || colon == s) return -1;
    size_t hl = (size_t)(colon - s);
    if (hl >= hostlen) return -1;
    memcpy(host, s, hl);
    host[hl] = '\0';
    *port = atoi(colon + 1);
    return (*port > 0 && *port < 65536) ? 0 : -1;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ----- server: echo `size` bytes back, `iters` times ----- */
static int run_server(const char *host, int port, size_t size, long iters) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "bad listen address: %s\n", host); return 1;
    }
    if (bind(ls, (struct sockaddr *)&addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (listen(ls, 1) != 0) { perror("listen"); return 1; }

    fprintf(stderr, "[e0] listening on %s:%d  size=%zu iters=%ld\n", host, port, size, iters);
    int cs = accept(ls, NULL, NULL);
    if (cs < 0) { perror("accept"); return 1; }
    set_nodelay(cs);
    fprintf(stderr, "[e0] client connected, echoing...\n");

    /* Echo until the client closes the link — the client does warmup
     * round-trips on top of `iters`, so a fixed count would close early. */
    (void)iters;
    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) { fprintf(stderr, "oom\n"); return 1; }
    long echoed = 0;
    for (;;) {
        if (read_full(cs, buf, size) != 0) break; /* EOF/reset = client done */
        if (write_full(cs, buf, size) != 0) { perror("send"); free(buf); return 1; }
        echoed++;
    }
    fprintf(stderr, "[e0] done, %ld round-trips echoed\n", echoed);
    free(buf);
    close(cs);
    close(ls);
    return 0;
}

/* ----- client: time `iters` round-trips, report stats ----- */
static int run_client(const char *host, int port, size_t size, long iters) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return 1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "bad connect address: %s\n", host); return 1;
    }
    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) { perror("connect"); return 1; }
    set_nodelay(s);

    uint8_t *buf = (uint8_t *)malloc(size);
    double *rtt = (double *)malloc((size_t)iters * sizeof(double));
    if (!buf || !rtt) { fprintf(stderr, "oom\n"); return 1; }
    memset(buf, 0xA5, size);

    /* warm up the link (TCP ramp, page-ins) */
    for (int w = 0; w < 64; w++) {
        if (write_full(s, buf, size) != 0 || read_full(s, buf, size) != 0) {
            perror("warmup"); return 1;
        }
    }

    double wall0 = now_sec();
    for (long i = 0; i < iters; i++) {
        double t0 = now_sec();
        if (write_full(s, buf, size) != 0) { perror("send"); return 1; }
        if (read_full(s, buf, size) != 0) { perror("recv"); return 1; }
        rtt[i] = (now_sec() - t0) * 1e6; /* us */
    }
    double wall1 = now_sec();

    qsort(rtt, (size_t)iters, sizeof(double), cmp_double);
    double sum = 0.0;
    for (long i = 0; i < iters; i++) sum += rtt[i];
    double mean = sum / (double)iters;
    double var = 0.0;
    for (long i = 0; i < iters; i++) { double d = rtt[i] - mean; var += d * d; }
    double stddev = sqrt(var / (double)iters);

    double rtt_min = rtt[0];
    double rtt_med = rtt[iters / 2];
    double rtt_p99 = rtt[(long)(iters * 0.99)];
    double rtt_max = rtt[iters - 1];

    /* effective throughput: 2 transfers (there + back) per round-trip */
    double total_gb = (double)iters * (double)size * 2.0 / 1e9;
    double thrpt = total_gb / (wall1 - wall0);

    /* TP per-token sync cost: 86 syncs/token, one-way latency ~= RTT/2 */
    double sync_ms_token = 86.0 * (rtt_med / 2.0) / 1000.0;

    printf("\n=== E0 Thunderbolt ping-pong ===\n");
    printf("peer=%s:%d  payload=%zu B  iters=%ld\n", host, port, size, iters);
    printf("RTT us:  min=%.2f  median=%.2f  p99=%.2f  max=%.2f\n",
           rtt_min, rtt_med, rtt_p99, rtt_max);
    printf("jitter:  stddev=%.2f us  (p99-median=%.2f us)\n", stddev, rtt_p99 - rtt_med);
    printf("throughput: %.2f GB/s  (%.1f s wall, %.2f GB moved)\n",
           thrpt, wall1 - wall0, total_gb);
    printf("TP est:  86 syncs/token x %.2fus one-way = %.2f ms/token sync overhead\n",
           rtt_med / 2.0, sync_ms_token);

    const char *verdict;
    if (rtt_med <= 50.0)       verdict = "GO  (RTT<=~50us) -> continue #05 Stage 2 TP skeleton";
    else if (rtt_med >= 150.0) verdict = "NO-GO (RTT>=~150us) -> fall back to single-M4 + Q5 + MTP (#08)";
    else                       verdict = "MARGINAL (50-150us) -> only if TP+Q5 still pencils to >=20 t/s";
    printf("VERDICT: %s\n", verdict);

    free(buf);
    free(rtt);
    close(s);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --listen  HOST:PORT [--size N] [--iters N]   (server / echo)\n"
        "  %s --connect HOST:PORT [--size N] [--iters N]   (client / measure)\n"
        "Defaults: --size 32768  --iters 10000\n", prog, prog);
}

int main(int argc, char **argv) {
    const char *listen_arg = NULL, *connect_arg = NULL;
    size_t size = 32768;
    long iters = 10000;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--listen") && i + 1 < argc) listen_arg = argv[++i];
        else if (!strcmp(argv[i], "--connect") && i + 1 < argc) connect_arg = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = strtol(argv[++i], NULL, 10);
        else { usage(argv[0]); return 2; }
    }
    if ((!listen_arg) == (!connect_arg)) { usage(argv[0]); return 2; }
    if (size == 0 || iters <= 0) { fprintf(stderr, "size/iters must be > 0\n"); return 2; }

    char host[256];
    int port;
    const char *hp = listen_arg ? listen_arg : connect_arg;
    if (parse_hostport(hp, host, sizeof(host), &port) != 0) {
        fprintf(stderr, "bad HOST:PORT: %s\n", hp); return 2;
    }

    return listen_arg ? run_server(host, port, size, iters)
                      : run_client(host, port, size, iters);
}
