constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

constant float dsv4_e2m1fn_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    int n_rot;
};

struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

struct ds4_metal_args_dsv4_indexer_qat {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
};

struct ds4_metal_args_dsv4_ratio4_shift {
    uint32_t width;
};

struct ds4_metal_args_dsv4_compressor_store_one {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

struct ds4_metal_args_dsv4_f32_to_f16_store_rows {
    uint32_t n_rows;
    uint32_t head_dim;
    /* Source rows are contiguous in the F32 scratch (head_dim * sizeof(float)).
     * Destination rows are contiguous in the F16 cache
     * (head_dim * sizeof(half)).  Both offsets are bound into the buffer base
     * via setBuffer:offset:, so the kernel itself just walks gid. */
};

// Lever A row layout for attn_comp_kv. The DS4 head dim is 512 with a 64-element
// RoPE tail; the 448-element non-RoPE prefix is quantised in 7 blocks of 64.
// Bytes 0..27   : 7 × float32 per-block scales
// Bytes 28..31  : pad to 8-byte alignment
// Bytes 32..479 : 448 × uchar FP8 (E4M3FN) magnitudes
// Bytes 480..607: 64 × half RoPE (precision-sensitive; never quantised)
// Total = 608 bytes per row.
struct ds4_metal_args_dsv4_f32_to_fp8_store_rows {
    uint32_t n_rows;
    uint32_t head_dim;
    uint32_t n_rot;
};

// Same row layout as above, used by both the heads8 kernels (direct read with
// in-kernel dequant) and the FlashAttention pack-in dequant-copy.
struct ds4_metal_args_dsv4_fp8_rows_to_f16 {
    uint32_t n_rows;
    uint32_t head_dim;
    uint32_t n_rot;
};

static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

static inline int dsv4_e4m3fn_encode_index(float ax) {
    ax = min(ax, 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return best;
}

static inline float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    return sign * dsv4_e4m3fn_value(dsv4_e4m3fn_encode_index(abs(x)));
}

static inline uchar dsv4_e4m3fn_encode_byte(float x) {
    const uchar sign = (x < 0.0f) ? (uchar)0x80 : (uchar)0x00;
    return sign | (uchar)dsv4_e4m3fn_encode_index(abs(x));
}

static inline float dsv4_e4m3fn_decode_byte(uchar b) {
    const float mag = dsv4_e4m3fn_value((int)(b & 0x7f));
    return (b & 0x80) ? -mag : mag;
}

static inline float dsv4_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 6.0f);
    int best = 0;
    float best_diff = abs(ax - dsv4_e2m1fn_values[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = abs(ax - dsv4_e2m1fn_values[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_values[best];
}

// Quantizes the non-RoPE part of a KV row through E4M3FN and writes the
// dequantized value back as float. DS4 uses this to match the FP8 KV-cache
// semantics while keeping the Metal graph's cache buffers float-addressable.
kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_base = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((device const float *) (src_base + (off + tid)*args.nb00));
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
            *((device float *) (dst_base + (off + tid)*args.nb0)) = q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *) (dst_base + i*args.nb0)) = *((device const float *) (src_base + i*args.nb00));
    }
}

// The official DS4 indexer applies a 128-wide Hadamard rotation and then an
// inplace FP4 activation-simulation pass to both indexer Q and indexer KV.
kernel void kernel_dsv4_indexer_hadamard_fp4_f32(
        constant ds4_metal_args_dsv4_indexer_qat & args,
        device   char  * x,
        threadgroup float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows || args.head_dim != 128u || tid >= 128u) {
        return;
    }

    threadgroup float *vals = scratch;
    threadgroup float *absbuf = scratch + 128;
    device float *xr = (device float *)(x + (uint64_t)row * args.row_stride);

    vals[tid] = xr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float v = vals[tid] * 0.08838834764831845f;
    const uint block = tid >> 5u;
    const uint lane = tid & 31u;
    const uint block_base = block * 32u;
    absbuf[tid] = abs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = max(absbuf[block_base + lane],
                                            absbuf[block_base + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float amax = max(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = exp2(ceil(log2(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
}

// Decode-side KV finalizer after RoPE. The normal RoPE kernel intentionally
// remains separate because tiny trigonometric codegen changes can flip later
// sampled tokens. This kernel only fuses the FP8 round-trip for the non-RoPE
// prefix with the F16-rounded raw-cache row used by FlashAttention.
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            raw[off + tid] = (float)((half)q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
        raw[i] = (float)((half)kv[i]);
    }
}

// Ratio-4 compression keeps two 4-row halves of recurrent state. After an
// emitted compressed row, the second half becomes the next window's previous
// half. The old encoder expressed this as four generic copies; this DS4-specific
// kernel performs the KV and score copies together.
kernel void kernel_dsv4_ratio4_shift_f32(
        constant ds4_metal_args_dsv4_ratio4_shift & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const uint n = 4u * args.width;
    if (gid >= n) return;

    state_kv[gid] = state_kv[n + gid];
    state_score[gid] = state_score[n + gid];
}

// One-token compressor frontier update. Decode appends exactly one projected KV
// row and one score row into a small recurrent state. The generic batch helper
// expresses this as APE copy, score add, and two set_rows operations; this
// kernel writes both state tensors directly while preserving the same
// score + APE arithmetic.
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char  * ape,
        device       float * state_kv,
        device       float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {
        return;
    }

    const uint pos_mod = args.pos % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * args.width + gid;
    const uint ape_i = pos_mod * args.width + gid;

    float ape_v;
    if (args.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}

// Commit the F32 producer-pipeline scratch slab into the F16 compressed-KV
// cache.  Each output element is the half-precision rounding of one F32 input.
// The compressed-KV F16 storage is the only DS4-side semantic precision change
// in #17: pool / rms_norm / rope_tail / FP8 / QAT still run in F32 inside the
// scratch tensor, so this kernel is the single point where the half() round
// happens.
kernel void kernel_dsv4_f32_to_f16_store_rows(
        constant ds4_metal_args_dsv4_f32_to_f16_store_rows & args,
        device const float * src,
        device       half  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint n = args.n_rows * args.head_dim;
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}

// Lever A producer-side commit: F32 producer scratch → FP8 (E4M3FN) + per-64-block
// scale + F16 RoPE tail, all in one pass. Replaces the previous "FP8 quantize
// (in-place on scratch) → F16 commit" sequence on the attn_comp path. The output
// row uses the 608-byte layout defined above so that consumer kernels can read
// directly from the cache without a separate dequant pass.
//
// One threadgroup per row. 64 threads cooperate: each handles one element per
// 64-element nope block (7 blocks total for DS4) and one element of the RoPE
// tail. RoPE values pass through half() exactly as the old #17 commit kernel
// did — Lever A leaves RoPE precision untouched.
kernel void kernel_dsv4_f32_to_fp8_store_rows(
        constant ds4_metal_args_dsv4_f32_to_fp8_store_rows & args,
        device const float * src,
        device       uchar * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows) return;

    const uint head_dim = args.head_dim;
    const uint n_rot    = args.n_rot;
    const uint n_nope   = head_dim - n_rot;
    const uint n_blocks = n_nope >> 6;                                 // 7 for DS4
    const uint scales_bytes = n_blocks * (uint)sizeof(float);          // 28
    const uint nope_off = (scales_bytes + 7u) & ~7u;                   // 32
    const uint rope_off = nope_off + n_nope;                           // 480
    const uint row_bytes = rope_off + n_rot * (uint)sizeof(half);      // 608

    device const float * src_row    = src + (uint64_t)row * head_dim;
    device       uchar * dst_row    = dst + (uint64_t)row * row_bytes;
    device       float * dst_scales = (device float *)dst_row;
    device       uchar * dst_fp8    = dst_row + nope_off;
    device       half  * dst_rope   = (device half *)(dst_row + rope_off);

    for (uint b = 0; b < n_blocks; b++) {
        const uint off = b << 6;
        const float v = (tid < 64u) ? src_row[off + tid] : 0.0f;
        if (tid < 64u) {
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax  = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid == 0u) {
            dst_scales[b] = scale;
        }
        if (tid < 64u) {
            dst_fp8[off + tid] = dsv4_e4m3fn_encode_byte(v / scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < n_rot) {
        dst_rope[tid] = half(src_row[n_nope + tid]);
    }
}

// Lever A FlashAttention pack-in helper: unpack 608-byte FP8+scale rows from the
// attn_comp cache into a flat F16 working buffer that the static-mixed /
// gathered / decode-mixed-batch FA kernels already know how to consume.  This
// keeps those FA kernels untouched while still letting Lever A halve the
// resident cache.
//
// One threadgroup per row, 128 threads each. Threads 0..111 cover the FP8 nope
// span (4 bytes per thread × 112 threads = 448 elements); threads 112..127
// cover the F16 RoPE tail (4 halves per thread × 16 threads = 64 elements).
// Each nope thread loads its block's scale and multiplies the dequantised value
// before storing into the destination half4.
kernel void kernel_dsv4_fp8_rows_to_f16(
        constant ds4_metal_args_dsv4_fp8_rows_to_f16 & args,
        device const uchar * src,
        device       half  * dst,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows) return;

    const uint head_dim = args.head_dim;
    const uint n_rot    = args.n_rot;
    const uint n_nope   = head_dim - n_rot;
    const uint n_blocks = n_nope >> 6;
    const uint scales_bytes = n_blocks * (uint)sizeof(float);
    const uint nope_off = (scales_bytes + 7u) & ~7u;
    const uint rope_off = nope_off + n_nope;
    const uint row_bytes = rope_off + n_rot * (uint)sizeof(half);

    device const uchar * src_row = src + (uint64_t)row * row_bytes;
    device       half4 * dst_row = (device half4 *)(dst + (uint64_t)row * head_dim);

    const uint n_nope4 = n_nope >> 2;     // 112 for DS4: 4 FP8 bytes per slot
    const uint n_rope4 = n_rot  >> 2;     // 16  for DS4: 4 halves per slot

    if (tid < n_nope4) {
        const uint block = (tid << 2) >> 6;             // (tid*4)/64
        const float scale = ((device const float *)src_row)[block];
        device const uchar4 *bytes = (device const uchar4 *)(src_row + nope_off + tid * 4u);
        const uchar4 b = *bytes;
        dst_row[tid] = half4(
            half(dsv4_e4m3fn_decode_byte(b.x) * scale),
            half(dsv4_e4m3fn_decode_byte(b.y) * scale),
            half(dsv4_e4m3fn_decode_byte(b.z) * scale),
            half(dsv4_e4m3fn_decode_byte(b.w) * scale));
    } else if (tid < n_nope4 + n_rope4) {
        const uint r = tid - n_nope4;
        device const half4 *halves = (device const half4 *)(src_row + rope_off + r * (uint)sizeof(half4));
        dst_row[tid] = *halves;
    }
}
