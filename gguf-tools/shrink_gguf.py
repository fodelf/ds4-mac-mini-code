#!/usr/bin/env python3
"""Shrink a DS4 GGUF by physically dropping disabled routed experts.

Inputs:
    --in  PATH      original DS4 GGUF (e.g. ds4flash.gguf)
    --mask PATH     DSXM mask file produced by make_expert_mask.py
    --out PATH      output GGUF

For each layer L the mask says which of the 256 routed experts to KEEP.
This tool rewrites blk.{L}.ffn_{gate,up,down}_exps.weight as 3D tensors
whose last dim is the kept count (not 256), preserving the original block
quantization layout (IQ2_XXS / Q2_K).  All other tensors are passed
through verbatim.  Two new GGUF metadata entries record the mapping for
the engine:

    ds4.expert_keep_map.kept_counts   array<i32>   length n_layers
    ds4.expert_keep_map.original_ids  array<i32>   length sum(kept_counts);
                                                  concatenated per layer

Together they let the engine route token -> kept slot at runtime
(task #12 in the deployment plan).

Design notes:
  - Streams the source file in 64 MB chunks; never mmaps and never loads
    the full GGUF or the tokenizer vocab into RAM.  This makes the tool
    safe on the 16 GB Mac Mini where DS4 lives.
  - GGUF KV entries are copied as raw byte ranges (no parsing of large
    arrays).  Only entries we INSERT are serialized fresh.
  - GGUF v3 only (matches what ds4 ships with).

WARNING: this rewrites tensor data offsets.  Output is a fully self-
contained GGUF, but you cannot binary-diff it against the input.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass


GGUF_MAGIC = b"GGUF"

# GGUF metadata value type ids.
GT_U8, GT_I8 = 0, 1
GT_U16, GT_I16 = 2, 3
GT_U32, GT_I32 = 4, 5
GT_F32 = 6
GT_BOOL = 7
GT_STRING = 8
GT_ARRAY = 9
GT_U64, GT_I64 = 10, 11
GT_F64 = 12

_SCALAR_BYTES = {
    GT_U8: 1, GT_I8: 1,
    GT_U16: 2, GT_I16: 2,
    GT_U32: 4, GT_I32: 4,
    GT_F32: 4,
    GT_BOOL: 1,
    GT_U64: 8, GT_I64: 8,
    GT_F64: 8,
}

# GGUF tensor types we touch (only routed experts get inspected for block size).
TT_F32 = 0
TT_F16 = 1
TT_Q4_0 = 2
TT_Q4_1 = 3
TT_Q5_0 = 6
TT_Q5_1 = 7
TT_Q8_0 = 8
TT_Q8_1 = 9
TT_Q2_K = 10
TT_Q3_K = 11
TT_Q4_K = 12
TT_Q5_K = 13
TT_Q6_K = 14
TT_Q8_K = 15
TT_IQ2_XXS = 16
TT_IQ2_XS = 17
TT_IQ3_XXS = 18
TT_IQ1_S = 19
TT_IQ4_NL = 20
TT_IQ3_S = 21
TT_IQ2_S = 22
TT_IQ4_XS = 23
TT_I8 = 24
TT_I16 = 25
TT_I32 = 26
TT_I64 = 27
TT_F64 = 28
TT_IQ1_M = 29
TT_BF16 = 30

# (block_elements, block_bytes) for every GGUF type we may encounter.
# A scalar type has block_elements = 1 and block_bytes = element_bytes.
_TYPE_LAYOUT = {
    TT_F32:    (1, 4),
    TT_F16:    (1, 2),
    TT_BF16:   (1, 2),
    TT_F64:    (1, 8),
    TT_I8:     (1, 1),
    TT_I16:    (1, 2),
    TT_I32:    (1, 4),
    TT_I64:    (1, 8),
    TT_Q4_0:   (32, 18),
    TT_Q4_1:   (32, 20),
    TT_Q5_0:   (32, 22),
    TT_Q5_1:   (32, 24),
    TT_Q8_0:   (32, 34),
    TT_Q8_1:   (32, 36),
    TT_Q2_K:   (256, 84),
    TT_Q3_K:   (256, 110),
    TT_Q4_K:   (256, 144),
    TT_Q5_K:   (256, 176),
    TT_Q6_K:   (256, 210),
    TT_Q8_K:   (256, 292),
    TT_IQ2_XXS:(256, 66),
    TT_IQ2_XS: (256, 74),
    TT_IQ3_XXS:(256, 98),
    TT_IQ1_S:  (256, 50),
    TT_IQ4_NL: (32, 18),
    TT_IQ3_S:  (256, 110),
    TT_IQ2_S:  (256, 82),
    TT_IQ4_XS: (256, 136),
    TT_IQ1_M:  (256, 56),
}


def _type_name(t: int) -> str:
    names = {
        TT_F32:"F32", TT_F16:"F16", TT_BF16:"BF16", TT_F64:"F64",
        TT_I8:"I8", TT_I16:"I16", TT_I32:"I32", TT_I64:"I64",
        TT_Q4_0:"Q4_0", TT_Q4_1:"Q4_1", TT_Q5_0:"Q5_0", TT_Q5_1:"Q5_1",
        TT_Q8_0:"Q8_0", TT_Q8_1:"Q8_1",
        TT_Q2_K:"Q2_K", TT_Q3_K:"Q3_K", TT_Q4_K:"Q4_K",
        TT_Q5_K:"Q5_K", TT_Q6_K:"Q6_K", TT_Q8_K:"Q8_K",
        TT_IQ2_XXS:"IQ2_XXS", TT_IQ2_XS:"IQ2_XS", TT_IQ3_XXS:"IQ3_XXS",
        TT_IQ1_S:"IQ1_S", TT_IQ4_NL:"IQ4_NL", TT_IQ3_S:"IQ3_S",
        TT_IQ2_S:"IQ2_S", TT_IQ4_XS:"IQ4_XS", TT_IQ1_M:"IQ1_M",
    }
    return names.get(t, f"TT{t}")


def _tensor_bytes(ttype: int, ne: list[int]) -> int:
    if ttype not in _TYPE_LAYOUT:
        raise SystemExit(f"unhandled tensor type id={ttype}")
    block_n, block_b = _TYPE_LAYOUT[ttype]
    n_elem = 1
    for d in ne:
        n_elem *= d
    if n_elem % block_n != 0:
        raise SystemExit(f"element count {n_elem} not multiple of block {block_n} for type {_type_name(ttype)}")
    return (n_elem // block_n) * block_b


# DSXM mask reader (must match make_expert_mask.py).
MASK_MAGIC = b"DSXM"
MASK_VERSION = 1


def read_mask(path: str) -> tuple[int, int, list[list[int]]]:
    """Returns (n_layers, n_expert, keep_lists) where keep_lists[L] is the
    sorted list of original expert ids that survive in layer L."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != MASK_MAGIC:
        raise SystemExit(f"{path}: bad magic {blob[:4]!r}, expected DSXM")
    version, n_layers, n_expert, reserved = struct.unpack("<IIII", blob[4:20])
    if version != MASK_VERSION:
        raise SystemExit(f"{path}: unsupported mask version {version}")
    bits = blob[20:]
    expected = (n_layers * n_expert + 7) // 8
    if len(bits) < expected:
        raise SystemExit(f"{path}: short mask body {len(bits)} < {expected}")
    keep = [[] for _ in range(n_layers)]
    for L in range(n_layers):
        for e in range(n_expert):
            idx = L * n_expert + e
            if bits[idx >> 3] & (1 << (idx & 7)):
                keep[L].append(e)
    return n_layers, n_expert, keep


# -----------------------------------------------------------------------------
# GGUF reader that records byte ranges of KV pairs and tensor info entries
# without parsing array contents (so the tokenizer vocab stays on disk).

@dataclass
class KvSpan:
    key: str
    type_id: int            # outer type
    start: int              # byte offset of the key length prefix
    end: int                # byte offset one past the value bytes


@dataclass
class TensorRow:
    name: str
    ne: list[int]
    ttype: int
    offset: int             # offset within data section, as stored in input
    info_start: int         # byte offset in file of this tensor info row
    info_end: int


class GgufScanner:
    def __init__(self, path: str):
        self.path = path
        self.fp = open(path, "rb")
        self.size = os.fstat(self.fp.fileno()).st_size
        self._scan_header()

    def close(self):
        self.fp.close()

    def _u32(self) -> int:
        return struct.unpack("<I", self.fp.read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self.fp.read(8))[0]

    def _read_string(self) -> str:
        n = self._u64()
        if n > (1 << 30):
            raise SystemExit(f"string too long ({n}) at {self.fp.tell()}")
        return self.fp.read(n).decode("utf-8", errors="replace")

    def _skip_string(self) -> None:
        n = self._u64()
        self.fp.seek(n, os.SEEK_CUR)

    def _skip_value(self, t: int) -> None:
        if t in _SCALAR_BYTES:
            self.fp.seek(_SCALAR_BYTES[t], os.SEEK_CUR)
            return
        if t == GT_STRING:
            self._skip_string()
            return
        if t == GT_ARRAY:
            elem_t = self._u32()
            n = self._u64()
            if elem_t == GT_STRING:
                for _ in range(n):
                    self._skip_string()
            elif elem_t in _SCALAR_BYTES:
                self.fp.seek(_SCALAR_BYTES[elem_t] * n, os.SEEK_CUR)
            elif elem_t == GT_ARRAY:
                # GGUF spec disallows nested arrays.
                raise SystemExit("nested arrays are not allowed by GGUF spec")
            else:
                raise SystemExit(f"unknown array elem type {elem_t}")
            return
        raise SystemExit(f"unknown value type {t}")

    def _scan_header(self):
        magic = self.fp.read(4)
        if magic != GGUF_MAGIC:
            raise SystemExit(f"{self.path}: not a GGUF file")
        self.version = self._u32()
        if self.version != 3:
            raise SystemExit(f"unsupported GGUF version {self.version} (only v3 is supported)")
        self.tensor_count = self._u64()
        self.kv_count = self._u64()

        self.alignment = 32
        self.kv_spans: list[KvSpan] = []
        for _ in range(self.kv_count):
            start = self.fp.tell()
            key = self._read_string()
            t = self._u32()
            # Pluck alignment if we see it, but still skip via the generic path.
            if key == "general.alignment" and t == GT_U32:
                # Peek without consuming so the skip path still advances cleanly.
                pos = self.fp.tell()
                val = self._u32()
                self.alignment = int(val)
                # Already advanced past the scalar; record span.
                end = self.fp.tell()
                self.kv_spans.append(KvSpan(key, t, start, end))
                continue
            self._skip_value(t)
            end = self.fp.tell()
            self.kv_spans.append(KvSpan(key, t, start, end))

        self.tensors: list[TensorRow] = []
        for _ in range(self.tensor_count):
            info_start = self.fp.tell()
            name = self._read_string()
            n_dims = self._u32()
            ne = [self._u64() for _ in range(n_dims)]
            ttype = self._u32()
            offset = self._u64()
            info_end = self.fp.tell()
            self.tensors.append(TensorRow(name, ne, ttype, offset, info_start, info_end))

        pos = self.fp.tell()
        pad = (-pos) % self.alignment
        if pad:
            self.fp.seek(pad, os.SEEK_CUR)
        self.data_section_start = self.fp.tell()

    def copy_byte_range(self, fp_out, start: int, end: int, chunk: int = 64 * 1024 * 1024) -> None:
        """Copy [start, end) from input to fp_out in chunks (no full-buffer load)."""
        self.fp.seek(start)
        remaining = end - start
        while remaining > 0:
            n = min(chunk, remaining)
            buf = self.fp.read(n)
            if len(buf) != n:
                raise SystemExit(f"short read at {self.fp.tell() - len(buf)}: got {len(buf)}/{n}")
            fp_out.write(buf)
            remaining -= n


# -----------------------------------------------------------------------------
# GGUF writer helpers (scalar + array of int32 only — that's all we add).

def _w_u32(fp, v: int) -> None: fp.write(struct.pack("<I", v))
def _w_u64(fp, v: int) -> None: fp.write(struct.pack("<Q", v))
def _w_i32(fp, v: int) -> None: fp.write(struct.pack("<i", v))

def _w_string(fp, s: str) -> None:
    b = s.encode("utf-8")
    _w_u64(fp, len(b))
    fp.write(b)

def _w_kv_array_i32(fp, key: str, values: list[int]) -> None:
    _w_string(fp, key)
    _w_u32(fp, GT_ARRAY)
    _w_u32(fp, GT_I32)
    _w_u64(fp, len(values))
    fp.write(struct.pack(f"<{len(values)}i", *values))


# -----------------------------------------------------------------------------
# Expert tensor classification.

import re

_EXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def _expert_layer_of(name: str) -> tuple[int, str] | None:
    m = _EXP_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


# -----------------------------------------------------------------------------
# Plan output tensor layout.

@dataclass
class OutTensor:
    name: str
    ne: list[int]
    ttype: int
    src_offset: int         # offset within source data section
    src_bytes: int          # bytes in source for this whole tensor
    out_bytes: int          # bytes in output for this tensor (== src_bytes for unmodified)
    keep_list: list[int] | None   # original-expert ids to keep, or None for verbatim copy
    out_offset: int = 0     # filled in after layout


def plan_output(scanner: GgufScanner, keep: list[list[int]], n_expert_mask: int) -> list[OutTensor]:
    # Compute source byte length per tensor by sorted-offset adjacency, ignoring
    # the in-file order of the tensor info table.
    file_data_end = scanner.size - scanner.data_section_start
    by_offset = sorted(scanner.tensors, key=lambda t: t.offset)
    src_size = {}
    for i, t in enumerate(by_offset):
        nxt = by_offset[i + 1].offset if i + 1 < len(by_offset) else file_data_end
        src_size[t.name] = nxt - t.offset

    out_list: list[OutTensor] = []
    for t in scanner.tensors:
        cls = _expert_layer_of(t.name)
        ne = list(t.ne)
        full_src_bytes = _tensor_bytes(t.ttype, ne)
        if full_src_bytes > src_size[t.name]:
            raise SystemExit(
                f"tensor {t.name}: computed {full_src_bytes} > slot {src_size[t.name]} — wrong type/layout"
            )

        if cls is None:
            out_list.append(OutTensor(
                name=t.name, ne=ne, ttype=t.ttype,
                src_offset=t.offset, src_bytes=src_size[t.name],
                out_bytes=full_src_bytes, keep_list=None,
            ))
            continue

        layer, part = cls
        if layer >= len(keep):
            raise SystemExit(f"{t.name}: layer {layer} not covered by mask (mask has {len(keep)} layers)")
        # Expert axis is the LAST dim (slowest-varying); GGUF stores fastest-first.
        if ne[-1] != n_expert_mask:
            raise SystemExit(
                f"{t.name}: last-dim {ne[-1]} != mask n_expert {n_expert_mask}"
            )
        new_ne = ne[:]
        new_ne[-1] = len(keep[layer])
        if new_ne[-1] < 1:
            raise SystemExit(f"{t.name}: layer {layer} keeps zero experts")
        new_bytes = _tensor_bytes(t.ttype, new_ne)
        out_list.append(OutTensor(
            name=t.name, ne=new_ne, ttype=t.ttype,
            src_offset=t.offset, src_bytes=src_size[t.name],
            out_bytes=new_bytes, keep_list=keep[layer],
        ))
    return out_list


# -----------------------------------------------------------------------------
# Compute output layout: KV section + tensor info section + aligned data section.
# This is two-pass because tensor info encodes data-section offsets, and the
# tensor info section size is fixed by the input (names + n_dims unchanged).

def measure_kv_section(scanner: GgufScanner) -> int:
    # Sum of byte spans of every KV pair we will copy.
    return sum(s.end - s.start for s in scanner.kv_spans)


def measure_new_kv_bytes(layer_count: int, kept_counts: list[int], original_ids_flat: list[int]) -> int:
    # ds4.expert_keep_map.kept_counts  array<i32> length=layer_count
    # ds4.expert_keep_map.original_ids array<i32> length=sum(kept_counts)
    n = 0
    for key, arr_len in (
        ("ds4.expert_keep_map.kept_counts", layer_count),
        ("ds4.expert_keep_map.original_ids", len(original_ids_flat)),
    ):
        key_b = key.encode("utf-8")
        n += 8 + len(key_b)        # u64 strlen + string bytes
        n += 4                     # outer type u32 = GT_ARRAY
        n += 4                     # elem type u32 = GT_I32
        n += 8                     # length u64
        n += 4 * arr_len           # i32 values
    return n


def measure_tensor_info_bytes(out_list: list[OutTensor]) -> int:
    n = 0
    for t in out_list:
        name_b = t.name.encode("utf-8")
        n += 8 + len(name_b)     # name string
        n += 4                   # n_dims u32
        n += 8 * len(t.ne)       # ne[] u64
        n += 4                   # ttype u32
        n += 8                   # offset u64
    return n


def assign_offsets(out_list: list[OutTensor], alignment: int) -> int:
    pos = 0
    for t in out_list:
        if pos % alignment != 0:
            pos += alignment - (pos % alignment)
        t.out_offset = pos
        pos += t.out_bytes
    return pos


# -----------------------------------------------------------------------------
# Write a single shrunken expert tensor by copying only the kept expert blocks.

def write_shrunken_expert(scanner: GgufScanner, t: OutTensor, fp_out,
                          n_expert_src: int, chunk: int = 64 * 1024 * 1024) -> None:
    assert t.keep_list is not None
    full_src_bytes = _tensor_bytes(t.ttype, t.ne[:-1] + [n_expert_src])
    if full_src_bytes % n_expert_src != 0:
        raise SystemExit(f"{t.name}: per-expert byte count not integral")
    per_expert = full_src_bytes // n_expert_src
    src_data_base = scanner.data_section_start + t.src_offset
    written = 0
    for new_idx, orig_e in enumerate(t.keep_list):
        scanner.fp.seek(src_data_base + orig_e * per_expert)
        remaining = per_expert
        while remaining > 0:
            n = min(chunk, remaining)
            buf = scanner.fp.read(n)
            if len(buf) != n:
                raise SystemExit(f"{t.name}: short read at expert {orig_e}")
            fp_out.write(buf)
            remaining -= n
            written += n
    if written != t.out_bytes:
        raise SystemExit(f"{t.name}: wrote {written} bytes, expected {t.out_bytes}")


def write_verbatim_tensor(scanner: GgufScanner, t: OutTensor, fp_out,
                          chunk: int = 64 * 1024 * 1024) -> None:
    src_start = scanner.data_section_start + t.src_offset
    # The tensor occupies its declared logical size; trailing slack (gap to
    # next tensor) is alignment padding and is regenerated on output.
    scanner.copy_byte_range(fp_out, src_start, src_start + t.out_bytes, chunk=chunk)


# -----------------------------------------------------------------------------
# Feasibility report (printed unconditionally; nothing is run without --do-it).

def feasibility(scanner: GgufScanner, out_list: list[OutTensor], out_path: str,
                keep: list[list[int]], n_expert_src: int) -> dict:
    src_total = scanner.size
    sum_out_tensor_bytes = sum(t.out_bytes for t in out_list)
    sum_src_tensor_bytes = sum(t.src_bytes for t in out_list)  # includes alignment slack
    routed_src = sum(t.src_bytes for t in out_list if t.keep_list is not None)
    routed_out = sum(t.out_bytes for t in out_list if t.keep_list is not None)
    total_kept = sum(len(k) for k in keep)
    total_slots = len(keep) * n_expert_src
    info = {
        "src_file_GB": src_total / 1e9,
        "out_file_GB_est": (src_total - routed_src + routed_out) / 1e9,
        "routed_src_GB": routed_src / 1e9,
        "routed_out_GB": routed_out / 1e9,
        "savings_GB": (routed_src - routed_out) / 1e9,
        "kept_experts": total_kept,
        "total_expert_slots": total_slots,
        "kept_pct": 100.0 * total_kept / total_slots,
        "n_tensors": len(out_list),
    }
    return info


def _print_feasibility(info: dict, out_path: str) -> None:
    print("feasibility (no work has been done yet):", file=sys.stderr)
    print(f"  source GGUF             : {info['src_file_GB']:.2f} GB", file=sys.stderr)
    print(f"  est output GGUF         : {info['out_file_GB_est']:.2f} GB -> {out_path}", file=sys.stderr)
    print(f"  routed-expert bytes in  : {info['routed_src_GB']:.2f} GB", file=sys.stderr)
    print(f"  routed-expert bytes out : {info['routed_out_GB']:.2f} GB", file=sys.stderr)
    print(f"  disk savings            : {info['savings_GB']:.2f} GB", file=sys.stderr)
    print(f"  experts kept            : {info['kept_experts']} / {info['total_expert_slots']} ({info['kept_pct']:.1f}%)", file=sys.stderr)
    print(f"  tensors total           : {info['n_tensors']}", file=sys.stderr)
    print(f"  read pattern            : 64 MB seek+read per chunk (no mmap, no full-tensor load)", file=sys.stderr)
    print(f"  write pattern           : single output file streamed in same chunk size", file=sys.stderr)
    print(f"  peak RAM                : O(KV table + tensor info table + 64 MB IO buffer)", file=sys.stderr)


# -----------------------------------------------------------------------------
# Main pipeline.

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True, help="source DS4 GGUF path")
    ap.add_argument("--mask", required=True, help="DSXM mask file (from make_expert_mask.py)")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--do-it", action="store_true",
                    help="actually write the output file; without this, just prints the feasibility report and exits")
    ap.add_argument("--chunk-mb", type=int, default=64,
                    help="IO chunk size in MB (default 64)")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        raise SystemExit("--in and --out must differ")

    chunk = max(1, args.chunk_mb) * 1024 * 1024

    n_layers, n_expert_mask, keep = read_mask(args.mask)
    scanner = GgufScanner(args.src)

    out_list = plan_output(scanner, keep, n_expert_mask)
    info = feasibility(scanner, out_list, args.out, keep, n_expert_mask)
    _print_feasibility(info, args.out)

    if not args.do_it:
        print("\n--do-it not passed; exiting without writing.", file=sys.stderr)
        scanner.close()
        return

    # Build the per-layer kept_counts and concatenated original_ids.
    kept_counts = [len(k) for k in keep]
    original_ids_flat: list[int] = []
    for k in keep:
        original_ids_flat.extend(k)

    # Lay out the output file.
    header_bytes = 4 + 4 + 8 + 8        # magic + version + tensor_count + kv_count
    new_kv_total = (
        measure_kv_section(scanner)
        + measure_new_kv_bytes(len(keep), kept_counts, original_ids_flat)
    )
    tensor_info_bytes = measure_tensor_info_bytes(out_list)
    pre_data = header_bytes + new_kv_total + tensor_info_bytes
    pad = (-pre_data) % scanner.alignment
    data_section_start = pre_data + pad
    total_data_bytes = assign_offsets(out_list, scanner.alignment)

    print(f"\nlayout: header={header_bytes}B  kv={new_kv_total}B  tensor_info={tensor_info_bytes}B  "
          f"pre_data={pre_data}B  align_pad={pad}B  data={total_data_bytes/1e9:.2f}GB  "
          f"total={(data_section_start + total_data_bytes)/1e9:.2f}GB", file=sys.stderr)

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = args.out + ".partial"
    with open(tmp_path, "wb") as fp:
        # Header.
        fp.write(GGUF_MAGIC)
        _w_u32(fp, 3)                                        # version
        _w_u64(fp, len(out_list))                            # tensor_count
        new_kv_count = len(scanner.kv_spans) + 2             # +2 inserted entries
        _w_u64(fp, new_kv_count)

        # KV: copy originals verbatim, then append our two new entries.
        for s in scanner.kv_spans:
            scanner.copy_byte_range(fp, s.start, s.end, chunk=chunk)
        _w_kv_array_i32(fp, "ds4.expert_keep_map.kept_counts", kept_counts)
        _w_kv_array_i32(fp, "ds4.expert_keep_map.original_ids", original_ids_flat)

        # Tensor info table.  Offsets are relative to the data section start.
        for t in out_list:
            _w_string(fp, t.name)
            _w_u32(fp, len(t.ne))
            for d in t.ne:
                _w_u64(fp, d)
            _w_u32(fp, t.ttype)
            _w_u64(fp, t.out_offset)

        # Sanity: pre-data length.
        pos = fp.tell()
        if pos != pre_data:
            raise SystemExit(f"layout mismatch: wrote {pos}B pre-data but planned {pre_data}B")

        # Align to data section.
        if pad:
            fp.write(b"\x00" * pad)

        # Data section: write each tensor.  Stream-only.
        for ti, t in enumerate(out_list):
            # Pad to alignment between tensors.
            pos = fp.tell()
            want = data_section_start + t.out_offset
            if pos > want:
                raise SystemExit(f"{t.name}: overshot — pos {pos} > expected {want}")
            if pos < want:
                fp.write(b"\x00" * (want - pos))

            if t.keep_list is None:
                write_verbatim_tensor(scanner, t, fp, chunk=chunk)
            else:
                write_shrunken_expert(scanner, t, fp, n_expert_src=n_expert_mask, chunk=chunk)

            if (ti + 1) % 64 == 0 or ti == len(out_list) - 1:
                done_GB = fp.tell() / 1e9
                print(f"  tensor {ti+1}/{len(out_list)}  written={done_GB:.2f} GB", file=sys.stderr)

    os.replace(tmp_path, args.out)
    scanner.close()
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
