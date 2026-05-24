#!/usr/bin/env python3
"""Analyze MoE router weights in a DS4 GGUF and rank experts by importance.

For each layer's `blk.N.ffn_gate_inp.weight` tensor (the router input
projection, F16, shape [n_embd, n_expert]), compute the per-expert L2 norm.
Each expert occupies a contiguous block of `n_embd` F16 values; the L2 norm
of that block is the magnitude of the routing direction the router learned
for that expert.

Low-norm experts are dropping candidates: they contribute little to the
softmax over all 256 routing logits and rarely make it into the top-6 active
set.  This script never runs the model — it streams the GGUF, seeks to each
router tensor (~3 MB per layer for V4 Flash), and ignores everything else.
Total RAM cost is dominated by one router layer at a time.

Usage:
    python3 gguf-tools/router_norms.py ds4flash.gguf --out /tmp/router_norms.json
    python3 gguf-tools/router_norms.py ds4flash.gguf --summary
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass


GGUF_MAGIC = b"GGUF"

# GGUF metadata value types.
GT_U8, GT_I8 = 0, 1
GT_U16, GT_I16 = 2, 3
GT_U32, GT_I32 = 4, 5
GT_F32 = 6
GT_BOOL = 7
GT_STRING = 8
GT_ARRAY = 9
GT_U64, GT_I64 = 10, 11
GT_F64 = 12

# GGUF tensor data type ids we care about.  Only F16 is needed for routers.
TT_F32 = 0
TT_F16 = 1


class GgufError(Exception):
    pass


class GgufReader:
    """Minimal streaming GGUF reader.  Only knows how to walk the header and
    locate tensor data offsets.  Does not load tensor data unless asked."""

    def __init__(self, path: str):
        self.path = path
        self.fp = open(path, "rb")
        self._read_header()

    def close(self):
        self.fp.close()

    def _u32(self) -> int:
        return struct.unpack("<I", self.fp.read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self.fp.read(8))[0]

    def _i32(self) -> int:
        return struct.unpack("<i", self.fp.read(4))[0]

    def _i64(self) -> int:
        return struct.unpack("<q", self.fp.read(8))[0]

    def _f32(self) -> float:
        return struct.unpack("<f", self.fp.read(4))[0]

    def _f64(self) -> float:
        return struct.unpack("<d", self.fp.read(8))[0]

    def _str(self) -> str:
        n = self._u64()
        if n > (1 << 30):
            raise GgufError(f"unreasonable string length {n} at offset {self.fp.tell()}")
        return self.fp.read(n).decode("utf-8", errors="replace")

    def _scalar(self, t: int):
        if t == GT_U8:     return self.fp.read(1)[0]
        if t == GT_I8:     return struct.unpack("<b", self.fp.read(1))[0]
        if t == GT_U16:    return struct.unpack("<H", self.fp.read(2))[0]
        if t == GT_I16:    return struct.unpack("<h", self.fp.read(2))[0]
        if t == GT_U32:    return self._u32()
        if t == GT_I32:    return self._i32()
        if t == GT_F32:    return self._f32()
        if t == GT_BOOL:   return self.fp.read(1)[0] != 0
        if t == GT_STRING: return self._str()
        if t == GT_U64:    return self._u64()
        if t == GT_I64:    return self._i64()
        if t == GT_F64:    return self._f64()
        raise GgufError(f"unknown scalar type {t}")

    def _value(self, t: int):
        if t != GT_ARRAY:
            return self._scalar(t)
        elem_t = self._u32()
        n = self._u64()
        if elem_t == GT_STRING:
            return [self._str() for _ in range(n)]
        # For array-of-scalar we don't need the values themselves, only to
        # advance the file pointer.  But callers do want known scalars (e.g.
        # alignment), so just collect them — arrays in headers are small.
        return [self._scalar(elem_t) for _ in range(n)]

    def _read_header(self):
        magic = self.fp.read(4)
        if magic != GGUF_MAGIC:
            raise GgufError(f"not a GGUF file: magic={magic!r}")
        self.version = self._u32()
        if self.version < 2 or self.version > 3:
            raise GgufError(f"unsupported GGUF version {self.version}")
        self.tensor_count = self._u64()
        self.kv_count = self._u64()

        # Walk metadata, remember anything we need.
        self.alignment = 32
        for _ in range(self.kv_count):
            key = self._str()
            t = self._u32()
            val = self._value(t)
            if key == "general.alignment":
                self.alignment = int(val)

        # Tensor info table.
        self.tensors: list[TensorInfo] = []
        for _ in range(self.tensor_count):
            name = self._str()
            n_dims = self._u32()
            ne = [self._u64() for _ in range(n_dims)]
            ttype = self._u32()
            offset = self._u64()
            self.tensors.append(TensorInfo(name, ne, ttype, offset))

        # Align file position up to data section start.
        pos = self.fp.tell()
        pad = (-pos) % self.alignment
        if pad:
            self.fp.seek(pad, os.SEEK_CUR)
        self.data_section_start = self.fp.tell()

    def read_tensor_f16(self, t: "TensorInfo") -> bytes:
        if t.ttype != TT_F16:
            raise GgufError(f"tensor {t.name!r} is not F16 (type id {t.ttype})")
        n_elem = 1
        for d in t.ne:
            n_elem *= d
        n_bytes = n_elem * 2
        self.fp.seek(self.data_section_start + t.offset)
        buf = self.fp.read(n_bytes)
        if len(buf) != n_bytes:
            raise GgufError(f"short read for {t.name!r}: {len(buf)} / {n_bytes}")
        return buf


@dataclass
class TensorInfo:
    name: str
    ne: list[int]
    ttype: int
    offset: int


def _l2_norms_f16(buf: bytes, n_embd: int, n_expert: int) -> list[float]:
    """Compute per-expert L2 norm.  Each expert is n_embd consecutive F16
    values, expert id is the outer index (ne[1])."""
    # Prefer numpy when available (fast).  Fall back to struct (slow but
    # zero-dep).  V4 Flash has ~110M F16 values across all routers — pure
    # Python handles it in seconds with iter_unpack.
    try:
        import numpy as np
        a = np.frombuffer(buf, dtype="<f2").astype("f4")
        a = a.reshape(n_expert, n_embd)
        sq = (a * a).sum(axis=1)
        return (sq ** 0.5).tolist()
    except ImportError:
        pass

    norms = []
    stride = n_embd * 2
    fmt = f"<{n_embd}e"
    for e in range(n_expert):
        chunk = buf[e * stride : (e + 1) * stride]
        vals = struct.unpack(fmt, chunk)
        s = 0.0
        for v in vals:
            s += v * v
        norms.append(s ** 0.5)
    return norms


_ROUTER_PREFIX = "blk."
_ROUTER_SUFFIX = ".ffn_gate_inp.weight"


def _layer_id_from_router_name(name: str) -> int | None:
    if not name.startswith(_ROUTER_PREFIX) or not name.endswith(_ROUTER_SUFFIX):
        return None
    mid = name[len(_ROUTER_PREFIX) : -len(_ROUTER_SUFFIX)]
    try:
        return int(mid)
    except ValueError:
        return None


def analyze(path: str) -> dict:
    r = GgufReader(path)
    routers: list[tuple[int, TensorInfo]] = []
    for t in r.tensors:
        layer = _layer_id_from_router_name(t.name)
        if layer is not None:
            routers.append((layer, t))
    routers.sort(key=lambda x: x[0])

    if not routers:
        raise GgufError("no router tensors (blk.N.ffn_gate_inp.weight) found")

    # All routers must agree on shape.
    first = routers[0][1]
    if len(first.ne) != 2:
        raise GgufError(f"router {first.name!r} has unexpected n_dims={len(first.ne)}")
    n_embd, n_expert = first.ne[0], first.ne[1]
    for _, t in routers:
        if t.ne != [n_embd, n_expert]:
            raise GgufError(f"router shape mismatch: {t.name!r} ne={t.ne} vs {first.ne}")

    by_layer = []
    for layer, t in routers:
        buf = r.read_tensor_f16(t)
        norms = _l2_norms_f16(buf, n_embd, n_expert)
        # rank descending: rank[0] is the expert with highest norm
        rank = sorted(range(n_expert), key=lambda e: -norms[e])
        by_layer.append({"layer": layer, "norms": norms, "rank": rank})
        print(
            f"layer {layer:3d}: min={min(norms):.4f} median={sorted(norms)[n_expert//2]:.4f} "
            f"max={max(norms):.4f}",
            file=sys.stderr,
        )

    # Global ranking across (layer, expert) pairs — useful if you want to
    # drop the absolute weakest globally rather than the weakest per-layer.
    flat = []
    for entry in by_layer:
        for e, v in enumerate(entry["norms"]):
            flat.append((entry["layer"], e, v))
    flat.sort(key=lambda x: -x[2])

    r.close()
    return {
        "n_embd": n_embd,
        "n_expert": n_expert,
        "n_layers": len(by_layer),
        "by_layer": by_layer,
        "global_rank": flat,
    }


def _print_summary(result: dict, top_k: int) -> None:
    n_expert = result["n_expert"]
    n_layers = result["n_layers"]
    print(f"\nGGUF router analysis:")
    print(f"  n_embd  = {result['n_embd']}")
    print(f"  n_expert= {n_expert}")
    print(f"  n_layers= {n_layers}")
    print()

    # Per-layer "drop floor": the L2 norm at the K-th kept expert.
    # Tokens that would prefer experts below this floor will see degraded
    # routing.
    print(f"  per-layer norm at keep-top-K cutoffs:")
    print(f"  {'layer':>5}  {'min':>8} {'k=8':>8} {'k=16':>8} {'k=32':>8} {'k=64':>8} {'k=128':>8} {'max':>8}")
    for entry in result["by_layer"]:
        norms = entry["norms"]
        rank = entry["rank"]
        # Sort descending; norms[rank[k-1]] is the K-th highest norm.
        def nth(k):
            return norms[rank[min(k, len(rank)) - 1]] if k >= 1 else float("nan")
        print(
            f"  {entry['layer']:>5d}  {min(norms):>8.4f} {nth(8):>8.4f} {nth(16):>8.4f} "
            f"{nth(32):>8.4f} {nth(64):>8.4f} {nth(128):>8.4f} {max(norms):>8.4f}"
        )

    # Distribution of the bottom experts across all layers.
    print()
    print(f"  global bottom-{top_k} (layer, expert, norm):")
    bottom = result["global_rank"][-top_k:]
    for layer, expert, norm in reversed(bottom):
        print(f"    L{layer:3d}.E{expert:3d}  norm={norm:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf", help="path to a DS4 GGUF file")
    ap.add_argument("--out", help="write full result as JSON to this path")
    ap.add_argument("--summary", action="store_true", help="print human-readable summary")
    ap.add_argument("--top", type=int, default=20, help="how many bottom experts to print in --summary")
    args = ap.parse_args()

    result = analyze(args.gguf)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.summary or not args.out:
        _print_summary(result, args.top)


if __name__ == "__main__":
    main()
