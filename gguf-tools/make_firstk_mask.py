#!/usr/bin/env python3
"""Make a DSXM expert-mask that keeps experts 0..K-1 in every layer (the
simplest possible "first-K" selection — for smoke/pipeline bring-up where
expert *quality* is not the goal). Output format matches make_expert_mask.py
(DSXM v1) so shrink_gguf.py consumes it directly.

Usage:
    python3 gguf-tools/make_firstk_mask.py --layers 43 --n-expert 256 \
        --keep 4 --out /tmp/mask-k4.bin
"""
import argparse, struct, sys

MAGIC = b"DSXM"
VERSION = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--n-expert", type=int, required=True)
    ap.add_argument("--keep", type=int, required=True, help="keep experts 0..keep-1 per layer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (1 <= a.keep <= a.n_expert):
        raise SystemExit(f"--keep {a.keep} out of [1,{a.n_expert}]")

    n_bits = a.layers * a.n_expert
    bits = bytearray((n_bits + 7) // 8)
    for L in range(a.layers):
        for e in range(a.keep):
            idx = L * a.n_expert + e
            bits[idx >> 3] |= 1 << (idx & 7)

    blob = MAGIC + struct.pack("<IIII", VERSION, a.layers, a.n_expert, 0) + bytes(bits)
    with open(a.out, "wb") as f:
        f.write(blob)
    print(f"wrote {a.out}: layers={a.layers} n_expert={a.n_expert} keep=first-{a.keep} "
          f"kept_slots={a.layers*a.keep}/{n_bits} bytes={len(blob)}", file=sys.stderr)


if __name__ == "__main__":
    main()
