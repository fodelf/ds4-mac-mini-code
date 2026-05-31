#!/usr/bin/env python3
"""Memory-safe GGUF metadata dumper. Reads ONLY the header/KV/tensor-info region
(never tensor data). Prints KV keys (+scalar values / array summaries) and tensor
infos (focusing on expert tensors). Usage: dump_gguf_meta.py FILE.gguf"""
import struct, sys

GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',
              8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
SCALAR_FMT = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),
              6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}

def main(path):
    f = open(path, 'rb')
    magic = f.read(4)
    assert magic == b'GGUF', f'bad magic {magic!r}'
    version, = struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))
    print(f'== {path}')
    print(f'version={version} n_tensors={n_tensors} n_kv={n_kv}')

    def rstr():
        ln, = struct.unpack('<Q', f.read(8))
        return f.read(ln).decode('utf-8', 'replace')

    def rval(t):
        if t in SCALAR_FMT:
            fmt, sz = SCALAR_FMT[t]
            return struct.unpack('<'+fmt, f.read(sz))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, = struct.unpack('<I', f.read(4))
            cnt, = struct.unpack('<Q', f.read(8))
            if et == 8:
                vals = [rstr() for _ in range(cnt)]
            else:
                fmt, sz = SCALAR_FMT[et]
                raw = f.read(sz*cnt)
                vals = list(struct.unpack('<'+fmt*cnt, raw))
            return ('arr', GGUF_TYPES.get(et,et), cnt, vals)
        raise ValueError(f'unknown type {t}')

    print('--- KV ---')
    for _ in range(n_kv):
        key = rstr()
        t, = struct.unpack('<I', f.read(4))
        v = rval(t)
        if isinstance(v, tuple) and v[0] == 'arr':
            _, et, cnt, vals = v
            head = vals[:12]
            print(f'  {key} : arr<{et}>[{cnt}] {head}{" ..." if cnt>12 else ""}')
        else:
            sv = v if not isinstance(v, str) else (v[:80] + ('...' if len(v)>80 else ''))
            print(f'  {key} : {GGUF_TYPES.get(t,t)} = {sv}')

    print('--- TENSORS (expert + sample) ---')
    shown = 0
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack('<I', f.read(4))
        dims = list(struct.unpack('<'+'Q'*nd, f.read(8*nd)))
        ttype, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        if ('_exps.' in name and ('blk.0.' in name or 'blk.1.' in name)) or shown < 6:
            print(f'  {name}  dims={dims} type={ttype} off={off}')
            shown += 1
    f.close()

if __name__ == '__main__':
    main(sys.argv[1])
