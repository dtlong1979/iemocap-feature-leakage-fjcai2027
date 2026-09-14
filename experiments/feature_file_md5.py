from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('files', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out = {}
    for f in args.files:
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        out[str(f)] = {'md5': digest, 'bytes': f.stat().st_size}
        print(f'{digest}  {f.stat().st_size:>12,}  {f}')
    groups = {}
    for name, rec in out.items():
        groups.setdefault(rec['md5'], []).append(name)
    for digest, names in groups.items():
        if len(names) > 1:
            print(f'byte-identical ({digest}): ' + ', '.join(names))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
