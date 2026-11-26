#!/usr/bin/env python3
"""Check the public dataset bundle without using CUDA or NumPy."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets'


def main():
    errors = []
    hashes = (DATA / 'SHA256SUMS').read_text(encoding='ascii').splitlines()
    count = 0
    for line in hashes:
        if not line.strip():
            continue
        expected, rel = line.strip().split(maxsplit=1)
        path = (DATA / rel).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():
            errors.append('Missing or unsafe path: ' + rel)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append('SHA256 mismatch: ' + rel)
        count += 1
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['cases']:
        for key in ('mesh', 'grid'):
            path = (DATA / item[key]).resolve()
            if not path.is_relative_to(DATA.resolve()) or not path.is_file():
                errors.append(f"Missing/unsafe manifest {item['id']} {key}: {item[key]}")
    def signature(filename):
        vertices=0; triangles=[]
        for line in (DATA/'obj'/filename).read_text(encoding='utf-8').splitlines():
            parts=line.split()
            if not parts: continue
            if parts[0]=='v': vertices+=1
            elif parts[0]=='f': triangles.append(tuple(int(v.split('/')[0]) for v in parts[1:]))
        return (vertices, tuple(triangles))
    sigs=[signature(f'deform_{i:03}.obj') for i in range(3)]
    if not (sigs[0]==sigs[1]==sigs[2]):
        errors.append('Bonus deform frames have inconsistent topology')
    if errors:
        for e in errors: print('FAIL:',e, file=sys.stderr)
        return 1
    print(f'PASS: {count} SHA256 files, {len(manifest["cases"])} cases, dynamic frame topology consistent')
    return 0

if __name__ == '__main__':
    sys.exit(main())
