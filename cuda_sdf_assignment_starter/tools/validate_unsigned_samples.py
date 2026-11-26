#!/usr/bin/env python3
"""Independent exact unsigned point-to-triangle samples for SDF diagnostics.

Computes the minimum *geometric* distance (not closest vertex distance) for
reproducibly sampled grid nodes. This does NOT verify inside/outside signs.
Exit 0 PASS, 1 FAIL, 2 invalid input. This is an educational oracle, not a
replacement for a robust signed-reference hidden grader.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from compare_sdf import load_sdf


def read_mesh(path):
    vertices, faces = [], []
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            tokens = raw.strip().split()
            if not tokens or tokens[0].startswith('#'):
                continue
            if tokens[0] == 'v':
                if len(tokens) < 4:
                    raise ValueError('invalid OBJ vertex')
                vertices.append([float(x) for x in tokens[1:4]])
            elif tokens[0] == 'f':
                if len(tokens) != 4:
                    raise ValueError('triangle faces only')
                faces.append([int(t.split('/')[0])-1 for t in tokens[1:]])
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(tris) or not len(verts) or not np.all(np.isfinite(verts)):
        raise ValueError('empty/invalid OBJ')
    if np.min(tris) < 0 or np.max(tris) >= len(verts):
        raise ValueError('invalid OBJ face index')
    return verts[tris]


def segment_d2(points, a, b):
    d = b - a
    length2 = np.dot(d, d)
    u = np.clip(((points-a) @ d) / length2, 0, 1) if length2 > 0 else 0.0
    delta = points - (a + np.asarray(u)[..., None] * d)
    return np.einsum('ij,ij->i', delta, delta)


def exact_unsigned(points, triangles):
    best = np.full(len(points), np.inf, dtype=np.float64)
    for a, b, c in triangles:
        ab, ac = b-a, c-a
        n = np.cross(ab, ac)
        n2 = float(np.dot(n, n))
        d2 = np.minimum.reduce((segment_d2(points, a, b),
                                segment_d2(points, b, c), segment_d2(points, c, a)))
        if n2 > 1e-28:
            ap = points - a
            d00, d01, d11 = np.dot(ab, ab), np.dot(ab, ac), np.dot(ac, ac)
            denominator = d00 * d11 - d01 * d01
            if denominator > 0:
                d20, d21 = ap @ ab, ap @ ac
                v = (d11*d20-d01*d21) / denominator
                w = (d00*d21-d01*d20) / denominator
                inside = (v >= -1e-12) & (w >= -1e-12) & (v + w <= 1 + 1e-12)
                plane_d2 = np.square((ap @ n)) / n2
                d2 = np.where(inside, np.minimum(d2, plane_d2), d2)
        best = np.minimum(best, d2)
    return np.sqrt(np.maximum(best, 0))


def evaluate(mesh_path, sdf_path, count, seed, max_error_dx, max_bad_fraction):
    dims, origin, dx, sdf = load_sdf(sdf_path)
    tris = read_mesh(mesh_path)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(sdf), size=min(count, len(sdf)), replace=False))
    nx, ny, nz = dims
    x = indices % nx
    y = (indices // nx) % ny
    z = indices // (nx*ny)
    points = origin + dx * np.stack((x, y, z), axis=1)
    expected = exact_unsigned(points, tris)
    finite = bool(np.all(np.isfinite(sdf[indices])))
    err = np.abs(np.where(np.isfinite(sdf[indices]), np.abs(sdf[indices]), expected) - expected)
    bad = (err > max_error_dx*dx) | ~np.isfinite(sdf[indices])
    bad_fraction = float(np.mean(bad))
    passed = bool(finite and bad_fraction <= max_bad_fraction)
    return {
        'schema_version': 1, 'reference': 'independent exact unsigned point-triangle distance',
        'passed': passed, 'finite': finite, 'samples': len(indices), 'seed': seed,
        'mae': float(np.mean(err)), 'max_error': float(np.max(err)),
        'bad_fraction': bad_fraction, 'grid_spacing': dx,
        'tolerances': {'max_error_dx': max_error_dx, 'max_bad_fraction': max_bad_fraction},
        'warning': 'Unsigned distance only. Must independently check sign and closed mesh topology.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--sdf', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--max-error-dx', type=float, default=0.25)
    parser.add_argument('--max-bad-fraction', type=float, default=0.01)
    a = parser.parse_args()
    if a.samples < 1 or a.seed < 0 or any(not math.isfinite(v) or v < 0 for v in
                                          (a.max_error_dx, a.max_bad_fraction)) or a.max_bad_fraction > 1:
        parser.error('invalid sample count, seed, or tolerances')
    try:
        report = evaluate(a.mesh, a.sdf, a.samples, a.seed,
                          a.max_error_dx, a.max_bad_fraction)
    except (OSError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
