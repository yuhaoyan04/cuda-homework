#!/usr/bin/env python3
"""Independent CPU/GPU output comparison. Run only on trusted, size-limited files."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_sdf(path: Path):
    # Same 3-line header as upstream SDFGen, followed by one ASCII float per line.
    with path.open('r', encoding='utf-8') as f:
        header1 = f.readline().split()
        header2 = f.readline().split()
        header3 = f.readline().split()
    if len(header1) != 3 or len(header2) != 3 or len(header3) != 1:
        raise ValueError(f'{path}: invalid SDF header')
    dims = tuple(int(v) for v in header1)
    if min(dims) <= 0 or math.prod(dims) > 30_000_000:
        raise ValueError(f'{path}: invalid/excessive grid dimensions')
    origin = np.asarray([float(v) for v in header2], dtype=np.float64)
    dx = float(header3[0])
    if not np.all(np.isfinite(origin)) or not (np.isfinite(dx) and dx > 0):
        raise ValueError(f'{path}: invalid grid metadata')
    values = np.loadtxt(path, dtype=np.float64, skiprows=3, ndmin=1)
    if values.size != math.prod(dims):
        raise ValueError(f'{path}: expected {math.prod(dims)} values, got {values.size}')
    return dims, origin, dx, values


def compare(ref_path: Path, candidate_path: Path, abs_tol_dx: float, rel_tol: float,
            max_bad_fraction: float, max_sign_fraction: float, sign_ignore_dx: float):
    rd, ro, rh, rv = load_sdf(ref_path)
    cd, co, ch, cv = load_sdf(candidate_path)
    if rd != cd or not np.allclose(ro, co, rtol=0, atol=1e-6 * rh) or not np.isclose(rh, ch, rtol=1e-7, atol=0):
        raise ValueError('reference and candidate grid metadata differ')
    finite = np.isfinite(cv)
    finite_all = bool(np.all(finite))
    if not np.all(np.isfinite(rv)):
        raise ValueError('reference contains NaN/Inf')
    # Failure if any NaN/Inf exists; use finite values to compute diagnostics.
    err = np.abs(np.where(finite, cv, rv) - rv)
    threshold = abs_tol_dx * rh + rel_tol * np.abs(rv)
    bad = (err > threshold) | ~finite
    # Sign is not tested near the zero level set: sign can be unstable at/on surface.
    sign_mask = (np.abs(rv) > sign_ignore_dx * rh) & finite
    sign_bad = ((rv > 0) != (cv > 0)) & sign_mask
    n_sign = int(np.sum(sign_mask))
    bad_fraction = float(np.mean(bad))
    sign_fraction = float(np.sum(sign_bad) / n_sign) if n_sign else 0.0
    passed = bool(finite_all and bad_fraction <= max_bad_fraction and sign_fraction <= max_sign_fraction)
    report = {
        'schema_version': 2, 'reference': str(ref_path), 'candidate': str(candidate_path),
        'resolution': list(rd), 'spacing': rh, 'samples': int(rv.size),
        'finite': finite_all, 'mae': float(np.mean(err)), 'rmse': float(np.sqrt(np.mean(err * err))),
        'max_error': float(np.max(err)), 'p95_error': float(np.quantile(err, .95)),
        'bad_fraction': bad_fraction, 'sign_error_fraction': sign_fraction,
        'sign_tested_samples': n_sign, 'passed': passed,
        'tolerances': {'abs_tol_dx': abs_tol_dx, 'rel_tol': rel_tol,
                       'max_bad_fraction': max_bad_fraction, 'max_sign_fraction': max_sign_fraction,
                       'sign_ignore_dx': sign_ignore_dx},
        'warning': 'PROVISIONAL only: CPU baseline is approximate. This CPU comparison is not a stand-alone final grading oracle; also use analytic/independent exact and hidden tests.'
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--abs-tol-dx', type=float, default=0.1)
    parser.add_argument('--rel-tol', type=float, default=1e-4)
    parser.add_argument('--max-bad-fraction', type=float, default=.01)
    parser.add_argument('--max-sign-fraction', type=float, default=.005)
    parser.add_argument('--sign-ignore-dx', type=float, default=.25)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in [args.abs_tol_dx, args.rel_tol,
               args.max_bad_fraction, args.max_sign_fraction, args.sign_ignore_dx]):
        parser.error('All tolerances must be finite and nonnegative')
    if args.max_bad_fraction > 1 or args.max_sign_fraction > 1:
        parser.error('Fractions must be <= 1')
    try:
        result = compare(args.reference,args.candidate,args.abs_tol_dx,args.rel_tol,
                         args.max_bad_fraction,args.max_sign_fraction,args.sign_ignore_dx)
    except (OSError, ValueError) as e:
        print('ERROR:',e)
        return 2
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','mae','max_error','bad_fraction','sign_error_fraction']},indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
