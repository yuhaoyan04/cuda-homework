#!/usr/bin/env python3
"""Independent analytic check for the supplied axis-aligned cube (half extent 0.55).

Return 0 only if the numeric AND sign checks pass; 1 for a failed check;
2 for malformed input or invalid checker arguments.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from compare_sdf import load_sdf


def evaluate(path, half_extent, max_mae_dx, max_error_dx, max_sign_fraction,
             sign_ignore_dx):
    dims, org, dx, values = load_sdf(path)
    nx, ny, nz = dims
    zs, ys, xs = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    points = org + dx * np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=1)
    q = np.abs(points) - half_extent
    exact = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
    finite = bool(np.all(np.isfinite(values)))
    # An invalid value must always fail; replace only for finite diagnostics.
    error = np.abs(np.where(np.isfinite(values), values, exact) - exact)
    tested = np.abs(exact) > sign_ignore_dx * dx
    sign_errors = (np.sign(values[tested]) != np.sign(exact[tested])) if finite else np.ones(int(np.sum(tested)), dtype=bool)
    n_sign = int(np.count_nonzero(tested))
    sign_fraction = float(np.mean(sign_errors)) if n_sign else 0.0
    mae = float(np.mean(error))
    max_error = float(np.max(error))
    passed = bool(finite and mae <= max_mae_dx * dx
                  and max_error <= max_error_dx * dx
                  and sign_fraction <= max_sign_fraction)
    return {
        'schema_version': 2, 'reference': 'analytic axis-aligned cube',
        'passed': passed, 'finite': finite, 'samples': int(values.size),
        'mae': mae, 'max_error': max_error,
        'rmse': float(np.sqrt(np.mean(error * error))),
        'sign_error_fraction_away_from_surface': sign_fraction,
        'sign_tested_samples': n_sign,
        'tolerances': {'max_mae_dx': max_mae_dx, 'max_error_dx': max_error_dx,
                       'max_sign_fraction': max_sign_fraction, 'sign_ignore_dx': sign_ignore_dx},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', required=True, type=Path)
    p.add_argument('--half-extent', type=float, default=0.55)
    p.add_argument('--report', type=Path)
    p.add_argument('--max-mae-dx', type=float, default=0.005,
                   help='maximum mean absolute error relative to grid spacing')
    p.add_argument('--max-error-dx', type=float, default=0.02,
                   help='maximum pointwise absolute error relative to grid spacing')
    p.add_argument('--max-sign-fraction', type=float, default=0.0)
    p.add_argument('--sign-ignore-dx', type=float, default=0.1)
    a = p.parse_args()
    for key in ('half_extent', 'max_mae_dx', 'max_error_dx', 'max_sign_fraction', 'sign_ignore_dx'):
        v = getattr(a, key)
        if not math.isfinite(v) or (v <= 0 if key == 'half_extent' else v < 0):
            p.error(f'--{key.replace("_", "-")} must be finite and positive/nonnegative')
    if a.max_sign_fraction > 1:
        p.error('--max-sign-fraction must be <= 1')
    try:
        report = evaluate(a.sdf, a.half_extent, a.max_mae_dx,
                          a.max_error_dx, a.max_sign_fraction, a.sign_ignore_dx)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if a.report:
        a.report.parent.mkdir(exist_ok=True, parents=True)
        a.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
