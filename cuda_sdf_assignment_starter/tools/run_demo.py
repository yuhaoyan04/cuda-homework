#!/usr/bin/env python3
"""Run a static SDF case using trusted CPU/GPU drivers and verification tools.

Exit status: 0 PASS, 1 incorrect/failed algorithm, 2 configuration/input error,
3 GPU template NOT_IMPLEMENTED, 4 missing GPU executable.
This is a demonstration harness, not the final hidden-test grading system.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def call(argv):
    print('+', ' '.join(map(str, argv)), flush=True)
    return subprocess.run([str(v) for v in argv], check=False).returncode


def executable(build, name, config='Release'):
    # Single-configuration CMake, MSVC multi-configuration, and bin/Release variants.
    candidates = []
    for directory in (build, build / config, build / 'bin' / config, build / 'bin'):
        for suffix in ('', '.exe'):
            candidates.append(directory / (name + suffix))
    return next((p for p in candidates if p.is_file()), None)


def summary(path, case_id, status, **extra):
    record = {'schema_version': 2, 'case': case_id, 'status': status, **extra}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Result:', status, '| summary:', path)
    return record


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--case', default='smoke_cube_16')
    a.add_argument('--build', type=Path, default=ROOT / 'build')
    a.add_argument('--config', default='Release', help='for MSVC multi-config builds')
    a.add_argument('--cpu-exe', type=Path, help='Optional exact path to sdfbench_cpu.exe (Windows / custom build trees)')
    a.add_argument('--gpu-exe', type=Path, help='Optional exact path to sdfbench_gpu.exe')
    a.add_argument('--output', type=Path, default=ROOT / 'results')
    a.add_argument('--with-gpu', action='store_true')
    a.add_argument('--repeat', type=int, default=3)
    a.add_argument('--abs-tol-dx', type=float, default=0.1,
                   help='PROVISIONAL CPU comparison, not the final grading gate')
    x = a.parse_args()
    if not 1 <= x.repeat <= 100:
        a.error('--repeat must be in [1, 100]')
    manifest = json.loads((ROOT / 'datasets/manifest.json').read_text(encoding='utf-8'))
    matches = [case for case in manifest['cases'] if case['id'] == x.case]
    if not matches:
        a.error('Unknown case: ' + x.case)
    case = matches[0]
    mesh = ROOT / 'datasets' / case['mesh']
    grid = ROOT / 'datasets' / case['grid']
    dest = x.output.resolve() / case['id']
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = dest / 'summary.json'
    cpu = x.cpu_exe.resolve() if x.cpu_exe else executable(x.build.resolve(), 'sdfbench_cpu', x.config)
    gpu = x.gpu_exe.resolve() if x.gpu_exe else executable(x.build.resolve(), 'sdfbench_gpu', x.config)
    if cpu is not None and not cpu.is_file():
        cpu = None
    if gpu is not None and not gpu.is_file():
        gpu = None
    cpath, gpath = dest / 'cpu.sdf', dest / 'gpu.sdf'
    # Do not accidentally check outputs from a previous run.
    stale = [cpath, dest / 'cpu.json', dest / 'analytic_cube.json',
             gpath, dest / 'gpu.json', dest / 'comparison.json', summary_path]
    for path in stale:
        path.unlink(missing_ok=True)
    if cpu is None:
        print('ERROR: sdfbench_cpu executable not found.', file=sys.stderr)
        print('Expected one of:', file=sys.stderr)
        for directory in (x.build.resolve(), x.build.resolve()/x.config,
                          x.build.resolve()/'bin'/x.config, x.build.resolve()/'bin'):
            print('  ' + str(directory/'sdfbench_cpu.exe'), file=sys.stderr)
        print('Find the actual path with PowerShell: Get-ChildItem -Recurse -Filter sdfbench_cpu.exe', file=sys.stderr)
        print('Then pass --cpu-exe "FULL_PATH_TO_sdfbench_cpu.exe"', file=sys.stderr)
        summary(summary_path, case['id'], 'CPU_EXECUTABLE_MISSING')
        return 2
    rc = call([cpu, '--mesh', mesh, '--grid', grid, '--out', cpath,
               '--report', dest / 'cpu.json', '--warmup', 1, '--repeat', x.repeat,
               '--exact-band', 1])
    if rc != 0 or not cpath.is_file() or not (dest / 'cpu.json').is_file():
        summary(summary_path, case['id'], 'CPU_FAILED', exit_code=rc)
        return 1
    if case['id'].startswith(('smoke_cube', 'cube_')):
        rc = call([sys.executable, ROOT / 'tools/validate_cube.py', '--sdf', cpath,
                   '--report', dest / 'analytic_cube.json'])
        if rc != 0:
            summary(summary_path, case['id'], 'CPU_ANALYTIC_FAIL', exit_code=rc)
            return 1
    if not x.with_gpu:
        summary(summary_path, case['id'], 'CPU_PASS',
                note='CPU-only static SDF test; no GPU performance score')
        return 0
    if gpu is None:
        print('ERROR: sdfbench_gpu executable not found. Use --gpu-exe FULL_PATH if built elsewhere.', file=sys.stderr)
        summary(summary_path, case['id'], 'GPU_EXECUTABLE_MISSING')
        return 4
    rc = call([gpu, '--mesh', mesh, '--grid', grid, '--out', gpath,
               '--report', dest / 'gpu.json', '--warmup', 2, '--repeat', x.repeat])
    if rc == 3:
        summary(summary_path, case['id'], 'NOT_IMPLEMENTED',
                note='Student CUDA template is blank; no score assigned')
        return 3
    if rc != 0 or not gpath.is_file() or not (dest / 'gpu.json').is_file():
        summary(summary_path, case['id'], 'GPU_FAILED', exit_code=rc)
        return 1
    rc = call([sys.executable, ROOT / 'tools/compare_sdf.py',
               '--reference', cpath, '--candidate', gpath,
               '--report', dest / 'comparison.json', '--abs-tol-dx', x.abs_tol_dx])
    if rc != 0:
        summary(summary_path, case['id'], 'CORRECTNESS_FAIL', exit_code=rc,
                comparison_report=str(dest / 'comparison.json'))
        return 1
    verification = json.loads((dest / 'comparison.json').read_text(encoding='utf-8'))
    cpu_report = json.loads((dest / 'cpu.json').read_text(encoding='utf-8'))
    gpu_report = json.loads((dest / 'gpu.json').read_text(encoding='utf-8'))
    tc, tg = cpu_report.get('median_ms'), gpu_report.get('median_ms')
    summary(summary_path, case['id'], 'PASS',
            cpu_median_ms=tc, gpu_median_ms=tg,
            speedup_cpu_over_gpu=tc / tg if tc is not None and tg is not None and tg > 0 else None,
            gpu_perf_valid=bool(verification['passed']),
            comparison_report=str(dest / 'comparison.json'),
            note='Provisional static check. CPU wall time vs GPU event time exclude data transfers; not final ranking.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
