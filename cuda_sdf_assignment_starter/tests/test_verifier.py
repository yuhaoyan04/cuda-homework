#!/usr/bin/env python3
"""Regression tests for format, correctness gate, failure reporting."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.folder=Path(self.tmp.name)
        self.reference=self.folder/'ref.sdf'
        self.candidate=self.folder/'gpu.sdf'
        self.log=self.folder/'check.json'
        data='2 2 2\n0 0 0\n0.1\n'+'\n'.join(['0.1', '-0.2','0.3','-0.4','0.5','-0.6','0.7','-0.8'])+'\n'
        self.reference.write_text(data)
        self.candidate.write_text(data)
    def tearDown(self):
        self.tmp.cleanup()
    def compare(self):
        return subprocess.run([sys.executable,str(ROOT/'tools/compare_sdf.py'),'--reference',str(self.reference),
                               '--candidate',str(self.candidate),'--report',str(self.log)],
                              capture_output=True,text=True)
    def test_same_result_passes(self):
        r=self.compare()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertTrue(json.loads(self.log.read_text())['passed'])
    def test_nan_fails(self):
        text=self.candidate.read_text()
        self.candidate.write_text(text.replace('0.7','nan'))
        self.assertEqual(self.compare().returncode,1)
    def test_wrong_sign_fails(self):
        text=self.candidate.read_text()
        self.candidate.write_text(text.replace('-0.8','0.8'))
        self.assertEqual(self.compare().returncode,1)
    def test_wrong_grid_fails(self):
        text=self.candidate.read_text()
        self.candidate.write_text(text.replace('0 0 0','0.1 0 0'))
        self.assertEqual(self.compare().returncode,2)
    def test_truncated_file_fails(self):
        lines=self.candidate.read_text().splitlines()
        self.candidate.write_text('\n'.join(lines[:-1])+'\n')
        self.assertEqual(self.compare().returncode,2)



class TestNewGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.reference = self.folder / 'ref.sdf'
        self.candidate = self.folder / 'candidate.sdf'
        self.report = self.folder / 'report.json'
        # 16^3 analytic cube samples, independent of the CPU baseline.
        import numpy as np
        n = 16
        dx = 2/(n-1)
        v = []
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    q = np.abs(np.array([-1+i*dx, -1+j*dx, -1+k*dx])) - .55
                    v.append(float(np.linalg.norm(np.maximum(q, 0)) + min(max(q), 0)))
        content = f'{n} {n} {n}\n-1 -1 -1\n{dx:.17g}\n' + ''.join(f'{x:.9g}\n' for x in v)
        self.reference.write_text(content)
        self.candidate.write_text(content)

    def tearDown(self):
        self.tmp.cleanup()

    def analytic(self):
        return subprocess.run([sys.executable, str(ROOT/'tools/validate_cube.py'),
                               '--sdf', str(self.candidate), '--report', str(self.report)],
                              capture_output=True, text=True)

    def test_analytic_true_cube_passes(self):
        r = self.analytic()
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertTrue(json.loads(self.report.read_text())['passed'])

    def test_analytic_all_zeros_fails(self):
        header = '\n'.join(self.candidate.read_text().splitlines()[:3]) + '\n'
        self.candidate.write_text(header + '0\n'*4096)
        r = self.analytic()
        self.assertEqual(r.returncode, 1, r.stdout+r.stderr)
        self.assertFalse(json.loads(self.report.read_text())['passed'])

    def test_analytic_nan_fails(self):
        s = self.candidate.read_text()
        self.candidate.write_text(s.replace('\n0.55\n', '\nnan\n', 1))
        # If 0.55 doesn't appear, forcibly corrupt the first sample value.
        if self.candidate.read_text() == s:
            lines=s.splitlines(); lines[3]='nan'
            self.candidate.write_text('\n'.join(lines)+'\n')
        self.assertEqual(self.analytic().returncode, 1)

    def test_provisional_comparison_rejects_dx_bias(self):
        lines = self.reference.read_text().splitlines()
        dx = float(lines[2])
        # Old default accepted a systematic 0.8*dx magnitude error.
        wrong = [str((1 if float(v) >= 0 else -1) * (abs(float(v)) + .8*dx)) for v in lines[3:]]
        self.candidate.write_text('\n'.join(lines[:3]+wrong)+'\n')
        r = subprocess.run([sys.executable, str(ROOT/'tools/compare_sdf.py'),
                            '--reference', str(self.reference), '--candidate', str(self.candidate),
                            '--report', str(self.report)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout+r.stderr)
        self.assertFalse(json.loads(self.report.read_text())['passed'])

    def test_unsigned_reference_rejects_wrong_magnitudes(self):
        # Test independent numeric oracle directly on a tiny point-triangle example.
        sys.path.insert(0, str(ROOT/'tools'))
        from validate_unsigned_samples import exact_unsigned
        import numpy as np
        tri = np.array([[[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]]])
        pts = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
        ds = exact_unsigned(pts, tri)
        self.assertAlmostEqual(ds[0], 2.0, places=10)
        self.assertAlmostEqual(ds[1], 0.0, places=10)
        self.assertTrue(0.99 < ds[2] < 1.5)


class TestStaticRegression(unittest.TestCase):
    def test_windows_release_executable_discovery(self):
        spec=importlib.util.spec_from_file_location('sdf_run_demo', ROOT/'tools/run_demo.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            build=Path(td)
            exe=build/'Release'/'sdfbench_cpu.exe'
            exe.parent.mkdir()
            exe.write_bytes(b'placeholder')
            self.assertEqual(module.executable(build, 'sdfbench_cpu', 'Release'), exe)
            self.assertIsNone(module.executable(build, 'sdfbench_gpu', 'Release'))

    def test_public_topology_edges_are_consistent(self):
        # An oriented closed surface uses each undirected edge twice with
        # opposite directions, including the genus-1 and disconnected meshes.
        from collections import Counter
        for name in ('cube.obj', 'torus.obj', 'two_cubes.obj'):
            oriented=Counter()
            for line in (ROOT/'datasets/obj'/name).read_text().splitlines():
                if not line.startswith('f '):
                    continue
                a,b,c=map(int,line.split()[1:])
                for u,v in ((a,b),(b,c),(c,a)):
                    oriented[(u,v)]+=1
            for edge, n in oriented.items():
                self.assertEqual(n, 1, (name,edge,n))
                self.assertEqual(oriented[(edge[1],edge[0])], 1, (name,edge))

    def test_cpu_far_mesh_has_no_initial_distance_cap(self):
        spec = importlib.util.spec_from_file_location('sdf_run_demo', ROOT/'tools/run_demo.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        binary = module.executable(ROOT/'build', 'sdfbench_cpu', 'Release')
        if binary is None:
            self.skipTest('Build CPU benchmark in build/ first for this optional integration test')
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)
            lines=(ROOT/'datasets/obj/cube.obj').read_text().splitlines()
            moved=[]
            for line in lines:
                if line.startswith('v '):
                    _,x,y,z=line.split()
                    moved.append(f'v {100+float(x)} {y} {z}')
                else:
                    moved.append(line)
            (dest/'far.obj').write_text('\n'.join(moved)+'\n')
            (dest/'test.grid').write_text((ROOT/'datasets/configs/grid_16.grid').read_text())
            r=subprocess.run([str(binary),'--mesh',str(dest/'far.obj'),
                 '--grid',str(dest/'test.grid'),'--out',str(dest/'far.sdf'),
                 '--report',str(dest/'far.json'),'--warmup','0','--repeat','1'],
                 capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            values=np.loadtxt(dest/'far.sdf',skiprows=3)
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertGreater(float(values.min()), 95.)
            self.assertLess(float(values.max()), 105.)


if __name__ == "__main__":
    unittest.main()
