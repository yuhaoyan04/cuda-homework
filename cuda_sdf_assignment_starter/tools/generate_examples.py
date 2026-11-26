#!/usr/bin/env python3
"""Generate a small reproducible closed OBJ suite without external Python dependencies."""
import json
import math
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / 'datasets' / 'obj'
CFG = ROOT / 'datasets' / 'configs'


def write_obj(path, vertices, faces):
    with path.open('w', encoding='utf-8') as f:
        f.write('# CUDA SDF teaching mesh: closed, consistently oriented triangles\n')
        for x, y, z in vertices:
            f.write(f'v {x:.10g} {y:.10g} {z:.10g}\n')
        for a, b, c in faces:
            f.write(f'f {a+1} {b+1} {c+1}\n')


def cube(h=.55):
    v=[(-h,-h,-h),(h,-h,-h),(h,h,-h),(-h,h,-h),
       (-h,-h,h),(h,-h,h),(h,h,h),(-h,h,h)]
    quad=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    faces=[]
    for a,b,c,d in quad:
        faces += [(a,b,c),(a,c,d)]
    return v,faces


def icosphere(subdiv=1, radius=.65):
    t=(1+math.sqrt(5))/2
    verts=[(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),
           (0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    faces=[(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),
           (5,11,4),(11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),
           (3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),
           (8,6,7),(9,8,1)]
    def norm(v):
        n=math.sqrt(sum(x*x for x in v))
        return tuple(x/n for x in v)
    verts=[norm(v) for v in verts]
    for _ in range(subdiv):
        new=[]; cache={}
        def middle(a,b):
            key=tuple(sorted((a,b)))
            if key not in cache:
                va,vb=verts[a],verts[b]
                cache[key]=len(verts)
                verts.append(norm(tuple((va[i]+vb[i])*.5 for i in range(3))))
            return cache[key]
        for a,b,c in faces:
            ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
            new += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces=new
    return [tuple(radius*x for x in v) for v in verts],faces


def disconnected_cubes(half_extent=.26, separation=.61):
    """Two separate outward-oriented watertight boxes, both within the grid."""
    verts, faces = [], []
    for cx in (-separation, separation):
        v, f = cube(half_extent)
        base = len(verts)
        verts.extend((x + cx, y, z) for x, y, z in v)
        faces.extend((a + base, b + base, c + base) for a, b, c in f)
    return verts, faces


def torus(n_major=32, n_minor=16, major_radius=.58, minor_radius=.19):
    """Closed triangulated genus-1 surface with a central hole."""
    verts, faces = [], []
    for i in range(n_major):
        u = 2*math.pi*i/n_major
        for j in range(n_minor):
            v = 2*math.pi*j/n_minor
            r = major_radius + minor_radius*math.cos(v)
            verts.append((r*math.cos(u), r*math.sin(u), minor_radius*math.sin(v)))
    for i in range(n_major):
        for j in range(n_minor):
            a = i*n_minor+j
            b = ((i+1)%n_major)*n_minor+j
            c = ((i+1)%n_major)*n_minor+((j+1)%n_minor)
            d = i*n_minor+((j+1)%n_minor)
            faces.extend(((a,b,c), (a,c,d)))
    return verts, faces


def write_grid(path,n,origin=-1.,extent=2.):
    path.write_text(f'# nx ny nz / origin / spacing\n{n} {n} {n}\n{origin} {origin} {origin}\n{extent/(n-1):.17g}\n',encoding='utf-8')

# Grid reader accepts only numeric records; no comments. Use numeric-only config.
def write_numeric_grid(path,n,origin=-1.,extent=2.):
    path.write_text(f'{n} {n} {n}\n{origin} {origin} {origin}\n{extent/(n-1):.17g}\n',encoding='utf-8')


def main():
    OBJ.mkdir(parents=True,exist_ok=True)
    CFG.mkdir(parents=True,exist_ok=True)
    cube_v,cube_f=cube()
    write_obj(OBJ/'cube.obj',cube_v,cube_f)
    v,f=disconnected_cubes()
    write_obj(OBJ/'two_cubes.obj',v,f)
    v,f=torus()
    write_obj(OBJ/'torus.obj',v,f)
    for level in (1,2,3):
        v,f=icosphere(level)
        write_obj(OBJ/f'icosphere_L{level}.obj',v,f)
    v,f=icosphere(2)
    for frame in range(3):
        # Fixed face topology, only vertex positions change. Controlled transform
        # (positive determinant) keeps watertightness/orientation.
        phase=frame*0.33
        warped=[]
        for x,y,z in v:
            a=phase * (0.3+0.7*(z+.65)/1.3)
            ca,sa=math.cos(a),math.sin(a)
            xx,yy=ca*x-sa*y,sa*x+ca*y
            sx=1+0.14*math.sin(phase)
            sz=1-0.11*math.sin(phase)
            warped.append((sx*xx,yy,sz*z))
        write_obj(OBJ/f'deform_{frame:03d}.obj',warped,f)
    for n in (16,32,64,128):
        write_numeric_grid(CFG/f'grid_{n}.grid',n)
    manifest={
        'schema_version':1,
        'description':'All OBJ faces are triangles. Each sample is a GRID NODE origin + dx*(i,j,k).',
        'cases':[
            {'id':'smoke_cube_16','mesh':'obj/cube.obj','grid':'configs/grid_16.grid','class':'smoke'},
            {'id':'cube_32','mesh':'obj/cube.obj','grid':'configs/grid_32.grid','class':'public'},
            {'id':'sphere_32','mesh':'obj/icosphere_L1.obj','grid':'configs/grid_32.grid','class':'public'},
            {'id':'sphere_64','mesh':'obj/icosphere_L2.obj','grid':'configs/grid_64.grid','class':'public'},
            {'id':'two_cubes_64','mesh':'obj/two_cubes.obj','grid':'configs/grid_64.grid','class':'public'},
            {'id':'torus_64','mesh':'obj/torus.obj','grid':'configs/grid_64.grid','class':'public'},
            {'id':'deform_000_64','mesh':'obj/deform_000.obj','grid':'configs/grid_64.grid','class':'dynamic'},
            {'id':'deform_001_64','mesh':'obj/deform_001.obj','grid':'configs/grid_64.grid','class':'dynamic'},
            {'id':'deform_002_64','mesh':'obj/deform_002.obj','grid':'configs/grid_64.grid','class':'dynamic'},
            {'id':'sphere_128','mesh':'obj/icosphere_L3.obj','grid':'configs/grid_128.grid','class':'stress_optional'}
        ],
        'assignment_note':'Static SDF is required. Dynamic frames are optional bonus only. The public validation gates are provisional; final grading uses a separate trusted process.'
    }
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    paths = sorted([*OBJ.glob('*.obj'), *CFG.glob('*.grid')])
    checksums = ''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
                        f'{path.relative_to(ROOT / "datasets").as_posix()}\n' for path in paths)
    (ROOT/'datasets'/'SHA256SUMS').write_text(checksums, encoding='utf-8')
    print('Generated',len(manifest['cases']),'test cases and',len(paths),'file checksums')

if __name__=='__main__':
    main()
