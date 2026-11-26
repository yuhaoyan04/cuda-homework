#include "student_sdf.cuh"

#include <cmath>
#include <limits>

namespace {

__device__ __forceinline__ float sqr(float x) { return x * x; }

// Ericson's closest-point regions, including triangle edges and vertices.
__device__ float point_triangle_distance2(
    float px, float py, float pz,
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz) {
    const float abx = bx-ax, aby = by-ay, abz = bz-az;
    const float acx = cx-ax, acy = cy-ay, acz = cz-az;
    const float apx = px-ax, apy = py-ay, apz = pz-az;
    const float d1 = abx*apx + aby*apy + abz*apz;
    const float d2 = acx*apx + acy*apy + acz*apz;
    float qx, qy, qz;
    if (d1 <= 0.f && d2 <= 0.f) { qx=ax; qy=ay; qz=az; }
    else {
        const float bpx=px-bx, bpy=py-by, bpz=pz-bz;
        const float d3=abx*bpx+aby*bpy+abz*bpz;
        const float d4=acx*bpx+acy*bpy+acz*bpz;
        if (d3 >= 0.f && d4 <= d3) { qx=bx; qy=by; qz=bz; }
        else {
            const float vc=d1*d4-d3*d2;
            if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
                const float v=d1/(d1-d3); qx=ax+v*abx; qy=ay+v*aby; qz=az+v*abz;
            } else {
                const float cpx=px-cx, cpy=py-cy, cpz=pz-cz;
                const float d5=abx*cpx+aby*cpy+abz*cpz;
                const float d6=acx*cpx+acy*cpy+acz*cpz;
                if (d6 >= 0.f && d5 <= d6) { qx=cx; qy=cy; qz=cz; }
                else {
                    const float vb=d5*d2-d1*d6;
                    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
                        const float w=d2/(d2-d6); qx=ax+w*acx; qy=ay+w*acy; qz=az+w*acz;
                    } else {
                        const float va=d3*d6-d5*d4;
                        if (va <= 0.f && (d4-d3) >= 0.f && (d5-d6) >= 0.f) {
                            const float w=(d4-d3)/((d4-d3)+(d5-d6));
                            qx=bx+w*(cx-bx); qy=by+w*(cy-by); qz=bz+w*(cz-bz);
                        } else {
                            const float denom=1.f/(va+vb+vc), v=vb*denom, w=vc*denom;
                            qx=ax+abx*v+acx*w; qy=ay+aby*v+acy*w; qz=az+abz*v+acz*w;
                        }
                    }
                }
            }
        }
    }
    return sqr(px-qx)+sqr(py-qy)+sqr(pz-qz);
}

// Count +X ray crossings using projected barycentrics in the YZ plane.
// A tiny deterministic ray offset avoids most shared-edge coincidences.
__device__ bool ray_hit_x(float py, float pz, float ax, float ay, float az,
                           float bx, float by, float bz, float cx, float cy, float cz,
                           float& hitx) {
    const float v0y=cy-ay, v0z=cz-az, v1y=by-ay, v1z=bz-az;
    const float v2y=py-ay, v2z=pz-az;
    const float den=v0y*v1z-v1y*v0z;
    if (fabsf(den) < 1.e-20f) return false;
    const float u=(v2y*v1z-v1y*v2z)/den;
    const float v=(v0y*v2z-v2y*v0z)/den;
    if (u < 0.f || v < 0.f || u+v > 1.f) return false;
    hitx=ax+u*(cx-ax)+v*(bx-ax);
    return true;
}

__global__ void sdf_bruteforce(const float* __restrict__ vertices,
                               const std::uint32_t* __restrict__ triangles,
                               int ntri, StudentGrid g, float* __restrict__ out,
                               int count) {
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= count) return;
    const int i=idx%g.nx, j=(idx/g.nx)%g.ny, k=idx/(g.nx*g.ny);
    const float px=g.ox+i*g.dx, py=g.oy+j*g.dx, pz=g.oz+k*g.dx;
    // Slightly perturb the parity ray in its transverse coordinates.
    const float ry=py+g.dx*1.234567e-5f, rz=pz+g.dx*2.345679e-5f;
    float best= INFINITY;
    unsigned crossings=0;
    for (int t=0;t<ntri;++t) {
        const unsigned ia=triangles[3*t]*3u, ib=triangles[3*t+1]*3u, ic=triangles[3*t+2]*3u;
        const float ax=vertices[ia], ay=vertices[ia+1], az=vertices[ia+2];
        const float bx=vertices[ib], by=vertices[ib+1], bz=vertices[ib+2];
        const float cx=vertices[ic], cy=vertices[ic+1], cz=vertices[ic+2];
        // Optimization 1: conservative point-to-AABB lower bound avoids the
        // more expensive closest-point region calculation when it cannot win.
#ifndef SDF_DISABLE_AABB_PRUNE
        const float dx=fmaxf(fminf(fminf(ax,bx),cx)-px, 0.f);
        const float dy=fmaxf(fminf(fminf(ay,by),cy)-py, 0.f);
        const float dz=fmaxf(fminf(fminf(az,bz),cz)-pz, 0.f);
        const float lx=fmaxf(fmaxf(px-fmaxf(fmaxf(ax,bx),cx),0.f),dx);
        const float ly=fmaxf(fmaxf(py-fmaxf(fmaxf(ay,by),cy),0.f),dy);
        const float lz=fmaxf(fmaxf(pz-fmaxf(fmaxf(az,bz),cz),0.f),dz);
        if (lx*lx+ly*ly+lz*lz < best) {
            const float d2=point_triangle_distance2(px,py,pz,ax,ay,az,bx,by,bz,cx,cy,cz);
            best=fminf(best,d2);
        }
#else
        const float d2=point_triangle_distance2(px,py,pz,ax,ay,az,bx,by,bz,cx,cy,cz);
        best=fminf(best,d2);
#endif
        // Optimization 2: reject ray tests outside the triangle's projected
        // YZ bounds before barycentric evaluation. This cannot remove a hit.
#ifndef SDF_DISABLE_RAY_AABB
        if (ry < fminf(fminf(ay,by),cy) || ry > fmaxf(fmaxf(ay,by),cy) ||
            rz < fminf(fminf(az,bz),cz) || rz > fmaxf(fmaxf(az,bz),cz)) continue;
#endif
        float hx;
        if (ray_hit_x(ry,rz,ax,ay,az,bx,by,bz,cx,cy,cz,hx) && hx > px) ++crossings;
    }
    const float d=sqrtf(best);
    out[idx]=(crossings&1u) ? -d : d;
}

} // namespace

cudaError_t student_sdf_cuda(const float* d_vertices, int num_vertices,
                             const std::uint32_t* d_triangles, int num_triangles,
                             StudentGrid grid, float* d_sdf, cudaStream_t stream) {
    if (!d_vertices || !d_triangles || !d_sdf || num_vertices < 3 || num_triangles < 1 ||
        grid.nx < 1 || grid.ny < 1 || grid.nz < 1 || !(grid.dx > 0.f)) return cudaErrorInvalidValue;
    const long long total=(long long)grid.nx*grid.ny*grid.nz;
    if (total > 2147483647LL) return cudaErrorInvalidValue;
    constexpr int threads=128;
    sdf_bruteforce<<<(unsigned)((total+threads-1)/threads),threads,0,stream>>>(
        d_vertices,d_triangles,num_triangles,grid,d_sdf,(int)total);
    return cudaGetLastError();
}
