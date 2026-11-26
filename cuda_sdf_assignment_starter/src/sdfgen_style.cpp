#include "sdf_common.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// Independent, pedagogical implementation of the SDFGen-style algorithm.
// Algorithmic reference: https://github.com/christopherbatty/SDFGen
// 1) exact triangle distances in a narrow band;
// 2) two sets of 8 ordered candidate-triangle sweeps;
// 3) parity from x-direction triangle intersections.
// This is NOT a bit-for-bit copy of upstream SDFGen. See README and docs.
namespace {
Vec3 operator+(Vec3 a,Vec3 b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(Vec3 a,Vec3 b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator*(Vec3 a,double s) {return {a.x*s,a.y*s,a.z*s};}
double dot(Vec3 a,Vec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
double norm2(Vec3 a) {return dot(a,a);}
double point_segment_dist2(Vec3 p,Vec3 a,Vec3 b) {
    Vec3 ab=b-a;
    const double len2=norm2(ab);
    if (len2<=1e-30) return norm2(p-a);
    double t=std::clamp(dot(p-a,ab)/len2,0.0,1.0);
    return norm2(p-(a+ab*t));
}
double point_triangle_dist2(Vec3 p,Vec3 a,Vec3 b,Vec3 c) {
    // Orthogonal projection to triangle plane if barycentric coordinates inside,
    // otherwise nearest of the three edges. Stable for degenerate triangles.
    Vec3 ab=b-a, ac=c-a, ap=p-a;
    const double d00=dot(ab,ab), d01=dot(ab,ac), d11=dot(ac,ac);
    const double den=d00*d11-d01*d01;
    if (den>1e-28*(d00*d11+1.0)) {
        const double d20=dot(ap,ab), d21=dot(ap,ac);
        const double v=(d11*d20-d01*d21)/den;
        const double w=(d00*d21-d01*d20)/den;
        if (v>=0 && w>=0 && v+w<=1) return norm2(p-(a+ab*v+ac*w));
    }
    return std::min({point_segment_dist2(p,a,b),point_segment_dist2(p,b,c),point_segment_dist2(p,c,a)});
}
int orientation(double x1,double y1,double x2,double y2,double& twice_area) {
    // Reproduce SDFGen's deterministic symbolic tie-breaking for projection boundaries.
    twice_area=y1*x2-x1*y2;
    if(twice_area>0) return 1;
    if(twice_area<0) return -1;
    if(y2>y1) return 1;
    if(y2<y1) return -1;
    if(x1>x2) return 1;
    if(x1<x2) return -1;
    return 0;
}
bool point_in_triangle_2d(double x0,double y0, double x1,double y1,
                          double x2,double y2,double x3,double y3,
                          double& a,double& b,double& c) {
    x1-=x0; y1-=y0; x2-=x0; y2-=y0; x3-=x0; y3-=y0;
    const int sa=orientation(x2,y2,x3,y3,a);
    if(sa==0) return false;
    const int sb=orientation(x3,y3,x1,y1,b);
    if(sb!=sa) return false;
    const int sc=orientation(x1,y1,x2,y2,c);
    if(sc!=sa) return false;
    const double sum=a+b+c;
    if(std::abs(sum)<1e-30) return false;
    a/=sum; b/=sum; c/=sum;
    return true;
}
int lower_clamped(double x,int band,int max_ind) {
    return static_cast<int>(std::clamp(std::floor(x)-band,0.0,static_cast<double>(max_ind)));
}
int upper_clamped(double x,int band,int max_ind) {
    return static_cast<int>(std::clamp(std::ceil(x)+band,0.0,static_cast<double>(max_ind)));
}
}

SDFVolume compute_sdfgen_style(const Mesh& mesh,const Grid& g,int exact_band) {
    if (exact_band<0 || exact_band>32) throw std::runtime_error("exact_band out of supported range [0,32]");
    const std::size_t n=g.size();
    // A finite grid-diagonal cap is not a valid bound if the mesh lies outside
    // the grid. Initialize to infinity and fill any unsourced nodes below.
    SDFVolume vol{g,std::vector<float>(n,std::numeric_limits<float>::infinity())};
    auto& phi=vol.phi;
    std::vector<std::int32_t> closest(n,-1);
    std::vector<std::int32_t> intersection(n,0);
    for(std::size_t tid=0;tid<mesh.faces.size();++tid) {
        const Triangle t=mesh.faces[tid];
        Vec3 a=mesh.vertices[t.a],b=mesh.vertices[t.b],c=mesh.vertices[t.c];
        const double ai=(a.x-g.origin.x)/g.dx,bi=(b.x-g.origin.x)/g.dx,ci=(c.x-g.origin.x)/g.dx;
        const double aj=(a.y-g.origin.y)/g.dx,bj=(b.y-g.origin.y)/g.dx,cj=(c.y-g.origin.y)/g.dx;
        const double ak=(a.z-g.origin.z)/g.dx,bk=(b.z-g.origin.z)/g.dx,ck=(c.z-g.origin.z)/g.dx;
        const int imin=lower_clamped(std::min({ai,bi,ci}),exact_band,g.nx-1);
        const int imax=upper_clamped(std::max({ai,bi,ci}),exact_band,g.nx-1);
        const int jmin=lower_clamped(std::min({aj,bj,cj}),exact_band,g.ny-1);
        const int jmax=upper_clamped(std::max({aj,bj,cj}),exact_band,g.ny-1);
        const int kmin=lower_clamped(std::min({ak,bk,ck}),exact_band,g.nz-1);
        const int kmax=upper_clamped(std::max({ak,bk,ck}),exact_band,g.nz-1);
        for(int k=kmin;k<=kmax;++k)
            for(int j=jmin;j<=jmax;++j)
                for(int i=imin;i<=imax;++i) {
                    const auto idx=g.index(i,j,k);
                    const float d=static_cast<float>(std::sqrt(point_triangle_dist2(g.point(i,j,k),a,b,c)));
                    if(d<phi[idx]) { phi[idx]=d;closest[idx]=static_cast<std::int32_t>(tid); }
                }
        const int y0=std::clamp(static_cast<int>(std::ceil(std::min({aj,bj,cj}))),0,g.ny-1);
        const int y1=std::clamp(static_cast<int>(std::floor(std::max({aj,bj,cj}))),0,g.ny-1);
        const int z0=std::clamp(static_cast<int>(std::ceil(std::min({ak,bk,ck}))),0,g.nz-1);
        const int z1=std::clamp(static_cast<int>(std::floor(std::max({ak,bk,ck}))),0,g.nz-1);
        for(int k=z0;k<=z1;++k) for(int j=y0;j<=y1;++j) {
            double w0,w1,w2;
            if(!point_in_triangle_2d(j,k,aj,ak,bj,bk,cj,ck,w0,w1,w2)) continue;
            const double fi=w0*ai+w1*bi+w2*ci;
            // Clamp in floating point before conversion: a far-away mesh can
            // have an intersection coordinate outside the int range.
            if(fi<0) ++intersection[g.index(0,j,k)];
            else if(fi<g.nx) ++intersection[g.index(static_cast<int>(std::ceil(fi)),j,k)];
        }
    }
    // Propagate triangle IDs, and *re-evaluate exact triangle distances* at each candidate.
    // In-place ordered Gauss-Seidel sweeps intentionally preserve SDFGen's sequential dependency.
    const int dir[8][3]={{+1,+1,+1},{-1,-1,-1},{+1,+1,-1},{-1,-1,+1},
                         {+1,-1,+1},{-1,+1,-1},{+1,-1,-1},{-1,+1,+1}};
    for(int pass=0;pass<2;++pass) for(const auto& d:dir) {
        const int di=d[0],dj=d[1],dk=d[2];
        const int istart=di>0?1:g.nx-2, iend=di>0?g.nx:-1;
        const int jstart=dj>0?1:g.ny-2, jend=dj>0?g.ny:-1;
        const int kstart=dk>0?1:g.nz-2, kend=dk>0?g.nz:-1;
        for(int k=kstart;k!=kend;k+=dk)
            for(int j=jstart;j!=jend;j+=dj)
                for(int i=istart;i!=iend;i+=di) {
                    const std::size_t target=g.index(i,j,k);
                    const Vec3 p=g.point(i,j,k);
                    auto check = [&](int ni,int nj,int nk) {
                        const auto source=g.index(ni,nj,nk);
                        const auto tri_id=closest[source];
                        if(tri_id<0) return;
                        const Triangle t=mesh.faces[static_cast<std::size_t>(tri_id)];
                        const float dd=static_cast<float>(std::sqrt(point_triangle_dist2(
                            p,mesh.vertices[t.a],mesh.vertices[t.b],mesh.vertices[t.c])));
                        if(dd<phi[target]) {phi[target]=dd;closest[target]=tri_id;}
                    };
                    check(i-di,j,k);
                    check(i,j-dj,k);
                    check(i-di,j-dj,k);
                    check(i,j,k-dk);
                    check(i-di,j,k-dk);
                    check(i,j-dj,k-dk);
                    check(i-di,j-dj,k-dk);
                }
    }
    // The propagation loops intentionally do not sweep every boundary node.
    // If no triangle was ever assigned, evaluate that node exactly rather than
    // exporting a cap value or +infinity. Most normal cases need no fallback.
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j)
        for(int i=0;i<g.nx;++i) {
            const auto idx=g.index(i,j,k);
            if(closest[idx]>=0) continue;
            const Vec3 p=g.point(i,j,k);
            double best=std::numeric_limits<double>::infinity();
            for(const Triangle& t:mesh.faces)
                best=std::min(best,point_triangle_dist2(p,mesh.vertices[t.a],
                                                         mesh.vertices[t.b],mesh.vertices[t.c]));
            phi[idx]=static_cast<float>(std::sqrt(best));
        }
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) {
        int total=0;
        for(int i=0;i<g.nx;++i) {
            const auto idx=g.index(i,j,k);
            total+=intersection[idx];
            if(total%2!=0) phi[idx]=-phi[idx];
        }
    }
    return vol;
}
