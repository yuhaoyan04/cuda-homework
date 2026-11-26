#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Mandatory task: STATIC one-frame SDF. Optional dynamic update has a separate bonus spec.
// This is the ONLY function students should implement/modify in the starter.
// All pointers are DEVICE pointers. Device memory is allocated and owned by the teacher harness.
// The vertices are XYZXYZ... float32, triangle indices are ABCABC... uint32.
// Grid values at (i,j,k) are sampled at origin + dx * (i,j,k).
// Output linear layout: i + nx*(j + ny*k); negative inside, positive outside.
// ALL GPU work must be ordered on `stream`. If internal streams are used,
// synchronize with `stream` using events before returning.
// Do NOT synchronize the whole device unnecessarily or write benchmark files here.
struct StudentGrid {
    float ox, oy, oz, dx;
    int nx, ny, nz;
};

cudaError_t student_sdf_cuda(
    const float* d_vertices, int num_vertices,
    const std::uint32_t* d_triangles, int num_triangles,
    StudentGrid grid, float* d_sdf,
    cudaStream_t stream);
