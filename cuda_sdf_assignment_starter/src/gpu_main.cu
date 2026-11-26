#include "sdf_common.hpp"
#include "student_sdf.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const auto err=(call); if(err!=cudaSuccess) \
    throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(err)); } while(0)

struct DeviceBuffers {
    float* vertices=nullptr;
    std::uint32_t* triangles=nullptr;
    float* output=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t start=nullptr,stop=nullptr;
    ~DeviceBuffers() {
        if(start) cudaEventDestroy(start);
        if(stop) cudaEventDestroy(stop);
        if(stream) cudaStreamDestroy(stream);
        if(output) cudaFree(output);
        if(triangles) cudaFree(triangles);
        if(vertices) cudaFree(vertices);
    }
};

static void usage() {
    std::cerr<<"Usage: sdfbench_gpu --mesh M.obj --grid G.grid --out out.sdf --report result.json [--warmup 2] [--repeat 10]\n";
}
int main(int argc,char**argv) {
    try {
        std::string mesh_path,grid_path,out_path,report_path;
        int warmup=2,repeat=10;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(i+1>=argc) {usage();return 2;}
            const std::string value=argv[++i];
            if(a=="--mesh") mesh_path=value;
            else if(a=="--grid") grid_path=value;
            else if(a=="--out") out_path=value;
            else if(a=="--report") report_path=value;
            else if(a=="--warmup") warmup=std::stoi(value);
            else if(a=="--repeat") repeat=std::stoi(value);
            else {std::cerr<<"Unknown argument: "<<a<<"\n";return 2;}
        }
        if(mesh_path.empty() || grid_path.empty() || out_path.empty() || report_path.empty() ||
           warmup<0 || repeat<1 || repeat>100 || warmup>20) {usage();return 2;}
        const Mesh mesh=read_obj(mesh_path);
        const Grid grid=read_grid_config(grid_path);
        std::vector<float> h_vertices; h_vertices.reserve(mesh.vertices.size()*3);
        for(const auto& v:mesh.vertices) {
            h_vertices.push_back(static_cast<float>(v.x));
            h_vertices.push_back(static_cast<float>(v.y));
            h_vertices.push_back(static_cast<float>(v.z));
        }
        std::vector<std::uint32_t> h_tris; h_tris.reserve(mesh.faces.size()*3);
        for(const auto& t:mesh.faces) {
            h_tris.push_back(t.a);h_tris.push_back(t.b);h_tris.push_back(t.c);
        }
        DeviceBuffers d;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.vertices),h_vertices.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.triangles),h_tris.size()*sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.output),grid.size()*sizeof(float)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream,cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&d.start));
        CUDA_CHECK(cudaEventCreate(&d.stop));
        CUDA_CHECK(cudaMemcpyAsync(d.vertices,h_vertices.data(),h_vertices.size()*sizeof(float),cudaMemcpyHostToDevice,d.stream));
        CUDA_CHECK(cudaMemcpyAsync(d.triangles,h_tris.data(),h_tris.size()*sizeof(std::uint32_t),cudaMemcpyHostToDevice,d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        const StudentGrid sg{static_cast<float>(grid.origin.x),static_cast<float>(grid.origin.y),
                              static_cast<float>(grid.origin.z),static_cast<float>(grid.dx),
                              grid.nx,grid.ny,grid.nz};
        std::vector<double> times;
        for(int i=0;i<warmup+repeat;++i) {
            // Sentinel fill is outside the compute interval. It exposes skipped writes.
            CUDA_CHECK(cudaMemsetAsync(d.output,0xff,grid.size()*sizeof(float),d.stream));
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
            CUDA_CHECK(cudaEventRecord(d.start,d.stream));
            const cudaError_t status=student_sdf_cuda(d.vertices,static_cast<int>(mesh.vertices.size()),
                                                       d.triangles,static_cast<int>(mesh.faces.size()),
                                                       sg,d.output,d.stream);
            if(status==cudaErrorNotSupported) {
                write_report_json(report_path,"gpu_student","NOT_IMPLEMENTED",{},mesh,grid,
                                  "Student CUDA method returns cudaErrorNotSupported; no GPU output was generated");
                std::cout<<"NOT_IMPLEMENTED: student/student_sdf.cu contains the blank template.\n";
                return 3;
            }
            CUDA_CHECK(status);
            CUDA_CHECK(cudaEventRecord(d.stop,d.stream));
            CUDA_CHECK(cudaEventSynchronize(d.stop));
            CUDA_CHECK(cudaGetLastError());
            float ms=0;
            CUDA_CHECK(cudaEventElapsedTime(&ms,d.start,d.stop));
            if(i>=warmup) times.push_back(static_cast<double>(ms));
        }
        std::vector<float> output(grid.size());
        CUDA_CHECK(cudaMemcpy(output.data(),d.output,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
        write_sdf_ascii(out_path,SDFVolume{grid,std::move(output)});
        write_report_json(report_path,"gpu_student","EXECUTED_UNVERIFIED",times,mesh,grid,
                          "GPU CUDA event time excludes initial H2D copies and output D2H copy; run verifier separately");
        std::cout<<"EXECUTED_UNVERIFIED: wrote GPU output and timing (correctness NOT yet checked).\n";
    } catch(const std::exception& e) {
        std::cerr<<"GPU ERROR: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
