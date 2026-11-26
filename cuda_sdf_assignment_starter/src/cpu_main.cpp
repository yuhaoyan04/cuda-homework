#include "sdf_common.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::cerr<<"Usage: sdfbench_cpu --mesh M.obj --grid G.grid --out out.sdf --report result.json [--warmup 1] [--repeat 3] [--exact-band 1]\n";
}
int main(int argc,char**argv) {
    try {
        std::string mesh_path,grid_path,out_path,report_path;
        int warmup=1,repeat=3,band=1;
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
            else if(a=="--exact-band") band=std::stoi(value);
            else {std::cerr<<"Unknown argument: "<<a<<"\n";return 2;}
        }
        if(mesh_path.empty() || grid_path.empty() || out_path.empty() || report_path.empty() ||
           warmup<0 || repeat<1 || repeat>100 || warmup>20) {usage();return 2;}
        const Mesh m=read_obj(mesh_path);
        const Grid g=read_grid_config(grid_path);
        SDFVolume result;
        std::vector<double> times;
        for(int i=0;i<warmup+repeat;++i) {
            const auto t0=std::chrono::steady_clock::now();
            SDFVolume next=compute_sdfgen_style(m,g,band);
            const auto t1=std::chrono::steady_clock::now();
            if(i>=warmup) times.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
            result=std::move(next);
        }
        write_sdf_ascii(out_path,result);
        write_report_json(report_path,"cpu_sdfgen_style","PASS",times,m,g,"CPU wall clock, excludes OBJ reading / output file writing");
        std::cout<<"PASS CPU: vertices="<<m.vertices.size()<<", triangles="<<m.faces.size()<<", grid="<<g.nx<<"x"<<g.ny<<"x"<<g.nz<<"\n";
        std::cout<<"Wrote "<<out_path<<" and "<<report_path<<"\n";
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
