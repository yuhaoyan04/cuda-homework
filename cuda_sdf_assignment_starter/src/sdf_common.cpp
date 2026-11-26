#include "sdf_common.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

std::size_t Grid::size() const {
    if (nx <= 0 || ny <= 0 || nz <= 0 || dx <= 0 || !std::isfinite(dx))
        throw std::runtime_error("Invalid grid dimensions or spacing");
    const auto x=static_cast<std::size_t>(nx), y=static_cast<std::size_t>(ny), z=static_cast<std::size_t>(nz);
    if (x > std::numeric_limits<std::size_t>::max() / y || x*y > std::numeric_limits<std::size_t>::max() / z)
        throw std::runtime_error("Grid size overflow");
    return x*y*z;
}
std::size_t Grid::index(int i,int j,int k) const {
    return (static_cast<std::size_t>(k)*static_cast<std::size_t>(ny)+static_cast<std::size_t>(j))*static_cast<std::size_t>(nx)+static_cast<std::size_t>(i);
}
Vec3 Grid::point(int i,int j,int k) const {
    return {origin.x+i*dx, origin.y+j*dx, origin.z+k*dx};
}
static std::uint32_t parse_index(const std::string& token, std::size_t num_vertices) {
    const std::size_t sep=token.find('/');
    const auto v=std::stoll(token.substr(0,sep));
    if (v <= 0 || static_cast<std::uint64_t>(v) > num_vertices)
        throw std::runtime_error("Only positive 1-based OBJ indices are supported");
    return static_cast<std::uint32_t>(v-1);
}
Mesh read_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open OBJ file: "+path);
    Mesh mesh;
    std::string line;
    std::size_t line_no=0;
    while (std::getline(f,line)) {
        ++line_no;
        std::istringstream in(line);
        std::string tag;
        if (!(in>>tag) || tag[0]=='#') continue;
        if (tag=="v") {
            Vec3 p;
            if (!(in>>p.x>>p.y>>p.z) || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                throw std::runtime_error("Invalid OBJ vertex at line "+std::to_string(line_no));
            mesh.vertices.push_back(p);
        } else if (tag=="f") {
            std::string a,b,c,extra;
            if (!(in>>a>>b>>c) || (in>>extra))
                throw std::runtime_error("Only triangular OBJ faces are accepted (line "+std::to_string(line_no)+")");
            mesh.faces.push_back({parse_index(a,mesh.vertices.size()),parse_index(b,mesh.vertices.size()),parse_index(c,mesh.vertices.size())});
        }
    }
    if (mesh.vertices.empty() || mesh.faces.empty()) throw std::runtime_error("OBJ has no vertices or faces");
    return mesh;
}
Grid read_grid_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open grid config: "+path);
    // Tiny text configuration: NX NY NZ / ORIGIN_X ORIGIN_Y ORIGIN_Z / SPACING
    Grid g;
    if (!(f>>g.nx>>g.ny>>g.nz>>g.origin.x>>g.origin.y>>g.origin.z>>g.dx))
        throw std::runtime_error("Grid config expects: nx ny nz, origin (xyz), dx");
    g.size();
    return g;
}
void write_sdf_ascii(const std::string& path, const SDFVolume& volume) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write SDF output: "+path);
    const Grid& g=volume.grid;
    if (volume.phi.size()!=g.size()) throw std::runtime_error("Incorrect SDF output size");
    f<<g.nx<<" "<<g.ny<<" "<<g.nz<<"\n";
    f<<std::setprecision(17)<<g.origin.x<<" "<<g.origin.y<<" "<<g.origin.z<<"\n";
    f<<std::setprecision(17)<<g.dx<<"\n";
    f<<std::setprecision(9);
    for (float v:volume.phi) f<<v<<"\n";
    if (!f) throw std::runtime_error("Failed writing SDF output");
}
static std::string escape_json(const std::string& s) {
    std::string r;
    for(char c:s) {
        if (c=='\\') r+="\\\\";
        else if (c=='\"') r+="\\\"";
        else if (c=='\n') r+="\\n";
        else if (c=='\r') r+="\\r";
        else if (c=='\t') r+="\\t";
        else if (static_cast<unsigned char>(c)<32) r+=' ';
        else r+=c;
    }
    return r;
}
void write_report_json(const std::string& path, const std::string& implementation,
                       const std::string& status, const std::vector<double>& times,
                       const Mesh& mesh, const Grid& grid, const std::string& detail) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot create report "+path);
    std::vector<double> sorted=times;
    std::sort(sorted.begin(),sorted.end());
    const double median=sorted.empty()?-1.0:(sorted.size()%2? sorted[sorted.size()/2] :
                                    0.5*(sorted[sorted.size()/2-1]+sorted[sorted.size()/2]));
    f<<"{\n  \"schema_version\": 1,\n  \"implementation\": \""<<escape_json(implementation)<<"\",\n";
    f<<"  \"status\": \""<<escape_json(status)<<"\",\n  \"detail\": \""<<escape_json(detail)<<"\",\n";
    f<<"  \"vertices\": "<<mesh.vertices.size()<<",\n  \"triangles\": "<<mesh.faces.size()<<",\n";
    f<<"  \"resolution\": ["<<grid.nx<<", "<<grid.ny<<", "<<grid.nz<<"],\n";
    f<<"  \"median_ms\": ";
    if (median<0) f<<"null"; else f<<std::setprecision(12)<<median;
    f<<",\n  \"times_ms\": [";
    for (std::size_t i=0;i<times.size();++i) {
        if(i) f<<", ";
        f<<std::setprecision(12)<<times[i];
    }
    f<<"]\n}\n";
}
