#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct Vec3 {
    double x=0, y=0, z=0;
};
struct Triangle { std::uint32_t a=0, b=0, c=0; };
struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> faces;
};
struct Grid {
    Vec3 origin{};
    double dx=0.1;
    int nx=0, ny=0, nz=0;
    std::size_t size() const;
    std::size_t index(int i,int j,int k) const;
    Vec3 point(int i,int j,int k) const;
};
struct SDFVolume {
    Grid grid;
    std::vector<float> phi;
};

Mesh read_obj(const std::string& path);
Grid read_grid_config(const std::string& path);
void write_sdf_ascii(const std::string& path, const SDFVolume& volume);
void write_report_json(const std::string& path, const std::string& implementation,
                       const std::string& status, const std::vector<double>& times,
                       const Mesh& mesh, const Grid& grid, const std::string& detail);

SDFVolume compute_sdfgen_style(const Mesh& mesh, const Grid& grid, int exact_band);
