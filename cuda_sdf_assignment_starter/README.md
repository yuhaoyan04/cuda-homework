# CUDA SDF 研究生大程序｜学生 Starter V1.0.2

**必做：静态三角网格 SDF 的 CUDA 实现、验证与优化（基础 100 分）。**  
**选做：固定拓扑动态 SDF 更新（最多 10 分加分；不做不影响基础满分资格）。**

先读 [PROJECT_SPEC.md](PROJECT_SPEC.md) 了解任务与评分；无图形学基础先看 [docs/GEOMETRY_PRIMER.md](docs/GEOMETRY_PRIMER.md)。公开案例见 [docs/DATASETS.md](docs/DATASETS.md)，实验报告见 [docs/EXPERIMENT_REPORT_TEMPLATE.md](docs/EXPERIMENT_REPORT_TEMPLATE.md)，选做动态说明见 [bonus/README.md](bonus/README.md)。

## 1. 环境与文件完整性

需要：Windows 11 / VS2022 或 Linux C++17 编译器，CMake 3.21+，Python 3、NumPy；GPU 编译还需支持 `sm_120` 的 CUDA Toolkit（推荐 12.8+）和 NVIDIA 驱动。

在解压后的 `cuda_sdf_assignment_starter` 目录执行：

```powershell
python -m pip install numpy
python tools/check_package.py
python -m unittest discover -s tests -v
```

若测试中 `test_cpu_far_mesh_has_no_initial_distance_cap` 在编译 CPU 之前被跳过，属于允许的情况；其余测试应通过。可使用 `python --version`、`nvcc --version`、`nvidia-smi` 检查环境。

## 2. Windows + VS2022 / RTX 5090 构建

从 **PowerShell 工程根目录**运行：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
```

```powershell
python tools/run_demo.py --case smoke_cube_16 --build build --config Release --repeat 3
python tools/run_demo.py --case smoke_cube_16 --build build --config Release --with-gpu --repeat 10
```

初始模板的预期结果：CPU 为 `CPU_PASS`，GPU 为 `NOT_IMPLEMENTED`（退出码 `3`）。这是**空白接口的正常提示**，并不表示 GPU 算法通过测试。

如已在 Visual Studio/CMake 其他目录编译，可指定程序完整路径：

```powershell
python tools/run_demo.py --case smoke_cube_16 --cpu-exe ".\build\Release\sdfbench_cpu.exe" --repeat 3
python tools/run_demo.py --case smoke_cube_16 --cpu-exe ".\build\Release\sdfbench_cpu.exe" --gpu-exe ".\build\Release\sdfbench_gpu.exe" --with-gpu --repeat 10
```

完成 `student/student_sdf.cu` 后重新编译，才应期待 GPU `PASS` 和 `results/<case>/gpu.sdf`。

## 3. Linux / CPU-only 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF
cmake --build build --config Release -j
python3 tools/run_demo.py --case smoke_cube_16 --build build --repeat 3
python3 -m unittest discover -s tests -v
```

Linux 下要构建 CUDA 时改为 `-DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` 并使用支持目标 GPU 的 Toolkit。

## 4. 公开静态测试建议顺序

每条命令运行前确保已经完成 CUDA 实现：

```powershell
python tools/run_demo.py --case cube_32 --with-gpu --repeat 10
python tools/run_demo.py --case sphere_32 --with-gpu --repeat 10
python tools/run_demo.py --case torus_64 --with-gpu --repeat 10
python tools/run_demo.py --case two_cubes_64 --with-gpu --repeat 10
python tools/run_demo.py --case sphere_128 --with-gpu --repeat 10
```

注意：`sphere_128` 公开压力测试可能对朴素算法较慢；建议先通过小规模正确性测试，再做优化和大规模测试。结果写入 `results/<case_id>/`。

完整自测中，静态立方体用 `tools/validate_cube.py` 核验解析距离/符号；球、环面等可用 `tools/validate_unsigned_samples.py` 做独立**无符号**距离抽样。公开 CPU/GPU 比较的默认绝对门槛 `0.1 * dx` 为**暂定自测值**，非正式成绩判定标准。

## 5. 修改与提交

- 主要实现 `student/student_sdf.cu`，不改 CUDA ABI、驱动、公共验证脚本和公开数据集。
- 至少提供 **2 项**优化，分别与基础版比较；记录 CUDA Events 耗时、正确性和原因解释。
- 提交代码、必要辅助文件、实验报告及运行说明。按 [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) 整理。**不要把 `build/`、`results/` 内全部大文件当作源代码提交**，原始测时与验证报告应按提交要求单独提供。
- 动态选做参考 [bonus/README.md](bonus/README.md)。当前发布包只有静态单帧驱动，没有动态流式自动判题器。

本 Starter 为学生开发包，**不包含教师私有 CUDA 参考实现、隐藏测试或最终排名程序**。正式评分以课程公布的统一规则为准。
