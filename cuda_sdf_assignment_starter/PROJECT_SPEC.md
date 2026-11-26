# CUDA 研究生大程序作业：静态三角网格 SDF 生成与性能优化

**学生发布版 V1.0.2（在 V1.0.1 Windows 修复版上整理）**

> **基础任务：静态 SDF GPU 生成、正确性验证与性能优化（100 分，必做）。**  
> **选做加分：固定拓扑动态网格的 SDF 连续更新（最多 10 分）。**  
> **目标环境：RTX 5090 / sm_120，CUDA 12.8+，CMake、C++17、Python 3 + NumPy。**  
> **学生需要编写的核心代码：`student/student_sdf.cu`；初始版返回 `NOT_IMPLEMENTED` 是正常状态。**

本文件是学生使用的主要任务说明。命令优先参考 `README.md`，数据说明见 `docs/DATASETS.md`，报告格式见 `docs/EXPERIMENT_REPORT_TEMPLATE.md`。作业截止时间和提交平台以教师单独通知为准。

## 1. 教学目标与研究背景

本课程大作业不是要求学生开发图形引擎，而是设计、实现并优化一个可以验证正确性的 CUDA 计算程序。

问题定义：已知**静态**三角形网格表面（OBJ 格式），在指定三维规则网格的每个采样位置计算**有符号距离场**（Signed Distance Field，SDF）。SDF 数值是空间采样点到模型表面的最近距离，**物体内部为负、外部为正**。**静态 SDF 实现和优化是全部学生的必做任务**。对固定拓扑、顶点坐标不断改变的变形网格进行连续帧更新，单独作为选做加分挑战。

学生应在项目中体现以下能力：

- 将一个数学定义明确的问题转化为并行 Kernel；
- 组织线程、数据布局、内存访问，处理多线程写入冲突；
- 进行正确性验证，理解近似误差、浮点误差及符号判定；
- 使用 CUDA Events 合理测量 GPU 工作耗时；
- 分析并优化实际瓶颈，而不是只提交不能解释的自动生成代码；
- 在统一数据集和 RTX 5090 上进行可复现的性能比较。

### 1.1 不需要的知识

课程**不要求**学生掌握 OpenGL、渲染、三维 CAD、网格生成或复杂空间数据结构。必要的点—三角形距离、符号判定及优化入门说明参见 `docs/GEOMETRY_PRIMER.md`。教师提供三角形数据文件、格式解析、统一采样点定义、参考计算、计时器和结果核对。学生可以自行探索更先进算法，但应以正确计算为首要目标。

### 1.2 学生需要理解的四个几何概念

1. **顶点**：三维坐标 \(v=(x,y,z)\)；
2. **三角形**：由三个顶点索引 \((a,b,c)\) 确定的一小块平面表面；
3. **网格**：许多三角形构成的封闭物体表面；
4. **SDF**：对位置 \(p\)，计算 \(\phi(p)=\pm\min_{t\in M}\mathrm{dist}(p,t)\)，其中内部使用负号，外部使用正号。

点到三角形的最短距离并不等于到三角形三个顶点的最近距离：最近点也可能位于三角形面内部或边上。这些情况都必须处理。

---

## 2. 工程组成与代码修改边界

学生发布包包含：

```text
cuda_sdf_assignment_starter/
├── README.md                         # Windows / Linux 快速开始
├── PROJECT_SPEC.md                   # 学生作业正式说明（本文件）
├── CMakeLists.txt                     # 统一构建规则
├── student/student_sdf.cu            # 必做：学生主要修改位置（初始为空）
├── student/student_sdf.cuh           # 固定 CUDA 接口，勿修改
├── src/                               # 公共 CPU 基线和 GPU 测试驱动
├── include/                           # 公共数据类型与 I/O
├── datasets/                          # 公开 OBJ 和 grid 数据、用例清单、校验和
├── tools/                             # 运行/比对/解析验证/数据校验脚本
├── tests/                             # 公开的验证器单元测试
├── docs/                              # 几何入门、数据说明、报告与提交清单
├── examples/                          # 公开的立方体 CPU 示例输出
├── bonus/README.md                    # 动态更新选做加分说明
└── third_party/                       # 上游来源与许可说明
```

**主要修改文件：** `student/student_sdf.cu`。可以按作业要求在 `student/` 内增加辅助 `.cu/.cuh` 文件，但必须明确列出，且不得改变公开 API；若新增 `.cu` 需要调整构建，应事先遵守教师公布的文件范围规则。不要修改 `student_sdf.cuh`、`src/gpu_main.cu`、CPU 参考、验证脚本、数据集与评分规则。教师最终评分将使用可信驱动与独立测试重新构建。

学生独立实现静态 SDF，需确保同一函数调用产生本帧全部输出。**不得使用模型名称、文件名或预存答案硬编码输出，也不得绕过统一计时区间。**

---

## 3. 输入数据：OBJ 三角网格与网格配置

### 3.1 OBJ 文件规范

采用简化 Wavefront OBJ：

```obj
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
f 1 2 3
```

- `v` 表示一个顶点，三个实数分别为 `x y z`。
- `f` 表示一个三角形，三个**从 1 开始计数**的顶点索引。
- 每个面都必须恰好由三个顶点构成；不接受四边形、多边形。
- 可以接受 `f 1/1 2/2 3/3` 形式索引，但是程序只使用其中的顶点索引；不需要 `vt`、`vn` 或材质。
- 不支持负索引。文件中的点坐标必须是有限数值。
- 教师提供封闭、正确连接、朝向一致的三角形网格，学生不必做网格修复。
- 每帧变形时，**顶点数量、面索引和面数保持不变，只更新顶点坐标**。

### 3.2 数据集说明

| 文件 | 顶点/三角形 | 用途 | 验证重点 |
|---|---:|---|---|
| `cube.obj` | 8 / 12 | 入门与解析参考 | 距离、内部符号、数据布局 |
| `icosphere_L1.obj` | 42 / 80 | 简单曲面近似 | 非轴对齐三角形、距离传播 |
| `icosphere_L2.obj` | 162 / 320 | 中等模型 | 访存、重复距离查询 |
| `icosphere_L3.obj` | 642 / 1280 | 较大示例 | 扩展性、吞吐测试 |
| `torus.obj` | 512 / 1024 | **必做静态公开用例** | 孔洞与符号判定 |
| `two_cubes.obj` | 16 / 24 | **必做静态公开用例** | 多连通分量 |
| `deform_000–002.obj` | 各 162 / 320 | **仅选做加分示例** | 固定拓扑连续帧更新（当前驱动只单帧运行） |

所有 OBJ 都已包含在发布包中，且可运行 `python3 tools/generate_examples.py` 重建。当前示例为**小型公开功能测试**，不是最终 RTX 5090 竞速数据集；静态必做的性能评测需要教师补充规模更大的隐藏用例。动态变形序列只用于可选加分，不能要求未选做的学生运行。

### 3.3 体素配置 `.grid`

为避免学生自行确定包围盒带来分歧，课程将体素网格配置从 OBJ 中剥离。示例：

```text
32 32 32
-1 -1 -1
0.06451612903225806
```

三行含义依次是：

1. `nx ny nz`：三维样本的个数；
2. `origin_x origin_y origin_z`：网格原点；
3. `dx`：各方向相同的采样间距（单位与 OBJ 相同）。

**严格采用 SDFGen 的节点采样定义，而不是体素中心采样：**

\[
 p(i,j,k)=\big(o_x+i\,dx,\ o_y+j\,dx,\ o_z+k\,dx\big).
\]

合法索引为 \(0\le i<n_x\)、\(0\le j<n_y\)、\(0\le k<n_z\)。

### 3.4 输出数组布局

输出是一维、连续、`float32` 的 SDF 数组。规定 `i` 最快变化：

\[
 \mathrm{offset}(i,j,k)=i+n_x(j+n_y k).
\]

所有学生实现必须严格遵守。符号约定：**内部负、外部正**。浮点零附近可能存在舍入误差，判题器会在规定窄带内降低符号判定敏感性。

---

## 4. CPU 基准算法与计时器

### 4.1 设计依据

`src/sdfgen_style.cpp` 是根据公开 SDFGen 的算法流程编写的**独立教学实现**，不应宣称与上游源码逐位相同。具体步骤：

1. **初始化**：把全部网格点距离初始化为大值，候选三角形编号初始化为 `-1`；
2. **窄带精确计算**：每个三角形的扩展包围盒范围内，逐点计算真正的点—三角形距离，更新最小值和对应的三角形 ID；
3. **交点累计**：沿 `x` 轴方向，为网格的每条 `(y,z)` 射线记录与三角形的交点；
4. **候选三角形传播**：执行 **8 个方向 × 2 轮**顺序扫描；每个网格点尝试从 7 个邻居传播候选三角形 ID，并重新计算到该候选三角形的欧氏距离；
5. **内部/外部符号**：沿 `x` 累计交点次数，奇数交点视为内部，取负号。

`exact_band` 默认是 `1`，代表三角形包围盒外扩 1 个采样单元的直接距离计算范围。距离值的局部计算基于连续三角形，不是单纯距离种子体素中心。

**精度边界：** 与 SDFGen 一样，顺序传播并不保证所有远处网格点都找到全局最近三角形；因此整个体积的结果**可能是近似 SDF**。实现已经修复初始距离的固定范围截断，并在没有候选三角形的节点上通过穷举最近三角形补齐，避免将无效初始值写为合法 SDF；这**不表示整个 CPU 结果已变成严格精确值**。教师应额外使用解析立方体或独立点—三角形距离抽样核验。

### 4.2 CPU 计时方法

CPU 运行程序 `sdfbench_cpu`：

- 先读取 OBJ 与 `.grid`；
- 执行指定次数的预热；
- 使用 `std::chrono::steady_clock` 在 `compute_sdfgen_style()` 调用前后计时；
- 记录每次运行耗时，计算中位数；
- **不把 OBJ 文件读取和 SDF 文件写入时间算进 CPU 算法时间**；
- 输出 `.sdf` 和计时 `.json` 文件。

计时接口示例：

```bash
./build/sdfbench_cpu \
  --mesh datasets/obj/cube.obj \
  --grid datasets/configs/grid_32.grid \
  --out results/cube_cpu.sdf \
  --report results/cube_cpu.json \
  --warmup 1 --repeat 3 --exact-band 1
```

CPU 时间不依赖 GPU；但对 CPU/GPU **速度比**的解释必须注明 CPU 型号、CPU 线程数、GPU 型号、编译参数及测试平台。此 CPU 基线默认单线程，不应把它解释为高度优化的多线程 CPU 极限性能。

### 4.3 CPU 实现与原版 SDFGen 的关系

- 保留算法步骤（窄带最近三角形初始化、8 方向扫描、两遍、射线奇偶内外判定）；
- 为便于课程调试，加入了更明确的数据类型、异常处理、退化边线段处理和计时；
- 输入网格范围由外部 `.grid` 固定，**不再依照 OBJ 包围盒自动生成范围**；
- 扫描/数值实现并非完全与原程序相同，因此只保证**算法级可对标**，不是已完成的 bitwise 一致性认证；
- 若教师需要与“官方 SDFGen 可执行文件”逐数值交叉验证，应单独下载固定 Git commit 的上游源码，适配同一网格参数并进行试验；这一部分是正式发布前的教师验收项。

---

## 5. GPU 空白实现、接口与计时器

### 5.1 学生唯一核心开发任务

GPU 源码文件：`student/student_sdf.cu`。

当前函数体只有：

```cpp
cudaError_t student_sdf_cuda(
    const float* d_vertices, int num_vertices,
    const std::uint32_t* d_triangles, int num_triangles,
    StudentGrid grid, float* d_sdf,
    cudaStream_t stream) {
    // TODO: 由学生实现
    return cudaErrorNotSupported;
}
```

**当前 GPU 文件是刻意留空的模板，不能生成合法距离场，也没有任何实际 GPU 性能。** 运行教师驱动时会报告 `NOT_IMPLEMENTED`，不会输出虚假的 SDF 或把空执行时间计为高分。

学生可以实现自己的算法（例如候选三角形传播、JFA、BVH 加速距离、其他正确算法），但必须遵守统一采样、输出、流同步和误差标准。

### 5.2 GPU 接口约定

- 所有 `d_` 开头指针均为**设备端内存**；教师驱动分配、初始化和释放设备内存。
- `d_vertices` 是 `[x0,y0,z0,x1,y1,z1,...]` 格式；每个坐标为 `float32`。
- `d_triangles` 是 `[a0,b0,c0,a1,b1,c1,...]` 格式；每个索引为 `uint32` 且从 0 开始。
- `grid` 包含 `ox,oy,oz,dx,nx,ny,nz`。
- `d_sdf` 容纳 `nx*ny*nz` 个 `float32`，学生实现必须填充**全部元素**。
- 所有与结果有关的设备任务必须与传入的 `stream` 有正确的执行顺序依赖。
- 如果使用内部 Stream，函数返回前必须确保传入的 `stream` 等待其完成；否则教师 CUDA Events 不能可靠覆盖所有工作。
- 不得依赖特定 OBJ 文件名、帧编号、公开结果或固定分辨率硬编码答案。

### 5.3 GPU 计时方法

教师程序 `src/gpu_main.cu`（学生不得修改）完成：

1. 主机端读取 OBJ 和 grid；
2. 申请 GPU 输入、输出内存，并在**计时前**复制静态模型到 GPU；
3. 填充输出缓冲区为非法哨兵值，便于检查遗漏写入（计时前）；
4. 在指定 CUDA Stream 上调用 `cudaEventRecord(start)`；
5. 调用 `student_sdf_cuda()`；
6. 在同一 Stream 上记录 `stop`，等待完成；
7. 用 `cudaEventElapsedTime()` 获得一次**GPU 计算区间耗时**；
8. 预热后重复执行，记录所有耗时；
9. 把计算结果复制回 CPU 并写出 `.sdf` 供核对。

GPU 运行命令：

```bash
./build/sdfbench_gpu \
  --mesh datasets/obj/cube.obj \
  --grid datasets/configs/grid_32.grid \
  --out results/cube_gpu.sdf \
  --report results/cube_gpu.json \
  --warmup 2 --repeat 10
```

当前 GPU 计时**不包括第一次 H2D 和最终 D2H 传输**，也不包括 host 端文件 I/O，可用于**必做静态任务**比较输入常驻显存时的计算耗时。**加分动态任务**若实际开放，教师须另行增加“每帧 H2D + 几何结构更新 + SDF 计算 + 规定结果传输”的端到端计时；不能用静态时间宣称动态实时速度。

**正式计分要求**：计时器应由教师可信构建流程控制；Student ABI 不应提供可信任的自报时间。单纯返回 `cudaSuccess` 并不意味着正确，必须核验输出。

### 5.4 RTX 5090 编译

RTX 5090 对应 Compute Capability 12.0，要求使用支持 `sm_120` 的 CUDA Toolkit（建议 CUDA 12.8 或更新版本）并匹配驱动。教师应冻结实际采用的 Toolkit 与驱动版本。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j
```

如果只需要在无 GPU 的开发机上构建 CPU 基线：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF
cmake --build build --config Release -j
```

在 Windows + MSVC 多配置 CMake 下，`tools/run_demo.py` 已兼容 `build/Release/sdfbench_cpu.exe` 和 `build/bin/Release` 等路径，可通过 `--config Release` 选择构建配置；但本次仅在 Linux 实测，仍须教师在真实 Windows/MSVC 上确认。

---

## 6. SDF 输出文件与 CPU/GPU 结果核对

### 6.1 `.sdf` 文本格式（沿用 SDFGen 风格）

```text
<nx> <ny> <nz>
<origin_x> <origin_y> <origin_z>
<dx>
<phi(0,0,0)>
<phi(1,0,0)>
...
```

例如 `32^3` 的完整 SDF 应具有 **32768** 个浮点值。行顺序和数组布局一致：i 最快变化，然后 j、k。教师驱动负责把 GPU 输出写为相同格式。

正式性能测试将文件 I/O 排除在计算计时之外，因为 ASCII 文件写入本身可能比 GPU 算法更耗时。长期扩展可改用二进制 SDF 存储，但必须同时修改 CPU、GPU 输出器和验证器。

### 6.2 独立核对命令

```bash
python3 tools/compare_sdf.py \
  --reference results/cube_cpu.sdf \
  --candidate results/cube_gpu.sdf \
  --report results/comparison.json \
  --abs-tol-dx 0.1 \
  --rel-tol 1e-4 \
  --max-bad-fraction 0.01 \
  --max-sign-fraction 0.005 \
  --sign-ignore-dx 0.25
```

当前工具输出：

| 字段 | 含义 |
|---|---|
| `finite` | 所有 GPU 输出值都不是 NaN/Inf |
| `mae` | 平均绝对误差 |
| `rmse` | 均方根误差 |
| `max_error` | 最大绝对误差 |
| `p95_error` | 绝对误差 95% 分位数 |
| `bad_fraction` | 不满足逐点误差容限的比例 |
| `sign_error_fraction` | 远离零面区域的内外符号错误率 |
| `passed` | 同时达到所有配置门槛才通过 |

逐点数值容限公式：

\[
  \lvert \phi_{\rm GPU}-\phi_{\rm CPU}\rvert
  \le \epsilon_{\rm abs}\,dx + \epsilon_{\rm rel}\lvert\phi_{\rm CPU}\rvert.
\]

零面附近小误差就会造成正负号切换，因此对于 \(\lvert\phi_{\rm CPU}\rvert\le\eta\,dx\) 的采样点，不单独计符号错误（但仍计距离误差）。

**警告：** 工具默认 `abs-tol-dx=0.1` 是修订后的**暂定演示阈值**，不能直接当成正式课程评分阈值。它比原来的 `1.0` 更严格，可阻止 `0.8dx` 系统偏差直接通过，但对与近似 CPU 不同的正确算法仍可能误判。正式评分应使用教师独立高精度**有符号**真值与隐藏数据校准，并保留 CPU 对比作为辅助诊断。

### 6.3 独立解析几何核验

为了验证 CPU 本身，额外提供**立方体的解析 SDF**：设立方体半边长 \(h=0.55\)，对任意点 \(p=(x,y,z)\)，令

\[
 q=(|x|-h,\,|y|-h,\,|z|-h).
\]

则其有符号距离为

\[
 \phi(p)=\|\max(q,0)\|_2+\min(\max(q_x,q_y,q_z),0).
\]

使用：

```bash
python3 tools/validate_cube.py --sdf results/cube_32/cpu.sdf --half-extent 0.55
```

这与 `compare_sdf.py` 的 CPU/GPU 对比是两套独立的检查：前者能发现 CPU 基线及几何符号问题；后者检查学生是否与 CPU 参考足够接近。`validate_cube.py` 现在真正按阈值返回 `PASS(0)` / `FAIL(1)` / `输入错误(2)`，不再只打印错误报告。

可对任意公开三角模型独立抽样验证**无符号**点—三角形精确距离：

```bash
python3 tools/validate_unsigned_samples.py \
  --mesh datasets/obj/torus.obj \
  --sdf results/torus_64/cpu.sdf \
  --samples 128 --report results/torus_64/unsigned.json
```

该工具不依赖 SDFGen 候选传播算法；**它不检查内外符号**，正式评分必须辅以独立符号真值和隐藏案例。

---

## 7. 如何运行：从零到一个完整测试

### 7.1 CPU 开发环境（无 CUDA）

```bash
# 依赖：CMake、支持 C++17 的编译器、Python3、numpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF
cmake --build build -j

# 运行一个最小案例：会生成 CPU SDF、CPU 计时 JSON、立方体解析误差报告
python3 tools/run_demo.py --case smoke_cube_16 --repeat 3

# 更复杂的模型
python3 tools/run_demo.py --case sphere_32 --repeat 3

# 检查核对工具是否能识别正常/故意损坏的输出
python3 -m unittest discover -s tests -v
```

### 7.2 CUDA 环境（RTX 5090）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j

# 初始 GPU 模板为空白；已构建 GPU 可执行文件时返回 NOT_IMPLEMENTED(3)，未构建则返回 4
python3 tools/run_demo.py --case smoke_cube_16 --with-gpu --repeat 10

# 学生完成 student/student_sdf.cu 后重新编译并运行上面的命令
```

`run_demo.py` 会组织 CPU、GPU（若请求）、立方体解析校验与比较器，并写入 `results/<case_id>/`。每次测试前会移除该用例的旧产物，以免读取陈旧结果。**它只是一用例演示工具，不是批量成绩管理系统。**

运行状态与退出码：

| 退出码 | 状态 | 解释 |
|---:|---|---|
| 0 | `CPU_PASS` / `PASS` | CPU 单独通过 / GPU 通过暂定比较规则 |
| 1 | `CPU_FAILED` / `CPU_ANALYTIC_FAIL` / `GPU_FAILED` / `CORRECTNESS_FAIL` | 执行失败或正确性不合格 |
| 2 | `CPU_EXECUTABLE_MISSING` 等输入错误 | 构建或输入缺失 |
| 3 | `NOT_IMPLEMENTED` | GPU 学生函数仍为空白 |
| 4 | `GPU_EXECUTABLE_MISSING` | 请求了 GPU 但找不到 GPU 可执行文件 |

Windows 多配置运行示例：`python tools/run_demo.py --case smoke_cube_16 --build build --config Release --with-gpu`。

### 7.3 首次运行时应该看到什么

- `sdfbench_cpu` 显示 `PASS CPU` 并输出 `.sdf` 与 `.json`；
- `validate_cube.py` 输出 MAE/最大误差/符号错误率；
- GPU 空白版显示 `NOT_IMPLEMENTED`，不生成 GPU `.sdf`；
- 当学生添加合法 CUDA 算法后，才开始产出 GPU `.sdf` 并运行误差核对；
- CPU/GPU 对比通过后，才能使用 GPU 计时进入性能评审。

---

## 8. 面向学生的阶段性任务

**阶段 A：读懂输入、接口和输出**

- 解释 OBJ 的顶点、三角形索引和 `.grid` 的采样位置；
- 理解 `d_sdf` 一维存储形式；
- 能说明 `cudaErrorNotSupported` 的作用；
- 学会读 CPU `.json` 和比较报告。

**阶段 B：完成正确 GPU 基础版本**

- 基于课程提供的几何距离定义，实现一个能填充所有体素的 GPU SDF 算法；
- 实现有效的内部/外部符号判定；
- 在公开小规模数据上达到规定正确性门槛；
- 使用 CUDA 错误检查解决越界与同步问题。

**阶段 C：性能优化**

- 至少从线程映射、内存布局、减少冗余计算、共享内存、GPU 规约或算子融合中选择两个方向探索；
- 对每次优化提供前后数据和解释；
- 可选研究三角形分块、表面初始化、并行传播、JFA、空间索引或 CUDA Graph。

**阶段 D：静态规模扩展与性能分析（必做）**

- 在 `cube`、`sphere`、`torus`、`two_cubes` 等**静态模型**上分别验证 SDF 正确性；
- 分析分辨率、三角面数对执行时间和临时显存的影响；
- 使用相同数据和计时边界对基础版本与优化版本做消融对比。

**阶段 E：报告和现场答辩**

- 提交代码、编译运行说明、误差报告、原始计时、优化记录；
- 解释自己的 Kernel 是怎样处理每个体素、怎样同步及避免竞争；
- 随机现场小改动，不要求在现场重写完整系统；
- 报告采用 `docs/EXPERIMENT_REPORT_TEMPLATE.md`，需提供原始计时记录和两项优化消融证据。

**阶段 F：动态网格连续更新（选做加分，可完全不做）**

- 在固定拓扑条件下连续处理 `deform_000–002.obj`，每帧都必须使用当前帧顶点坐标；
- 复用已分配内存、索引拓扑和恰当的中间数据，说明相对全量重算的收益；
- 测量每帧数据更新、几何更新和 GPU 计算的总时间；
- 此部分另计最高 10 分，**不是 A–E 任一基础任务的前置条件**。详见 `bonus/README.md`。

---

## 9. 教师评测与建议评分方法

**基础静态必做合计 100 分；动态更新选做加分最高 10 分。**未选择动态加分不会影响静态 SDF 的满分资格。建议保留 **30 + 25 + 30 + 15 分**结构：

| 项目 | 分值 | 证据 |
|---|---:|---|
| 距离与符号正确性 | 30 | SDFGen 风格 CPU 对照、解析模型、独立精确抽样及隐藏测试 |
| CUDA 程序实现与优化 | 25 | 自定义 Kernel、访存、线程组织、同步、异常处理 |
| RTX 5090 静态性能 | 30 | CUDA Events 中位数、规模扩展、预先固定的测试组权重 |
| 报告与答辩 | 15 | 真实实验数据、性能分析、现场解释和修改 |

静态性能 30 分可以进一步拆为：基础计算表现 10 分、规模扩展能力 8 分、优化消融收益 7 分、显存效率与稳定性 5 分。**正式性能评分的具体参考时间、数据集权重及判定门槛以教师后续统一公布的规则为准；公开自测阈值不等于最终评分阈值。**

选做动态加分最多 10 分：

| 项目 | 加分 |
|---|---:|
| 至少三个连续变形帧正确、无上一帧残留 | 3 |
| 复用固定拓扑、GPU 资源与合理缓存 | 2 |
| 相对逐帧全量重算具有可验证的优化收益 | 3 |
| 提供端到端逐帧计时、实验和显存分析 | 2 |

最终记分建议：`min(100, 基础得分 + 已通过正确性核验的动态加分)`。加分采用独立流程，不参与静态 CUDA Event 排名。**当前工程尚未实现加分的动态自动评测，不应在未实现的前提下宣称自动给分。**

正式打分建议：

1. **编译/运行冒烟测试**：失败则不进入性能测试。
2. **正确性检查**：距离容限、符号错误率和非法数值门槛全部通过才能给该组性能分。
3. **GPU 计时**：教师固定 GPU 驱动、CUDA 版本、测试输入、预热/重复次数，使用中位数。
4. **汇总**：在公开规则下对多组性能取加权几何平均加速比；高精度算法不能因为与近似 CPU 基线不同而被无理处罚。
5. **教师审核**：源代码、性能分析和现场问答共同构成作业完成证据。

**当前公开工具说明：** `gpu_main.cu` 提供静态 CUDA Events 计时；`compare_sdf.py` 的阈值仅用于自测，不能替代教师正式评分。动态加分需另行提交可复现实验，当前未提供自动化动态评分程序。

---

## 10. 参考资料

- Christopher Batty / SDFGen（代码与历史 CPU 基线）：https://github.com/christopherbatty/SDFGen
- NVIDIA CUDA GPU Compute Capability（RTX 5090：12.0）：https://developer.nvidia.com/cuda/gpus
- NVIDIA Toolkit/Architecture Matrix（Blackwell 支持）：https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- NVIDIA CUDA C++ Best Practices Guide：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
