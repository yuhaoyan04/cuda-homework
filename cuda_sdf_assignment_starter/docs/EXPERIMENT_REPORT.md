# CUDA SDF 实验报告

## 环境与构建

| 项目 | 本次记录 |
|---|---|
| 系统 / 编译器 | Linux x86_64，GCC 11.4，CMake 3.27.7 |
| GPU / 驱动 | NVIDIA H800，驱动 535.161.08 |
| CUDA | Toolkit 12.3.52；本次以 `sm_90` 编译 |
| 构建命令 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build build -j4` |
| 说明 | 课程目标为 RTX 5090 / CUDA 12.8+，本报告环境与目标不同，结果不可代替目标硬件成绩。 |

## 算法

一个 CUDA 线程负责一个采样点，按 `i + nx*(j + ny*k)` 写出结果。线程遍历全部三角形，使用 Ericson 最近点区域算法计算点到三角形面的精确欧氏距离；通过沿 +X 的射线奇偶交点判断内外，内部为负。射线的 Y/Z 坐标加很小的确定性偏移，减少恰好落在共享边上的歧义。

距离最小值使用三角形 AABB 到采样点的下界剪枝：下界不小于当前最小距离时跳过最近点区域计算，但仍做射线交点判断。线程到输出的映射连续、合并写入；网格和三角形数据均从驱动提供的设备内存读取，Kernel 使用传入 stream，不做全设备同步。

时间复杂度为 O(nx·ny·nz·T)，符号计算和最近距离均需遍历 T 个三角形；额外显存为 O(1)（不计输入和输出）。这是正确性优先的穷举实现，大网格/高面数下性能有限。退化三角形、极端坐标范围和非封闭网格尚未专项评估。

## 正确性与性能数据

| 用例 | GPU / CPU 比较 | 解析验证 | 结论 |
|---|---|---|---|
| cube，16³，12 三角形 | MAE `5.9807e-8`，最大误差 `2.38e-7`，符号错误率 0 | CPU 解析立方体 MAE `7.71e-9`，最大误差 `2.93e-8`，符号错误率 0 | smoke 用例通过 |

运行命令：`python3 tools/run_demo.py --case smoke_cube_16 --build build --with-gpu --repeat 3`。驱动输出的原始 CPU/GPU timing JSON、SDF、解析检查、比较报告和 summary 均保存在 `results/smoke_cube_16/`。本报告未记录 CPU/GPU 加速比或目标 GPU 数据，也没有对独立优化版本做计时消融，因此不能据此声称达到课程要求的性能分析分数。

## 未完成与复现边界

目前仅在 H800 上跑过 cube 16³ smoke。sphere、torus、two_cubes 和更大网格的独立正确性及性能结果尚未生成。至少两项优化的独立消融、CUDA Events 原始数据分析、目标 RTX 5090/CUDA 12.8+ 复测仍需补齐。使用项目 README 的 Linux CUDA 构建步骤，并将架构参数改为目标卡要求的 `120`，即可复现对应环境构建；本机 Toolkit 12.3 不支持该目标架构。
