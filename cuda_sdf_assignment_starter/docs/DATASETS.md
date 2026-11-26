# 公开测试数据说明

## 目录与格式

- `datasets/obj/`：OBJ 三角形网格。`v x y z` 为顶点，`f a b c` 为三角形（OBJ 索引从 1 开始），不支持四边形或负索引。
- `datasets/configs/`：网格 `.grid` 三行分别为 `nx ny nz`、`ox oy oz`、`dx`。**在节点采样**，不是体素中心。样本线性下标 `i + nx * (j + ny*k)`。
- `datasets/manifest.json`：公开命名用例与网格、模型对应关系。
- `datasets/SHA256SUMS`：OBJ 和 GRID 的 SHA256 校验值；建议用 `python tools/check_package.py` 校验。

## 公开案例

| case ID | 输入模型 | 网格 | 类别 / 检查重点 |
|---|---|---|---|
| `smoke_cube_16` | `cube.obj` (8 顶点 / 12 面) | 16³ | 静态冒烟 + 解析 SDF |
| `cube_32` | `cube.obj` | 32³ | 静态精度与符号 |
| `sphere_32` | `icosphere_L1.obj` (42 / 80) | 32³ | 静态球面 |
| `sphere_64` | `icosphere_L2.obj` (162 / 320) | 64³ | 静态中等规模 |
| `two_cubes_64` | `two_cubes.obj` (16 / 24) | 64³ | 静态分离实体 |
| `torus_64` | `torus.obj` (512 / 1024) | 64³ | 静态孔洞/内外符号 |
| `sphere_128` | `icosphere_L3.obj` (642 / 1280) | 128³ | **公开压力测试**（朴素算法可能慢） |
| `deform_000_64`–`deform_002_64` | 三帧 `deform_*.obj` (各 162 / 320) | 64³ | **动态选做**；当前静态驱动只能独立计算各帧 |

数据示例仅用于调试。评分可使用不同于公开样例的隐藏网格和采样范围；不得针对名称、顶点数量或公开结果硬编码。

## 验证边界

- 立方体解析验证既检查距离也检查正负号。
- 独立无符号抽样仅校验距离绝对值，**不能证明符号正确**。
- CPU SDFGen 风格参考在远离表面处可能是近似结果，CPU/GPU 差异不应成为唯一数学正确性依据。
- 公开比较器使用**暂定演示阈值**，不代表正式评分阈值。
