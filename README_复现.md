# Utonia 预训练推理复现

本目录包含 [Pointcept/Utonia 官方推理源码](https://github.com/Pointcept/Utonia)，
源码提交见 `UPSTREAM_REVISION.txt`。使用作者提供的编码器和 ScanNet 线性分割头，
无需从头训练。此流程验证预训练权重推理，不等同于复现论文全部数据集的指标。

新增：需要查看 3D box 输出时，见 [ScanNet 小样例 box 实验](README_box实验.md)。
该实验使用预训练语义预测和聚类生成启发式框，未训练检测头。

已完成可训练检测链路：见 [Utonia + SUN RGB-D 检测训练与验收](README_检测训练.md)。
包含真实预训练骨干、Transformer 检测头、旋转 box、10 次在线训练、小集过拟合和独立验证。

## 本机实测结果（2026-09-07）

两个 GPU 都已运行成功，保持官方网络、权重、FlashAttention 和默认 1024 注意力分块。
编码器参数量为 **137,253,744**。官方 `sample1` 共 **273,530** 个点，
`scale=0.5` 后保留 **187,983** 个体素点，全部层拼接特征为 **1386** 维。

| 验证 | GPU | 首次模型前向 | 峰值分配显存 | 峰值预留显存 |
| --- | --- | --- | --- | --- |
| RGB + 法线、PCA、分割、特征保存 | RTX 3090 / cuda:0 | 2.59 秒 | 4.15 GiB | 6.08 GiB |
| 仅坐标、PCA | RTX 3090 / cuda:1 | 2.28 秒 | 4.15 GiB | 6.08 GiB |

显存统计覆盖模型与推理后处理；前向耗时仅包含模型调用及 GPU 同步，不包含下载、
模型加载、输入预处理和文件导出。这是单次运行记录，不是稳定吞吐基准。
数字适用于当前示例与配置，不代表任意点数的点云都只需要这些显存。

结果分别在 `outputs/sample1/`、`outputs/geometry_only_gpu1/`，各自含 `report.json`。
`outputs/sample1/features_grid.npy` 已保存，约 994 MiB；原始点到体素点的映射也已验证。
官方编码器权重的 SHA256 已与 Hugging Face 元数据核对一致，记录见 `assets-manifest.json`。
依赖检查 `python -m pip check` 已通过。

## 环境与启动

环境名称区分大小写：`Utonia`。

```bash
conda activate Utonia
cd /home/xcy/WorkSpace/Pointcloud/Utonia
bash scripts/run_demo.sh
```

默认使用 `cuda:0`，在 `outputs/sample1/` 保存：

- `preview.png`：输入 RGB、PCA 特征着色与语义分割预览。
- `input.ply`、`pca.ply`、`segmentation.ply`：完整原始点数、原始坐标的点云，可用 Open3D 或 CloudCompare 打开。
- `semantic_labels.npy`：与输入点一一对应的 0–19 类别索引，类别名称见 `classes.json`。
- `semantic_scannet_ids.npy`：ScanNet 原始类别 ID。
- `report.json`：实际 GPU、点数、参数量、首次 forward 耗时及 PyTorch 峰值显存。

预览图是三维点云的静态视角；完整三维结果在 PLY 文件中。
PCA 颜色表达特征相似性，不是语义标签。语义分割使用监督训练的 ScanNet 20 类分割头，
不要把它理解成任意场景、任意类别的零样本分割模型。

## 使用自己的点云

支持 `.npz`，以及 Open3D 能读取的 `.ply`、`.pcd` 等点云格式。
NPZ 至少包含 `coord: (N, 3)`；可包含 `color: (N, 3)` 和 `normal: (N, 3)`。
`color` 使用 RGB **0–255**，法线通常为单位法向量。缺失颜色或法线按官方示例补零。
室内场景坐标通常按米准备；尺度不合适时需要调整 `--scale`。

```bash
# 仅导出特征 PCA，适合先检查自己的数据
bash scripts/run_demo.sh --input /path/to/cloud.ply --task pca --output outputs/custom

# 不使用颜色/法线
bash scripts/run_demo.sh --task pca --wo-color --wo-normal --output outputs/geometry_only

# 室外 LiDAR：保持自车在原点，路面与 XY 平面对齐
bash scripts/run_demo.sh --input /path/to/lidar.npz --task pca --outdoor --scale 0.1 --output outputs/outdoor

# 单个物体：先归一化坐标
bash scripts/run_demo.sh --input /path/to/object.ply --task pca --normalize-coord --scale 1 --output outputs/object

# 保存全部层拼接的特征（1386 维）及映射
bash scripts/run_demo.sh --task pca --save-features --output outputs/features
```

`features_grid.npy` 是体素采样后的特征，`inverse.npy` 将原始点映射到体素点，
`coord.npy` 保留原始坐标。恢复逐点特征可使用：

```python
import numpy as np
features = np.load('outputs/features/features_grid.npy', mmap_mode='r')
inverse = np.load('outputs/features/inverse.npy')
# 对大点云分块读取，避免一次分配 N × 1386 的数组
first_chunk = features[inverse[:10000]]
```

`--scale` 越大，固定 0.01 体素网格一般会保留越多点，显存和耗时也通常更高。
降低 scale 会改变特征与预测，不是无损加速。如果出现 OOM，先减小 scale 或裁切点云。
`--no-flash --patch-size 512` 是官方支持的普通注意力回退方案，会改变注意力分块，
不能保证与默认 FlashAttention 配置得到相同结果。

## 两张显卡

安装时检测到当前主机是 **2 × RTX 3090 24GB**，驱动 `535.247.01`。
环境使用 Python 3.10、PyTorch 2.5.1+cu121、spconv-cu121、FlashAttention 2.7.4.post1，
避免要求安装系统级 CUDA Toolkit 或升级驱动。

原始官方 `environment.yml` 使用 CUDA 12.4 / PyTorch 2.5.0；本机采用上述兼容组合。
不改变网络结构和权重，但不同软件版本及 PCA 随机性可能使可视化有差异。

单个默认推理任务使用一张卡。两张 24GB 卡不会自动合并为 48GB 显存。
可在两个终端分别处理不同点云，输出目录也应不同：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_demo.sh --input /path/to/a.ply --task pca --output outputs/a
CUDA_VISIBLE_DEVICES=1 bash scripts/run_demo.sh --input /path/to/b.ply --task pca --output outputs/b
```

脚本也支持直接指定 `--device cuda:1`。RTX 4090 同样具备 24GB 显存；
这套 CUDA 12.1 / FlashAttention 配置支持其架构，但 4090 的性能需要在实际机器上测量。

## 重建与官方示例

当前环境已创建后，不必重新执行安装。需要在另一台 Linux x86_64 GPU 主机重建时：

```bash
bash scripts/setup_environment.sh
conda activate Utonia
bash scripts/run_demo.sh
```

核心版本在 `requirements-inference.txt`；完整 Python 依赖在 `requirements-lock.txt`，
Conda 环境实录在 `outputs/environment-installed.yml`。后者包含本机路径，仅作追溯。
官方权重来自 [Pointcept/Utonia](https://huggingface.co/Pointcept/Utonia)，
示例数据来自 [pointcept/demo](https://huggingface.co/datasets/pointcept/demo)。
`checkpoints/utonia.pth` 是推理专用编码器；不需要下载体积更大的原始预训练检查点。

原始 `demo/0_pca_indoor.py` 等脚本保持不变，默认弹出窗口并使用用户缓存目录下载资源。
有桌面显示环境时可运行 `python demo/0_pca_indoor.py`；SSH 下建议使用上面的无窗口脚本。
VGGT 视频重建属于额外模型流程，本环境没有安装 VGGT。

代码为 Apache-2.0；官方权重为 CC-BY-NC-4.0，具体见上游仓库许可说明。
