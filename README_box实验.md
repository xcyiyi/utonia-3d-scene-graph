# ScanNet 小样例的 3D box 实验

已下载 MMDetection3D 官方发布的 ScanNet `scene0000_00` 演示点云，
共 40,684 个 XYZRGB 点，原始文件 976,416 字节；并取得同场景测试元数据中的 27 个参考框。
这是单场景迷你样例，不是完整 ScanNet 数据集，也不是独立测试集上的泛化评估。

## 方法

**Utonia 预训练编码器 → 官方 ScanNet 语义头 → 按预测类别进行 DBSCAN → 拟合轴对齐框。**

当前没有经过训练的 Utonia 检测头，所以这些是启发式实例框。没有训练新模型，
也没有把参考框、真实实例 ID 或真实语义标签传入预测过程。
参考框只在预测框保存以后用于评估和可视化对照。

点云法线缺失，按官方模型约定补零。墙和地板不作为检测对象；
保留 ScanNet 常用的 18 个物体类别。
固定参数：`scale=0.5`、语义概率阈值 `0.45`、DBSCAN 半径 `0.18m`、
`min_samples=5`、每个实例至少 40 个点。未根据参考框搜索或调整这些参数。

## 实际结果

本次产生 **36 个预测框**，参考框 **27 个**。

| IoU 阈值 | 正确匹配 TP | 多余框 FP | 未匹配参考框 FN | 精确率 | 召回率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 23 | 13 | 4 | 63.89% | 85.19% |
| 0.50 | 19 | 17 | 8 | 52.78% | 70.37% |

匹配按分数降序、同类别、一对一进行。以上是固定阈值下的单场景 precision/recall，
**不是 AP/mAP，也不能代表训练后的 Utonia 检测器性能**。
此样例及同源 ScanNet 场景可能被预训练或语义头训练覆盖，不作未见场景性能保证。

GPU 为 RTX 3090：编码器、特征上采样及语义头约 **1.05 秒**，
空间聚类约 **0.21 秒**，峰值分配显存约 **1.17GiB**，峰值预留约 **1.96GiB**。
计时不包含程序启动、权重加载、预处理及 HTML/图片导出；不是稳定吞吐基准。

结果可以初步覆盖不少家具，但存在多余碎片框、漏检和相邻物体合并。
分数为簇内点的平均语义概率，**不是经过校准的检测置信度**。
如果下一步要评价真正的检测能力，需要训练实例／框预测头并在独立验证集评估。

## 打开结果

结果目录：`outputs/scannet_mini_boxes/`。

- `boxes.html`：可离线打开的交互式 3D 对照图，左侧红色预测框，右侧绿色参考框；可旋转、缩放，悬停框线查看类别。
- `boxes_preview.png`：静态对比图。数字对应框 ID；预测框与参考框 ID 不是相同实例编号。
- `boxes.json`：框坐标、类别、分数、点数及方法说明。
- `boxes.npy`：`(36, 7)` 的 float32 数组，使用 ScanNet 轴对齐坐标系。
- `boxes_original.npy`：相同预测框转换到原始 `.bin` 点云坐标系后的中心与 yaw。
- `boxes.csv`：适合表格查看的框信息。
- `coordinate_transform.json`：坐标变换矩阵与旋转约定。
- `input_aligned.ply`、`semantic.ply`：轴对齐输入点云和语义结果。
- `instance_assignment.npy`：每个输入点的预测实例 ID，`-1` 表示未分配；低置信度点及聚类噪声可以未分配。
- `report.json`：参数、匹配明细、显存和运行时间。

为了便于看清室内布局，可视化隐藏少量最高处的顶棚点。
预测、评估、保存的 PLY 和 box 均未应用这个显示裁剪。

所有 box 格式为：

```text
[cx, cy, cz, dx, dy, dz, yaw]
```

中心是**几何中心**，不是底面中心；距离单位为米，yaw 单位为弧度。
`boxes.npy` 是轴对齐框，yaw 为零。原始坐标系的框保留由轴对齐变换产生的朝向，
yaw 表示绕 +Z 轴从 +X 方向逆时针旋转。它不是额外学习的物体朝向预测。
类别 ID 是 18 类检测索引；`semantic_class_id` 是官方语义头的 20 类索引。

```python
import json
import numpy as np

boxes = np.load('outputs/scannet_mini_boxes/boxes.npy')
with open('outputs/scannet_mini_boxes/boxes.json') as f:
    objects = json.load(f)['boxes']
for obj, box in zip(objects, boxes):
    print(obj['label'], obj['score'], box)
```

## 复跑

```bash
conda activate Utonia
cd /home/xcy/WorkSpace/Pointcloud/Utonia
python scripts/prepare_scannet_mini.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python scripts/box_baseline.py
```

准备脚本检测到已下载文件后只验证 SHA256，不重复下载。
几何、聚类和同类别一对一匹配的检查：

```bash
python -m unittest discover -s tests -p 'test_box_baseline.py' -v
```

原始数据位于 `data/scannet_mini/`，下载地址和 SHA256 在 `manifest.json` 中。
点云来源：[MMDetection3D ScanNet demo](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/scannet)。
参考框来源：[MMDetection3D ScanNet test metadata](https://github.com/open-mmlab/mmdetection3d/tree/main/tests/data/scannet)。
