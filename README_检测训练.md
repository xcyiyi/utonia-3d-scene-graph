# Utonia + SUN RGB-D：真实可训练的 3D Box 检测闭环

全量 SUN RGB-D V1 的审计、多尺度 adapter、双卡训练及新结果见 [README_SUNRGBD_Benchmark.md](README_SUNRGBD_Benchmark.md)。本文保留原 8/4 场景 prototype 的历史记录。

本次已实际完成预训练骨干检查、官方数据子集下载、预处理、GT 检查、检测头训练、10 次在线 smoke、1,500 步小集过拟合、独立验证和保存权重后的独立推理。结果证明链路可训练；**当前检测头只用 8 个场景训练，尚不能可靠检测新场景**。

这份实现使用真正的 Utonia PTv3 编码器，不包含 PointNet++，也不使用此前的语义分割 + DBSCAN 启发式框。检测 Adapter 和 Head 是本项目新训练的模块，不是官方发布的检测权重，也不声称严格复现 3DETR 的全部架构、损失或评估协议。

## 实际环境

- Conda：现有 `Utonia`，解释器 `/home/xcy/miniconda3/envs/Utonia/bin/python`。
- Python 3.10.21，PyTorch 2.5.1+cu121，CUDA runtime 12.1，驱动 535.247.01。
- 实际 GPU：**2 × NVIDIA RTX 3090，单卡 24GB**。
- 本任务没有新建环境，没有修改 Python / PyTorch / CUDA，也没有安装新依赖。
- 冻结训练：GPU 0、batch size 2、每场景 20,000 个输入点、64 queries、FP32、无梯度累积。
- 冻结在线 smoke 峰值 PyTorch allocated 0.670 GiB，reserved 0.719 GiB；GPU context 等额外显存不在此数内。
- GPU 1 上解冻骨干、batch size 1 的单步反向更新也通过，allocated 峰值 2.679 GiB。这只是当前点数和配置的测量，不代表任意规模训练都只需这些显存。
- 主训练没有实现或运行 DDP；当前规模单卡足够。两张卡的显存也不会自动合并。

## 骨干真实输出与检测接口

代码：`utonia/model.py`。本仓库是 Pointcept 团队的独立推理实现，PTv3 的 Point 结构保留层次池化关系；无需把完整 Pointcept 或旧 3DETR 装入环境。

权重：`checkpoints/utonia.pth`，137,253,744 参数。

首先用已有官方 `data/sample1.npz` 做真实 forward，输入 273,530 点，scale=0.5。阶段编号与代码 `enc0`–`enc4` 一致，从零开始：

| Stage | XYZ shape | Feature shape | Batch shape | Offset |
|---|---|---|---|---|
| 0 | [187983, 3] | [187983, 54] | [187983] | [187983] |
| 1 | [69097, 3] | [69097, 108] | [69097] | [69097] |
| 2 | [18401, 3] | [18401, 216] | [18401] | [18401] |
| 3 | [4718, 3] | [4718, 432] | [4718] | [4718] |
| 4 | [1133, 3] | [1133, 576] | [1133] | [1133] |

完整字段及 dtype：`outputs/detection/backbone_shapes.json`。`grid_coord` 为 [N,3] 整数；Stage 0 的 `inverse` 把原始点映射到初始体素。各后续层的 `pooling_inverse` 把上一层点映射到本层，`pooling_parent` 引用上一层 Point。`batch` 是逐点场景 ID，`offset` 是批内累计点数，不能当成单场景长度列表。

实际检测接口：

```text
XYZ / RGB / normals -> 9维输入 -> 官方 Utonia（冻结）
                                  ├─ Stage 3: [N3,432]
                                  └─ Stage 4: [N4,576]
                                       pooling_inverse 映射到 Stage 3
                                              ↓
                                  [N3,1008]，保留 Stage 3 XYZ
                                              ↓
                           FPS 最多 1024 个 memory tokens
                                              ↓
                           UtoniaDetectionAdapter：1008 -> 192
                                              ↓
                           3 层 Transformer Decoder、6 个 attention heads
                           64 个空间采样锚点 + 可学习 object queries
                                              ↓
                           分类 10+背景、中心、尺寸、sin(2yaw)/cos(2yaw)
                                              ↓
                           [B,64,7] 旋转 3D boxes
```

本次 SUN RGB-D 的 Stage 3 为 **213–1,134 点，432 维，XYZ 为 [N,3]**；Stage 4 为 64–412 点。例：训练场景 006307 的 Stage 3 为 [713,432]、XYZ [713,3]，融合后 [713,1008]。第 3 层的物理网格间距约 16cm（初始 2cm × 8），更密的 Stage 2 约 8cm，可作为后续小目标检测消融选项。

选择 Stage 3 是本轮空间细节、语义和成本的工程折中，不是已证明的最佳层。只把深层特征恢复到 Stage 3，不恢复到全部原始点。FPS、投影和 Transformer 都使用当前 PyTorch，无旧版 PointNet2 CUDA extension。

## SUN RGB-D 数据与坐标

用户指定目录严格保留为 **`/home/xcy/dabaset/SUNRGBD`**。目前是可扩展的 12 场景子集，**没有下载完整 SUN RGB-D，也没有下载完整 ScanNet**。

来源全部为 [SUN RGB-D 官方数据目录](https://rgbd.cs.princeton.edu/data/)：

- `SUNRGBDMeta3DBB_v2.mat`：`wget -c` 下载完整约 9.4MB 检测元数据。
- 官方 `SUNRGBDtoolbox.zip`：按 HTTP Range 提取 `allsplit.mat` 和投影参考函数。
- 官方 `SUNRGBD.zip`：按 HTTP Range 只提取 12 场景的 RGB 与深度图。
- `range_cache/` 保存已完成字节块，中断后可继续；ZIP CRC 校验，文件 SHA256 记入 manifest。
- 不需要 2D 检测标注、语义分割标注、MATLAB 或 vote labels。

官方划分总计 5285 train / 5050 test；检测工作中把后者作为 val。子集用种子 37 从存在 1–32 个目标类别标注的场景中抽样，ID 固定保存在 `outputs/detection/subset_ids.json`。本次抽到的 12 个场景均来自 Kinect v2，不能代表跨传感器表现。

- 训练 8 场景：6307、5327、5177、6244、6665、5516、6309、6201；44 个 GT 框。
- 验证 4 场景：1110、1156、270、1232；26 个 GT 框。
- 两者 ID 不重叠。验证样本不参与梯度更新、类别统计、阈值选择或模型选择。
- 训练集只覆盖 7 类；desk、bookshelf、bathtub 未被训练子集覆盖，验证集中包含其中的 desk 和 bookshelf。

预处理遵循官方深度解码：uint16 循环移位、转换为米、8m 上限、去除无效深度、1 起始像素索引反投影、Rtilt 变换。RGB 与 XYZ 对齐，固定随机采样 20,000 点，Open3D 估计朝向相机的法线。标注转换参照 [VoteNet 作者 SUN RGB-D 代码](https://github.com/facebookresearch/votenet/blob/main/sunrgbd/sunrgbd_utils.py)：交换 coeffs 的前两维并乘 2，yaw 为 orientation 的 atan2。

输出格式：

```text
[cx, cy, cz, dx, dy, dz, yaw]
```

中心和完整尺寸单位为米；坐标 X 向右、Y 向前、Z 向上；yaw 绕 +Z 逆时针，单位弧度。yaw 与 yaw+π 描述同一无方向 cuboid，本实现使用双角回归并输出 [-π/2, π/2] 范围的等价角。它不预测物体“正面朝向”。

骨干坐标居中、缩放仅作用于网络输入；另通过 `origin_coord` 传播原始米制坐标，检测框始终在原始 upright-depth 坐标系。推理不会按 GT 决定框的位置或数量。

检查记录：`outputs/detection/gt/dataset_sanity.json`；所有 GT 的逐框包含点数、尺寸合法性、有限性及划分均有记录。GT 俯视图 `outputs/detection/gt/gt_preview.png`，各场景离线交互图 `outputs/detection/gt/*_gt.html`。

## 已执行的训练与结果

损失：Hungarian 一对一匹配；类别交叉熵（背景权重 0.1）、米制中心 L1、尺寸 L1、双角向量 MSE；中间 decoder 层辅助监督。没有使用 GT 聚类或手工匹配生成预测框。当前版本没有实现原版 3DETR 的所有 GIoU、增强和训练日程。

1. 单场景 forward：logits [1,64,11]，boxes [1,64,7]，heading vector [1,64,2]。
2. 10 次在线训练：每次重新运行 Utonia，完成 matching → loss → backward → optimizer.step，loss/预测/梯度全有限。
3. Adapter 参数最大变化 0.002998；Utonia 所有参数在 smoke 前后计算 SHA256，完全一致，且没有骨干梯度。
4. 缓存真实冻结特征后，8 场景训练 1,500 步，head LR 3e-4 余弦下降至 1e-5，batch=2。缓存阶段不采用 GT 选点；验证特征只用于评估。
5. 前 50 步平均 loss **2.7033**，末 50 步 **0.01004**；1,500 步损失全部有限。
6. 独立 GPU 微调单步检查：452 个骨干参数张量获得梯度，真实权重更新通过，backbone LR=1e-5、head LR=1e-4。此检查不改动交付的冻结训练权重。
7. 保存权重后，独立 `detect.py` 重加载并在验证场景 001110 推理，输出 2 个 score≥0.3 的框；与缓存特征评估输出做数值一致性检查。

| 评估子集 | mAP@0.25 | mAP@0.5 | IoU0.5 / score≥0.3 的 TP / FP / FN |
|---|---:|---:|---|
| 8 场景训练集（过拟合检查） | 100% | 100% | 44 / 0 / 0 |
| 4 场景验证集（未训练） | 17.10% | 0% | 0 / 14 / 26 |

验证集 IoU0.25、score≥0.3 时为 TP=3 / FP=11 / FN=23，precision=21.43%、recall=11.54%。

**指标口径：**精确旋转 3D IoU（XY 旋转多边形交集 × 高度交集），全点插值 AP；每 query 只取最可能的前景类，无 NMS；AP 使用所有 query，不设 0.3 截断。0.3 仅用于导出框和 P/R。因此某类 AP 非零而在 0.3 阈值下 TP 为零是可能的。mAP 只平均本子集中存在 GT 的 7 类，缺失类 AP=null；**不能与完整 SUN RGB-D 的标准十类 mAP 直接比较**。

结果文件：`outputs/detection/training/final_metrics.json`；loss 曲线 `outputs/detection/loss_curve.png`；GT/预测对照 `outputs/detection/box_comparison.png`。

这说明 Utonia 特征接入检测头在工程上可行，监督链路能拟合训练框；训练集 100% 是过拟合验收，不是泛化成绩。验证表现较弱，且数据、类别覆盖有限，目前不能判断 Utonia 相比随机初始化或其他预训练骨干的检测增益。要评估“作为 backbone 效果怎么样”，应扩展官方训练集，保持完整验证集独立，再比较冻结/解冻、Stage2/3、预训练/随机初始化等同配置基线。当前权重不能直接作为可靠场景图节点检测器部署。

## 直接运行与输出文件

已有权重：`outputs/detection/training/detector_head.pt`（Adapter + Head + optimizer 状态，骨干单独保存于官方权重文件）。

```bash
conda activate Utonia
python scripts/detect.py \
  --input ~/dabaset/SUNRGBD/processed_v2/001110.npz \
  --output outputs/detection/my_boxes
```

输出：`boxes.npy` [K,7]、`boxes.json`（类别及置信度）、离线 `boxes.html`、全部 64 查询的 `all_queries.npz`、时间/显存 `report.json`。阈值默认 0.3，可用 `--score-threshold` 调整。

输入 NPZ 的 `coord/color/normal` 为 [N,3]，颜色 [0,255]，XYZ 必须是米制、Z-up；也支持 PLY。没有颜色或法线时沿用通用读取器的零填充，可能影响效果。本轮评估数据使用真实 RGB 和估计法线。`detect.py` 只读取几何、颜色和法线，不读取文件中的 GT boxes/labels。

已有预测示例：

- 训练：`outputs/detection/training/predictions/train_overfit/006307_boxes.html` 和同名 `_boxes.npy`。
- 验证：`outputs/detection/standalone_val_001110/boxes.html` 和 `boxes.npy`。
- 完整 12 场景预测在 `outputs/detection/training/predictions/`。

从现有数据完整重跑（写入新输出目录）：

```bash
conda activate Utonia
python scripts/inspect_backbone.py
python scripts/train_detection.py --phase all --iterations 1500 \
  --output outputs/detection/rerun
```

本次实际执行为 `--phase smoke`，之后 `--phase overfit --resume .../smoke_head.pt`。`--resume` 是载入模型初始化权重，当前不会恢复 optimizer/scheduler/RNG，因此不是精确断点恢复；分阶段运行和 `--phase all` 的随机轨迹也可能略有不同。交付指标来自本目录保存的日志及权重。

解冻微调（不会读取冻结特征缓存）：

```bash
python scripts/train_detection.py --phase finetune \
  --resume outputs/detection/training/detector_head.pt \
  --head-lr 1e-4 --backbone-lr 1e-5 --batch-size 1 --iterations 100 \
  --output outputs/detection/finetune_run
```

可选 `--amp` 使用 BF16 autocast；此次验收使用 FP32，没有对 AMP 单独出具运行结果。完整解冻训练需根据点数再测显存。

重新准备相同官方子集（无需完整 ZIP）：

```bash
conda activate Utonia
mkdir -p ~/dabaset/SUNRGBD
wget -c -O ~/dabaset/SUNRGBD/SUNRGBDMeta3DBB_v2.mat \
  https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat
python scripts/download_sunrgbd.py --toolbox
python scripts/download_sunrgbd.py --ids-file outputs/detection/subset_ids.json
python scripts/prepare_sunrgbd.py
```

改变 JSON 中的 train/val ID 可扩展数据，预处理会核验每个 ID 的官方划分；应先保证完整类覆盖和场景多样性。正式大规模运行还需优化数据加载、增强、训练日程及断点续训，这些不是本次小集闭环的验证范围。

## 代码与验收入口

- `detection/data.py`：米制坐标保留、体素化、变长 batch。
- `detection/model.py`：真实 Utonia、Adapter、Decoder、FPS、冻结/解冻学习率组。
- `detection/loss.py`：匹配与损失。
- `detection/geometry.py` / `evaluate.py`：旋转框几何、离线可视化与评估。
- `scripts/train_detection.py` / `detect.py`：训练和无 GT 推理。
- `outputs/detection/training/smoke_audit.json`：10 次在线验收。
- `outputs/detection/finetune_audit.json`：解冻更新验收。
- `outputs/detection/acceptance_summary.json`：过拟合与独立推理检查摘要。
- `outputs/detection-unit-tests.log`：5 项测试，包括旋转 IoU、重复框 AP、空 GT、匹配置换、坐标和 batch、padding 独立性；一项测试可含多个断言。

```bash
python -m unittest discover -s tests -p 'test_detection.py' -v
python -m pip check
```

参考：[官方 Utonia 仓库](https://github.com/Pointcept/Utonia)、[SUN RGB-D 官方主页](https://rgbd.cs.princeton.edu/)、[VoteNet 数据提取流程](https://github.com/facebookresearch/votenet/blob/main/sunrgbd/README.md)、[3DETR 作者实现](https://github.com/facebookresearch/3detr)。
