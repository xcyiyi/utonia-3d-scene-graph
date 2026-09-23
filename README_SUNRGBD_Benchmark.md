# Utonia → SUN RGB-D 全量检测实验

> 本文保留早期单轮实验记录和当时的后续命令。A/B/C 的最终 100 轮结果与当前代码入口见 [README_DETECTOR.md](README_DETECTOR.md)；不要将下文“B/C 未完成”等历史状态当作现状。

保留已有 Utonia 和 Transformer 检测头，在现有 `Utonia` Conda 环境升级。这里是自定义检测实验，并非 3DETR 原模型的数值复现。预训练编码器本身不输出检测框，检测能力需要有监督训练 adapter/head，并通过充分训练的对照实验判断预训练收益。

## 数据与评估协议

- 数据目录：`/home/xcy/dabaset/SUNRGBD`；完整预处理：`processed_full_v1`；原 `processed_v2` 小样本实验保留。
- 官方 split：训练 **5285**，验证 **5050**。共 20670 个 RGB/depth 文件通过下载 CRC 检查。
- 使用 V1 检测标注，类别顺序为 bed/table/sofa/chair/toilet/desk/dresser/night_stand/bookshelf/bathtub。固定十类计算 mAP，输出每类 AP25/AP50。
- 每场景保存 50000 点，训练/验证使用 20000 点。训练每轮重新采样并增强，验证固定 seed。
- 保留训练 1183 个、验证 1019 个无目标场景；主结果覆盖全部 5050 验证场景。额外输出 4031 个非空场景的结果，便于理解与过滤空场景的作者实现之间的差异。
- 仅排除 V1 训练场景 7334 和 8481 各一个高度为零的原始标注；不修改验证标注。详见 `outputs/benchmark/audit/invalid_v1_annotations.json`。
- AP 使用旋转 3D IoU 和 all-point 积分；推理采用类别感知 AABB NMS（0.25）、空框过滤（至少 5 个输入点）、objectness > 0.05，并为保留框输出十类评分。这与 AP 的旋转框 IoU 是两个不同步骤。

参考官方实现：[3DETR 数据与增强](https://github.com/facebookresearch/3detr/blob/main/datasets/sunrgbd.py)、[VoteNet 框定义](https://github.com/facebookresearch/votenet/blob/main/sunrgbd/model_util_sunrgbd.py)、[3DETR 评估](https://github.com/facebookresearch/3detr/blob/main/utils/ap_calculator.py)。本项目空场景策略、法向量输入、自定义结构均应在论文对比时披露。

## 坐标、框和评估审计

点云、GT、预测均为米制 upright-depth：X 向右、Y 向前、Z 向上。框为 `cx cy cz dx dy dz yaw`，center 是几何中心，尺寸为完整边长。官方半尺寸 coeffs 转换为 `[coeff1, coeff0, coeff2] × 2`。内部 yaw 零点沿 +X，绕 +Z 逆时针为正，与 VoteNet clockwise heading 符号相反；独立生成的八角点一致。框具有 π 周期。

backbone 输入平移并缩放 0.5，但始终单独携带原始米制 XYZ，回归、GT 和评估使用同一坐标。训练和独立推理共用 `DetectionHead.forward` 和正式后处理。

全 V1 验证集 GT → evaluator：**AP25=100%，AP50=100%**，共 17273 个目标。旧四场景 GT 加小扰动（2 cm 中心、3% 尺寸、3° 角度）仍为 100/100；大中心偏差、尺寸误差会降低 AP。报告：`outputs/benchmark/audit/full_gt_v1_sanity.json`、`gt_evaluation_sanity.json`。

旋转 IoU 与独立 CPU 多边形裁剪器对比、GIoU 有限差分梯度、相同旋转框 GIoU=1、空目标反向传播、增强角点一致性、法向量长度、固定十类评估等共 13 项测试已通过。GIoU 使用两框 XY 凸包与包围 Z 区间构成的外包体。

## 原 AP50=0 的定位

全部四个旧验证场景的 256 个 query，没有任何一个与任何 GT 达到几何 IoU 0.5。14 个置信度 ≥0.3 的预测平均中心偏差 **0.638m**，平均尺寸绝对误差 **0.107m**；13 个类别与几何最接近的 GT 相同。主要证据指向定位和漏检，分类阈值无法修复几何偏差。

在固定几何匹配的 26 对诊断框中，仅将中心换成 GT 后，有 9 对达到 IoU50；仅换尺寸为 1 对；仅换 yaw 为 0 对；同时换中心和尺寸为 21 对。这是误差分解，不能当成模型 AP。yaw 仍有较大预测误差，且长宽交换/旋转对称性会影响角度误差的解读。

- 全部预测误差表：`outputs/benchmark/audit/predictions.csv`
- GT 覆盖表：`outputs/benchmark/audit/gt_coverage.csv`
- 四场景总图：`outputs/benchmark/audit/all_val_overlays.png`
- 交互图：同目录 `001110_overlay.html`、`001156_overlay.html`、`000270_overlay.html`、`001232_overlay.html`（GT 绿色，预测红色）。

原 17.10/0 是 V2 四场景、出现的七类平均，不能与新的 V1 全量十类指标直接作改善对比。

## 模型与资源

四个实际验证场景的层级测量如下；点数随场景、采样、增强变化：

| 层级 | 点数范围 | 通道 | 原始空间等效网格 |
|---|---:|---:|---:|
| Stage1 | 5384–7647 | 108 | 0.04m |
| Stage2 | 1830–3002 | 216 | 0.08m |
| Stage3 | 574–1053 | 432 | 0.16m |
| Stage4 | 155–387 | 576 | 0.32m |

Stage2/3/4 分别投影到 192 维，通过 pooling inverse 对齐到 Stage2 坐标，学习尺度权重融合。FPS 最多保留 1536 个 memory tokens，128 个空间 query 结合对应 feature 和 learned query embedding。memory/query 都包含 XYZ Fourier 编码；三层六头 decoder。中心为 reference XYZ 加偏移，尺寸 softplus+0.02，heading 为 sin(2yaw)/cos(2yaw)。支持 `query_type: learned` 对照。

分类、中心、尺寸、heading、GIoU 损失及各自权重在 YAML 中；Hungarian 代价包含 class/center/size/heading/GIoU。分别记录 matched IoU、中心距离、正 query 比率。

实机为 **2×RTX3090，每卡24GB**。A/B/C 双卡反向传播检查均通过，四样本/卡：冻结 A 约 1.03 GiB，解冻 B/C 约 4.37 GiB PyTorch peak allocated；这是检查批次的峰值，正式场景峰值以 CSV 为准。AdamW、BF16 检测头 AMP、FP32 稀疏 backbone、梯度裁剪与余弦 LR；解冻时 backbone LR=1e-5，head LR=1e-4。

## 执行与结果

正式配置：[configs/utonia_sunrgbd_full.yaml](configs/utonia_sunrgbd_full.yaml)。双卡全量数据 smoke 的 10 个 optimizer steps 已通过，验证仅 16 场景，不能用于评价性能。其目录为 `outputs/benchmark/full_smoke`。

冻结 A 的一个完整 epoch 与全量验证已完成，输出到 `outputs/benchmark/short_A`。全程约 **926 秒（15.4 分钟）**，661 个双卡 optimizer steps，覆盖全部 5285 训练场景；验证全部 5050 场景。长训练尚未启动。

| 指标 | 一轮短训练结果 |
|---|---:|
| 验证固定十类 mAP25 | **29.40%** |
| 验证固定十类 mAP50 | **2.28%** |
| train total / val total | 4.4704 / 3.5987 |
| train class / center / size | 0.6026 / 0.1090 / 0.2054 |
| train heading / GIoU loss | 0.4762 / 0.7537 |
| val class / center / size | 0.3896 / 0.0960 / 0.1557 |
| val heading / GIoU loss | 0.3962 / 0.6745 |
| val batch-averaged matched IoU | 0.4222 |
| val batch-averaged matched center distance | 0.2017m |
| 两卡中最高训练 peak allocated | 0.891 GiB |

loss 分项是未乘权重的主 decoder 输出，total 还包含权重和辅助 decoder losses，因此不能直接相加。匹配统计按 batch 再按样本数平均，并非所有目标的全局加权平均。显存数字是 PyTorch allocated，不包含驱动、上下文及部分第三方分配；nvidia-smi 验证期间约 1.6–2.0GB/卡。

| 类别 | AP25 (%) | AP50 (%) |
|---|---:|---:|
| bed | 71.30 | 7.27 |
| table | 26.64 | 0.43 |
| sofa | 46.98 | 1.96 |
| chair | 50.58 | 4.48 |
| toilet | 53.08 | 7.02 |
| desk | 13.47 | 0.12 |
| dresser | 3.57 | 0.02 |
| night_stand | 21.50 | 1.35 |
| bookshelf | 0.39 | 0.0011 |
| bathtub | 6.46 | 0.15 |

这证明全量数据上检测头能够学习，但 **AP50 仍然很低**，严格定位与部分类别识别尚未达到可用水平。一轮数据遍历不足以判断充分训练的最终性能，更不能判断预训练相对随机初始化的收益。当前正确性检查通过，适合继续受控训练观察 AP50 曲线；不要把 29.40% 解释为已经完成高质量 detector 训练。

损失与每类指标：`epochs.csv`；完整评估：`epoch_000_metrics.json`；曲线：`visualizations/loss_curves.png`；四场景预测：`visualizations/val_overlays.png` 与 HTML。新图以 0.1 置信度显示，每个几何框只显示最高分标签；这是可视化过滤，不改变上述 AP。

```bash
conda activate Utonia
# 已执行的短训练（新输出目录运行，避免覆盖）
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 \
  scripts/train_benchmark.py --output outputs/benchmark/short_A --epochs 1

# 审阅短训练结果后，续训 A 到配置的 180 epochs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 \
  scripts/train_benchmark.py --output outputs/benchmark/short_A \
  --resume outputs/benchmark/short_A/last.pt

# 独立 B：预训练初始化，全网络微调
torchrun --standalone --nproc_per_node=2 scripts/train_benchmark.py \
  --ablation B --output outputs/benchmark/B

# 独立 C：相同 Utonia 架构，随机初始化，全网络训练
torchrun --standalone --nproc_per_node=2 scripts/train_benchmark.py \
  --ablation C --output outputs/benchmark/C

# 阶段式 A → B（与独立 B 需分开报告）
torchrun --standalone --nproc_per_node=2 scripts/train_benchmark.py \
  --ablation B --initialize outputs/benchmark/short_A/best_AP25.pt \
  --output outputs/benchmark/A_then_B
```

`--epochs 1` 只限制停止轮次，不把学习率压缩到一轮，因此后续可以在相同配置和双卡数下恢复 optimizer/scheduler/RNG。smoke 的不完整 epoch 不允许精确 resume；`--initialize` 只加载权重。训练 DDP sampler 补齐一个样本（每轮记录5286次），验证没有补齐或重复。

每轮保存 CSV、TensorBoard、每类 AP、`best_AP25.pt`、`best_AP50.pt`、`last.pt`。A/B/C 已支持并检查梯度同步，但尚未完成充分训练的三组性能对比；目前不能据此宣称 Utonia 预训练提升多少。

独立输出 box（不读取 GT）：

```bash
python scripts/detect.py \
  --input /home/xcy/dabaset/SUNRGBD/processed_full_v1/001110.npz \
  --head outputs/benchmark/short_A/best_AP25.pt \
  --output outputs/benchmark/short_A/inference_001110 --score-threshold 0.1
```

输出 `boxes.npy`（N×7）、`boxes.json`（类别、分数与框）、`boxes.html`（点云与框）。默认可视化分数门槛0.3；短训练可能没有高置信框，调低门槛只用于观察，不能作为性能提升。

上述独立推理已实际通过：场景001110在阈值0.1下导出10个框，纯模型前向约0.76秒，`GT_used=false`。这是包含误检的真实短训练输出，不是人工修正后的框。完整验证的非空4031场景辅助结果为 AP25=29.83%、AP50=2.30%；主结果仍采用全部5050场景。
