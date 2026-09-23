# Utonia SUN RGB-D 3D detector

This repository contains an implemented, custom 3D detector that uses the Utonia encoder. It predicts oriented cuboids as `(cx, cy, cz, dx, dy, dz, yaw)` in metric upright-depth coordinates. It does **not** yet contain a 3DSSG relation head or a complete scene graph pipeline.

## Model and data

- Backbone: [Utonia](https://github.com/Pointcept/Utonia), initialized from its released pretrained encoder for A and B. C builds the same architecture with random weights.
- Detector: Stage 2/3/4 multiscale adapter, up to 1536 FPS memory tokens, 128 spatial queries, and a three-layer Transformer decoder. Hungarian matching trains category, center, size, periodic heading, and rotated 3D IoU/GIoU terms.
- Dataset: SUN RGB-D V1, 5285 training scenes and 5050 validation scenes. Validation uses ten fixed categories and oriented 3D IoU with all-point AP. See [the configuration](configs/utonia_sunrgbd_full.yaml) and [evaluation code](detection/evaluate.py).
- Implementations: [model](detection/model.py), [data loading](detection/data.py), [training](scripts/train_benchmark.py), and [inference](scripts/detect.py).

## Completed 100-epoch experiments

The figures below come from each run's `epochs.csv` and `epoch_099_metrics.json`; epoch numbering starts at zero. All 100 epochs in each run validated on 5050 scenes. The recorded `last.pt` checkpoints mark epoch 99 complete.

| Run | Backbone training | Epoch 99 AP25 / AP50 | Best AP25 checkpoint (paired AP25 / AP50) | Best AP50 checkpoint (paired AP25 / AP50) |
| --- | --- | ---: | ---: | ---: |
| A | Pretrained, frozen | 61.4786% / 27.3884% | epoch 31: 63.0365% / 22.3925% | epoch 92: 62.5203% / 28.4882% |
| B | Pretrained, fine-tuned | 67.1043% / 39.5759% | epoch 44: 68.8095% / 36.7179% | epoch 96: 67.4075% / 41.7150% |
| C | Random initialization, trained | 52.2829% / 25.9180% | epoch 99: 52.2829% / 25.9180% | epoch 99: 52.2829% / 25.9180% |

A used effective batch 8; B and C used logical batch 20. B switched to two microbatches of 10 after an out-of-memory event, while C used microbatch 10 from the beginning. B and C skipped two and one nonfinite-gradient updates respectively. Each run used one random seed. These results show the observed performance of these runs; they are not a strict single-variable estimate of pretraining's causal effect. The cosine schedule retained the configuration's 180-epoch horizon while each run stopped at 100 epochs.

Large checkpoints, the SUN RGB-D data, experiment outputs, and local wheels are excluded from this source repository. A local trained B model is `outputs/benchmark/B_100/best_AP50.pt`; its checkpoint is about 1.67 GB and must be provided separately to reproduce its predictions. The released Utonia encoder checkpoint is downloaded separately; see [assets-manifest.json](assets-manifest.json) and [the upstream project](https://github.com/Pointcept/Utonia).

## Running the existing detector

Use the project dependencies in [environment.yml](environment.yml) and [requirements-inference.txt](requirements-inference.txt). Set `dataset.processed_dir` in [the YAML configuration](configs/utonia_sunrgbd_full.yaml) to a local SUN RGB-D V1 preprocessed directory. The preprocessing entry point is [prepare_sunrgbd_full.py](scripts/prepare_sunrgbd_full.py).

```bash
python scripts/detect.py \
  --input /path/to/processed_full_v1/001110.npz \
  --head /path/to/best_AP50.pt \
  --output outputs/inference_001110
```

The inference script saves oriented boxes, classes, scores, and a visualization. It does not need ground-truth boxes. For training options and the full protocol, see [README_SUNRGBD_Benchmark.md](README_SUNRGBD_Benchmark.md), bearing in mind its early-run results are historical.
