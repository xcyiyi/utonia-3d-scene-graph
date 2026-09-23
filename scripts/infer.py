"""Run official Utonia weights offline and export results without a GUI."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
import torch
import utonia


def write_ply(path, coord, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coord)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(color, 0, 1))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write {path}")


def load_input(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            point = {k: data[k].copy() for k in ("coord", "color", "normal") if k in data}
    else:
        cloud = o3d.io.read_point_cloud(str(path))
        point = {"coord": np.asarray(cloud.points).copy()}
        if cloud.has_colors():
            point["color"] = np.asarray(cloud.colors).copy() * 255
        if cloud.has_normals():
            point["normal"] = np.asarray(cloud.normals).copy()
    if "coord" not in point or point["coord"].ndim != 2 or point["coord"].shape[1] != 3:
        raise ValueError("Input must contain coord with shape (N, 3)")
    if len(point["coord"]) < 16:
        raise ValueError("Input must contain at least 16 points")
    for key in ("coord", "color", "normal"):
        if key not in point:
            point[key] = np.zeros_like(point["coord"])
        # Preserve the official sample's dtypes until its ToTensor transform.
        if not np.issubdtype(point[key].dtype, np.number):
            raise ValueError(f"{key} must be numeric")
        if key != "color" and not np.issubdtype(point[key].dtype, np.floating):
            point[key] = point[key].astype(np.float32)
        if point[key].shape != point["coord"].shape or not np.isfinite(point[key]).all():
            raise ValueError(f"{key} must be finite and have shape (N, 3)")
    if point["color"].min() < 0 or point["color"].max() > 255:
        raise ValueError("NPZ color must be RGB in [0, 255]")
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/sample1.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/utonia.pth")
    parser.add_argument("--head", type=Path, default=ROOT / "checkpoints/utonia_linear_prob_head_sc.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sample1")
    parser.add_argument("--task", choices=["pca", "seg", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--outdoor", action="store_true", help="Keep LiDAR origin; road must be on XY plane")
    parser.add_argument("--normalize-coord", action="store_true", help="Normalize individual objects")
    parser.add_argument("--wo-color", action="store_true")
    parser.add_argument("--wo-normal", action="store_true")
    parser.add_argument("--no-flash", action="store_true", help="Official fallback with smaller attention patches")
    parser.add_argument("--patch-size", type=int, default=1024, help="Only used with --no-flash")
    parser.add_argument("--save-features", action="store_true", help="Save grid features and original-to-grid inverse map")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if args.scale <= 0 or args.patch_size <= 0:
        parser.error("scale and patch-size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for spconv inference. Run on the GPU host, outside the restricted sandbox.")
    torch.cuda.set_device(device)
    utonia.utils.set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    point = load_input(args.input)
    if args.wo_color:
        point["color"].fill(0)
    if args.wo_normal:
        point["normal"].fill(0)
    coord = point["coord"].copy()
    rgb = point["color"].copy() / 255
    config = None
    if args.no_flash:
        config = dict(enable_flash=False, enc_patch_size=[args.patch_size] * 5)
    elif utonia.model.flash_attn is None:
        raise RuntimeError("Install FlashAttention or explicitly select --no-flash")
    model = utonia.load(str(args.checkpoint), custom_config=config).to(device).eval()
    transform = utonia.transform.default(args.scale, not args.outdoor, args.normalize_coord)
    point = transform(point)
    grid_count = len(point["coord"])
    print(f"Input: {len(coord):,} points; grid: {grid_count:,}; GPU: {torch.cuda.get_device_name(device)}", flush=True)
    official_pca = runpy.run_path(str(ROOT / "demo/0_pca_indoor.py"))["get_pca_color"]
    official_seg = runpy.run_path(str(ROOT / "demo/2_sem_seg.py"))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    colors = {"input": rgb}
    with torch.inference_mode():
        point = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
        point = model(point)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started
        # Match the official PCA demo: concatenate the deepest three levels.
        for _ in range(2):
            parent = point.pop("pooling_parent")
            parent.feat = torch.cat([parent.feat, point.feat[point.pop("pooling_inverse")]], dim=-1)
            point = parent
        pca_feat = point.feat
        # Segmentation and saved features use all five encoder levels.
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            pca_feat = pca_feat[inverse]
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        if not torch.isfinite(point.feat).all():
            raise RuntimeError("Model produced non-finite features")
        original_inverse = point.inverse.cpu().numpy()
        if args.task in ("pca", "both"):
            pca = official_pca(pca_feat, brightness=1.2, center=True)
            colors["pca"] = pca.cpu().numpy()[original_inverse]
        if args.task in ("seg", "both"):
            ckpt = utonia.load(str(args.head), ckpt_only=True)
            head = official_seg["SegHead"](**ckpt["config"]).to(device).eval()
            head.load_state_dict(ckpt["state_dict"])
            pred = head(point.feat).argmax(dim=-1).cpu().numpy()[original_inverse]
            np.save(args.output / "semantic_labels.npy", pred)
            class_ids = np.asarray(official_seg["VALID_CLASS_IDS_20"])[pred]
            np.save(args.output / "semantic_scannet_ids.npy", class_ids)
            colors["segmentation"] = np.asarray(official_seg["CLASS_COLOR_20"])[pred] / 255
            (args.output / "classes.json").write_text(json.dumps(official_seg["CLASS_LABELS_20"], indent=2) + "\n")
        if args.save_features:
            np.save(args.output / "features_grid.npy", point.feat.cpu().numpy())
            np.save(args.output / "inverse.npy", original_inverse)
            np.save(args.output / "coord.npy", coord)
        torch.cuda.synchronize(device)
        report = {
            "input": str(args.input.resolve()), "checkpoint": str(args.checkpoint.resolve()),
            "gpu": torch.cuda.get_device_name(device), "device": str(device),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "flash_attention": not args.no_flash, "scale": args.scale,
            "seed": args.seed, "outdoor": args.outdoor, "normalize_coord": args.normalize_coord,
            "without_color": args.wo_color, "without_normal": args.wo_normal,
            "input_points": len(coord), "grid_points": grid_count,
            "feature_channels": point.feat.shape[1], "features_finite": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "forward_seconds": forward_seconds,
            "inference_and_postprocess_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "spconv-cu121", "torch-scatter", "open3d", "timm"]},
        }
    for name, color in colors.items():
        write_ply(args.output / f"{name}.ply", coord, color)
    # A static preview works through SSH; PLY files retain all points for 3D viewing.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample = np.random.default_rng(args.seed).choice(len(coord), min(len(coord), 30000), replace=False)
    fig = plt.figure(figsize=(6 * len(colors), 6))
    for i, (name, color) in enumerate(colors.items(), 1):
        ax = fig.add_subplot(1, len(colors), i, projection="3d")
        ax.scatter(*coord[sample].T, c=color[sample], s=0.4, linewidths=0, rasterized=True)
        ax.set_box_aspect(np.maximum(np.ptp(coord, axis=0), 1e-3))
        ax.set_title(name)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(args.output / "preview.png", dpi=160)
    plt.close(fig)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
