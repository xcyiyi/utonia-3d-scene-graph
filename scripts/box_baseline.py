"""Utonia semantic predictions + spatial clustering -> heuristic 3D boxes.

This is NOT a trained box detector. Reference boxes are read only after predictions
have been finalized. No ground-truth instance IDs enter the inference pipeline.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import utonia
from sklearn.cluster import DBSCAN

from infer import load_input, write_ply

EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def corners(box):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return np.asarray(box[:3]) + signs * np.asarray(box[3:6]) / 2


def iou3d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lower = np.maximum(a[:3] - a[3:6] / 2, b[:3] - b[3:6] / 2)
    upper = np.minimum(a[:3] + a[3:6] / 2, b[:3] + b[3:6] / 2)
    intersection = float(np.prod(np.maximum(upper - lower, 0)))
    union = float(np.prod(a[3:6]) + np.prod(b[3:6]) - intersection)
    return intersection / union if union > 0 else 0.0


def evaluate(predictions, references, threshold):
    used, matches = set(), []
    for pred in sorted(predictions, key=lambda x: -x["score"]):
        candidates = [(iou3d(pred["box"], ref["box"]), i) for i, ref in enumerate(references)
                      if i not in used and pred["label"] == ref["label"]]
        overlap, index = max(candidates, default=(0.0, -1))
        if index >= 0 and overlap >= threshold:
            used.add(index)
            matches.append(dict(prediction_id=pred["id"], reference_id=references[index]["id"], iou=overlap))
    tp = len(matches)
    return dict(iou_threshold=threshold, true_positive=tp,
                false_positive=len(predictions) - tp, false_negative=len(references) - tp,
                precision=tp / len(predictions) if predictions else 0.0,
                recall=tp / len(references) if references else 0.0, matches=matches)


def make_boxes(coord, labels, confidence, class_names, args):
    predictions, assignment = [], np.full(len(coord), -1, dtype=np.int32)
    # Ignore wall/floor (ScanNet20 indices 0 and 1); retain its 18 object classes.
    for class_id in range(2, 20):
        indices = np.flatnonzero((labels == class_id) & (confidence >= args.min_confidence))
        if len(indices) < args.min_points:
            continue
        clusters = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=8).fit_predict(coord[indices])
        for cluster_id in np.unique(clusters[clusters >= 0]):
            members = indices[clusters == cluster_id]
            if len(members) < args.min_points:
                continue
            low, high = coord[members].min(0), coord[members].max(0)
            size = high - low
            if np.any(size <= 1e-6):
                continue
            center = (low + high) / 2
            index = len(predictions)
            assignment[members] = index
            predictions.append(dict(id=index, label=class_names[class_id],
                                    class_id=class_id - 2, semantic_class_id=class_id,
                                    score=float(confidence[members].mean()),
                                    points=len(members), box=[*center.tolist(), *size.tolist(), 0.0],
                                    min_xyz=low.tolist(), max_xyz=high.tolist()))
    return predictions, assignment


def visualize(coord, rgb, predicted, references, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    selection = np.random.default_rng(37).choice(len(coord), min(len(coord), 25000), replace=False)
    # Remove only high ceiling points from the preview for a clearer room view.
    # Inference, exported point clouds and boxes retain all points.
    visible = selection[coord[selection, 2] < np.quantile(coord[:, 2], 0.99) - 0.08]
    shown = coord[visible]
    point_colors = [f"rgb({r},{g},{b})" for r, g, b in (rgb[visible] * 255).astype(np.uint8)]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=(f"Utonia clustering baseline: {len(predicted)} boxes",
                                        f"Reference annotations: {len(references)} boxes"))
    for column, (boxes, color) in enumerate(((predicted, "#ff6347"), (references, "#26d18b")), 1):
        fig.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="markers",
                                  marker=dict(size=1.6, color=point_colors), showlegend=False,
                                  hoverinfo="skip"), row=1, col=column)
        for item in boxes:
            vertices, xs, ys, zs = corners(item["box"]), [], [], []
            for i, j in EDGES:
                xs.extend([vertices[i, 0], vertices[j, 0], None])
                ys.extend([vertices[i, 1], vertices[j, 1], None])
                zs.extend([vertices[i, 2], vertices[j, 2], None])
            text = f"#{item['id']} {item['label']}"
            if "score" in item:
                text += f" | mean semantic probability {item['score']:.3f}"
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=text,
                                      line=dict(color=color, width=5), text=text,
                                      hoverinfo="text", showlegend=False), row=1, col=column)
            c = item["box"]
            fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2] + c[5] / 2], mode="text",
                                      text=[f"{item['id']}: {item['label']}"],
                                      textfont=dict(size=10, color=color), showlegend=False,
                                      hoverinfo="skip"), row=1, col=column)
    scene = dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                 camera=dict(eye=dict(x=1.3, y=-1.5, z=1.7)))
    fig.update_layout(title="ScanNet scene0000_00 — heuristic boxes, not a trained detector",
                      scene=scene, scene2=scene, height=850, margin=dict(l=0, r=0, b=20, t=90))
    fig.write_html(out / "boxes.html", include_plotlyjs=True, full_html=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    figure = plt.figure(figsize=(16, 8))
    for index, (boxes, color, title) in enumerate(((predicted, "tomato", "Predicted heuristic boxes"),
                                                  (references, "seagreen", "Reference boxes")), 1):
        ax = figure.add_subplot(1, 2, index, projection="3d")
        ax.scatter(*shown.T, c=rgb[visible], s=1, depthshade=False, linewidths=0)
        for item in boxes:
            vertices = corners(item["box"])
            ax.add_collection3d(Line3DCollection([vertices[[i, j]] for i, j in EDGES], colors=color, linewidths=1))
            c = item["box"]
            ax.text(c[0], c[1], c[2] + c[5] / 2, str(item["id"]), color="black", fontsize=7)
        ax.set_box_aspect(np.maximum(np.ptp(coord, axis=0), .01))
        ax.view_init(elev=55, azim=-65)
        ax.set_title(f"{title} ({len(boxes)})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
    figure.suptitle("ScanNet scene0000_00 | Utonia + semantic head + DBSCAN | NOT a trained detector")
    figure.tight_layout()
    figure.savefig(out / "boxes_preview.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/scannet_mini/scene0000_00.npz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/scannet_mini/annotations.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scannet_mini_boxes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.18, help="DBSCAN radius in meters")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse cached semantic outputs for this exact input/scale")
    args = parser.parse_args()
    if args.eps <= 0 or args.scale <= 0 or args.min_samples < 1 or args.min_points < 1 or not 0 <= args.min_confidence <= 1:
        parser.error("Invalid clustering or scale parameter")
    import hashlib
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    cache_key = dict(input_sha256=input_hash, scale=args.scale, seed=37)
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_input(args.input)
    coord, rgb = data["coord"].copy(), data["color"].copy() / 255
    official = runpy.run_path(str(ROOT / "demo/2_sem_seg.py"))
    cache = args.output / "semantic_predictions.npz"
    if args.reuse_predictions:
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["cache_key"])) != cache_key:
                raise ValueError("Cached predictions do not match this input and scale")
            labels, confidence = saved["labels"], saved["confidence"]
        profile = json.loads((args.output / "inference_profile.json").read_text())
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; run outside the GPU-restricted sandbox")
        torch.cuda.set_device(device)
        utonia.utils.set_seed(37)
        model = utonia.load(str(ROOT / "checkpoints/utonia.pth")).to(device).eval()
        ckpt = utonia.load(str(ROOT / "checkpoints/utonia_linear_prob_head_sc.pth"), ckpt_only=True)
        head = official["SegHead"](**ckpt["config"]).to(device).eval()
        head.load_state_dict(ckpt["state_dict"])
        point = utonia.transform.default(args.scale)(data)
        grid_points = len(point["coord"])
        print(f"Utonia inference: {len(coord)} points, {grid_points} grid points", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            point = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            point = model(point)
            while "pooling_parent" in point:
                parent = point.pop("pooling_parent")
                parent.feat = torch.cat([parent.feat, point.feat[point.pop("pooling_inverse")]], dim=-1)
                point = parent
            logits = head(point.feat)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite predictions")
            scores, categories = logits.softmax(-1).max(-1)
            labels = categories[point.inverse].cpu().numpy()
            confidence = scores[point.inverse].cpu().numpy()
            torch.cuda.synchronize(device)
            profile = dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                           cuda=torch.version.cuda, grid_points=grid_points,
                           forward_and_head_seconds=time.perf_counter() - start,
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
        np.savez_compressed(cache, labels=labels, confidence=confidence, cache_key=json.dumps(cache_key))
        (args.output / "inference_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
        del point, model, head
        torch.cuda.empty_cache()
    start = time.perf_counter()
    predictions, assignment = make_boxes(coord, labels, confidence, official["CLASS_LABELS_20"], args)
    cluster_seconds = time.perf_counter() - start
    boxes = np.asarray([item["box"] for item in predictions], dtype=np.float32).reshape(-1, 7)
    np.save(args.output / "boxes.npy", boxes)
    np.save(args.output / "instance_assignment.npy", assignment)
    payload = dict(method="utonia_semantic_dbscan_baseline", trained_detection_head=False,
                   coordinate_frame="ScanNet axis-aligned", box_format=["cx", "cy", "cz", "dx", "dy", "dz", "yaw"],
                   units="meters; yaw in radians", center_origin="geometric center",
                   score_definition="Mean point semantic probability; not calibrated detection confidence",
                   boxes=predictions)
    (args.output / "boxes.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "boxes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "class_id", "score", "points", "cx", "cy", "cz", "dx", "dy", "dz", "yaw"])
        writer.writerows([[p["id"], p["label"], p["class_id"], p["score"], p["points"], *p["box"]] for p in predictions])

    # Reference annotations enter the pipeline only after box predictions are saved.
    annotations = json.loads(args.annotations.read_text())
    references = annotations["boxes"]
    inverse_transform = np.linalg.inv(np.asarray(annotations["original_to_aligned"]))
    original_boxes = boxes.copy()
    original_boxes[:, :3] = boxes[:, :3] @ inverse_transform[:3, :3].T + inverse_transform[:3, 3]
    original_boxes[:, 6] = np.arctan2(inverse_transform[1, 0], inverse_transform[0, 0])
    np.save(args.output / "boxes_original.npy", original_boxes)
    (args.output / "coordinate_transform.json").write_text(json.dumps(dict(
        original_to_aligned=annotations["original_to_aligned"],
        aligned_to_original=inverse_transform.tolist(),
        boxes_npy_frame="ScanNet axis-aligned",
        boxes_original_npy_frame="original scene0000_00.bin coordinates; yaw around +Z",
        box_origin="geometric center", rotation_convention="local box X rotated counterclockwise by yaw about +Z"), indent=2) + "\n")
    np.save(args.output / "reference_boxes.npy", np.asarray([b["box"] for b in references], dtype=np.float32))
    report = dict(method=payload["method"], trained_detection_head=False,
                  scope="single public demo scene; not a held-out ScanNet benchmark evaluation",
                  reference_source="MMDetection3D test metadata; same scene with axis alignment",
                  input=str(args.input), input_points=len(coord), predicted_boxes=len(predictions),
                  reference_boxes=len(references), predicted_classes=dict(Counter(p["label"] for p in predictions)),
                  parameters=dict(scale=args.scale, eps_meters=args.eps, min_samples=args.min_samples,
                                  min_points=args.min_points, min_confidence=args.min_confidence),
                  inference=profile, clustering_seconds=cluster_seconds,
                  evaluation=[evaluate(predictions, references, t) for t in (0.25, 0.5)])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_ply(args.output / "input_aligned.ply", coord, rgb)
    semantic_rgb = np.asarray(official["CLASS_COLOR_20"])[labels] / 255
    write_ply(args.output / "semantic.ply", coord, semantic_rgb)
    visualize(coord, rgb, predictions, references, args.output)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Boxes and interactive comparison: {args.output}")


if __name__ == "__main__":
    main()
