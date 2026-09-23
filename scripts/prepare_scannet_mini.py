"""Download and prepare the public MMDetection3D ScanNet demo scene."""
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/scannet_mini"
ASSETS = {
    "scene0000_00.bin": (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/scannet/scene0000_00.bin",
        "87874538e4fcceed168bf25118f7bb82b8badf3ce20d5fce3a1fd1132f19ab77",
    ),
    "scannet_infos.pkl": (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/tests/data/scannet/scannet_infos.pkl",
        "80708a2e50137e7d4f679b85c889e56926664949599cc5e5c3a7e1924470ec43",
    ),
}
CLASSES = ["cabinet", "bed", "chair", "sofa", "table", "door", "window",
           "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator",
           "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"]


class BasicUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Refusing pickle global: {module}.{name}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (url, sha) in ASSETS.items():
        path = DATA / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response:
                temporary.write_bytes(response.read())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Unexpected upstream content: {name}")
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"SHA256 mismatch: {path}")
        provenance[name] = dict(url=url, sha256=sha, bytes=path.stat().st_size)
    with (DATA / "scannet_infos.pkl").open("rb") as stream:
        info = BasicUnpickler(stream).load()["data_list"][0]
    assert info["lidar_points"]["lidar_path"] == "scene0000_00.bin"
    raw = np.fromfile(DATA / "scene0000_00.bin", dtype=np.float32).reshape(-1, 6)
    assert len(raw) == 40684 and np.isfinite(raw).all()
    alignment = np.asarray(info["axis_align_matrix"], dtype=np.float64)
    coord = (raw[:, :3] @ alignment[:3, :3].T + alignment[:3, 3]).astype(np.float32)
    np.savez_compressed(DATA / "scene0000_00.npz", coord=coord, color=raw[:, 3:],
                        normal=np.zeros_like(coord))
    boxes = []
    for index, obj in enumerate(info["instances"]):
        box = obj["bbox_3d"] + [0.0]
        center, size = np.asarray(box[:3]), np.asarray(box[3:6])
        count = int(np.all((coord >= center - size / 2) & (coord <= center + size / 2), axis=1).sum())
        boxes.append(dict(id=index, box=box, class_id=obj["bbox_label_3d"],
                          label=CLASSES[obj["bbox_label_3d"]], enclosed_points=count))
    annotations = dict(scene="scene0000_00", box_format="cx,cy,cz,dx,dy,dz,yaw",
                       center_origin="geometric center", units="meters; yaw in radians",
                       coordinate_frame="ScanNet axis-aligned", classes=CLASSES,
                       original_to_aligned=alignment.tolist(), boxes=boxes)
    (DATA / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    manifest = dict(description="One public ScanNet demo scene, not the full benchmark dataset",
                    scene="scene0000_00", points=len(coord), reference_boxes=len(boxes),
                    annotations_source="MMDetection3D test metadata for the same scene",
                    sources=provenance, normals="absent; zero-filled",
                    evaluation_scope="Single-scene sanity check; no dataset-level AP claim")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
