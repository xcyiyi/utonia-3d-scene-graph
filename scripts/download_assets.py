"""Download official inference assets into the project (resumable HF cache)."""
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "fb39b40d33d215d29d1d4717ffb86287f4700c4d"

if __name__ == "__main__":
    for filename in ("utonia.pth", "utonia_linear_prob_head_sc.pth"):
        print(hf_hub_download(
            repo_id="Pointcept/Utonia", filename=filename,
            local_dir=ROOT / "checkpoints", revision=MODEL_REVISION,
        ))
    print(hf_hub_download(
        repo_id="pointcept/demo", repo_type="dataset", filename="sample1.npz",
        local_dir=ROOT / "data",
    ))
