#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda run -n Utonia python --version >/dev/null 2>&1; then
    conda create -n Utonia python=3.10 pip setuptools=75.8.0 -y
fi
conda activate Utonia
python -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-inference.txt \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
flash_wheel='flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
if [[ -f "wheels/$flash_wheel" ]]; then
    python -m pip install --no-deps "wheels/$flash_wheel"
else
    python -m pip install --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$flash_wheel"
fi
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python scripts/download_assets.py
