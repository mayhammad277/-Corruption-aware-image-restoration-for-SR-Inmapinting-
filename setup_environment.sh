#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup for the SPEED+ / big-lama inpainting pipeline.
# Run this once (locally or in a fresh venv/conda env) before running
# inpainting_lama.py. Re-run is safe (idempotent-ish) but not required.
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh /path/to/install/dir
# ---------------------------------------------------------------------------
set -euo pipefail

INSTALL_DIR="${1:-$(pwd)/lama_env}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "> Cloning big-lama repo into $INSTALL_DIR/lama"
if [ ! -d "lama" ]; then
    git clone https://github.com/advimman/lama.git
fi

echo "> Installing Python dependencies"
pip install --upgrade pip
pip install wldhx.yadisk-direct
pip uninstall --yes --quiet osqp || true
pip install -U scikit-survival
pip uninstall -y kornia || true
pip install kornia --no-dependencies
pip install kornia-rs
pip install pytorch-lightning
pip install hydra-core
pip install webdataset
pip install torch torchvision torchaudio torchtext
pip install -r lama/requirements.txt --quiet
pip install wget --quiet
pip uninstall -y albumentations || true
pip install albumentations==0.5.2 --quiet
pip uninstall -y opencv-python opencv-contrib-python || true
pip install opencv-python
pip install scikit-image tqdm matplotlib numpy opencv-python

echo "> Downloading big-lama checkpoint"
if [ ! -d "lama/big-lama" ]; then
    curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
    unzip -o big-lama.zip -d lama/
fi

echo "> Setup complete."
echo "  LAMA_REPO   = $INSTALL_DIR/lama"
echo "  LAMA_CKPT   = $INSTALL_DIR/lama/big-lama"
echo ""
echo "Next: run inpainting_lama.py, e.g."
echo "  python inpainting_lama.py \\"
echo "      --lama-repo $INSTALL_DIR/lama \\"
echo "      --lama-ckpt $INSTALL_DIR/lama/big-lama \\"
echo "      --speedplus-dir /path/to/speed/speed \\"
echo "      --project-dir /path/to/space_restoration \\"
echo "      --work-dir /path/to/inpaint_work \\"
echo "      --patch-lama --run-all"
