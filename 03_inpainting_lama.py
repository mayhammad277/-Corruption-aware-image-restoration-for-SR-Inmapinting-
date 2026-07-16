#!/usr/bin/env python3
"""
inpainting_lama.py

Standalone (non-Colab) version of "03_Inpainting_LaMa.ipynb".
Paper: A Corruption-Aware Image Restoration Framework for Spaceborne Optical
       Navigation

Pipeline:
    1. Generate spacecraft-aware masks for SPEED+ images
    2. Prepare a LaMa-format input directory (img.png + img_mask001.png)
    3. Run big-lama inference via subprocess (PYTHONPATH + predict.py)
    4. Produce qualitative figures
    5. Produce quantitative evaluation (PSNR / SSIM) + CSV
    6. Copy all outputs to a local results directory

Prerequisites:
    Run setup_environment.sh once to clone big-lama, install dependencies,
    download the big-lama checkpoint, and (optionally) patch the repo for
    compatibility with modern numpy / albumentations / torch. This script
    assumes that environment already exists; it does NOT pip install or
    git clone anything itself.

Example:
    python inpainting_lama.py \
        --lama-repo   /opt/lama_env/lama \
        --lama-ckpt   /opt/lama_env/lama/big-lama \
        --speedplus-dir /data/speed/speed \
        --project-dir   /data/space_restoration \
        --work-dir      /data/inpaint_work \
        --patch-lama --run-all
"""

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Mask-generation parameters (unchanged from the original notebook)
# ----------------------------------------------------------------------------
IMG_SIZE = 256
MASK_MAX_COVERAGE = 0.25
SC_OVERLAP_LIMIT = 0.40
MASK_N_MIN = 1
MASK_N_MAX = 2
MASK_MIN_RECT = 15
MASK_MAX_RECT = 70


# ----------------------------------------------------------------------------
# Spacecraft-aware mask generator
# ----------------------------------------------------------------------------
def get_spacecraft_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, sc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sc.sum() < 100:
        _, sc = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sc = cv2.morphologyEx(sc, cv2.MORPH_CLOSE, k, iterations=2)
    sc = cv2.morphologyEx(sc, cv2.MORPH_OPEN, k, iterations=1)
    return (sc > 0).astype(np.float32)


def sc_overlap_ratio(patch, sc_mask):
    sc_px = sc_mask.sum()
    return float((patch * sc_mask).sum() / sc_px) if sc_px > 0 else 0.0


def make_smart_mask(img_bgr, max_attempts=60):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    max_px = int(h * w * MASK_MAX_COVERAGE)
    sc_mask = get_spacecraft_mask(img_bgr)
    strategy = random.choice(["edge", "corner", "free"])

    for _ in range(random.randint(MASK_N_MIN, MASK_N_MAX)):
        if mask.sum() >= max_px:
            break
        placed = False
        for _ in range(max_attempts):
            rh = random.randint(MASK_MIN_RECT, MASK_MAX_RECT)
            rw = random.randint(MASK_MIN_RECT, MASK_MAX_RECT)
            if strategy == "edge":
                e = random.choice(["top", "bottom", "left", "right"])
                if e == "top":
                    top, left = 0, random.randint(0, w - rw)
                elif e == "bottom":
                    top, left = h - rh, random.randint(0, w - rw)
                elif e == "left":
                    top, left = random.randint(0, h - rh), 0
                else:
                    top, left = random.randint(0, h - rh), w - rw
            elif strategy == "corner":
                c = random.choice(["tl", "tr", "bl", "br"])
                top = 0 if c in ("tl", "tr") else h - rh
                left = 0 if c in ("tl", "bl") else w - rw
            else:
                top = random.randint(0, h - rh)
                left = random.randint(0, w - rw)
            patch = np.zeros((h, w), dtype=np.float32)
            patch[top : top + rh, left : left + rw] = 1.0
            if (
                (mask + patch).clip(0, 1).sum() <= max_px
                and sc_overlap_ratio(patch, sc_mask) <= SC_OVERLAP_LIMIT
            ):
                mask = np.clip(mask + patch, 0, 1)
                placed = True
                break
        if not placed:
            fb = MASK_MIN_RECT
            t, l = random.choice([0, h - fb]), random.choice([0, w - fb])
            mask[t : t + fb, l : l + fb] = 1.0

    return (mask * 255).astype(np.uint8)


def preview_masks(all_imgs, speedplus_dir, project_dir, n=4):
    """Save a qualitative preview of the automatic mask generator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_imgs = random.sample(all_imgs, min(n, len(all_imgs)))
    fig, axes = plt.subplots(len(preview_imgs), 4, figsize=(18, 4 * len(preview_imgs)))
    if len(preview_imgs) == 1:
        axes = [axes]
    for c, t in enumerate(["Original", "Spacecraft (red)", "Mask (white=inpaint)", "Input to LaMa"]):
        axes[0][c].set_title(t, fontsize=10, fontweight="bold")

    for row, fname in enumerate(preview_imgs):
        bgr = cv2.imread(os.path.join(speedplus_dir, fname))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sc = get_spacecraft_mask(bgr)
        msk = make_smart_mask(bgr)
        msk_f = msk.astype(np.float32) / 255.0
        sc_vis = rgb.astype(np.float32) / 255.0
        sc_vis[sc > 0, 0] = 1.0
        sc_vis[sc > 0, 1] *= 0.2
        sc_vis[sc > 0, 2] *= 0.2
        masked = rgb.copy()
        masked[msk > 0] = 0
        axes[row][0].imshow(rgb)
        axes[row][1].imshow(sc_vis.clip(0, 1))
        axes[row][2].imshow(msk, cmap="gray", vmin=0, vmax=255)
        axes[row][2].set_xlabel(
            "Cov: {:.1f}%  SC: {:.1f}%".format(msk_f.mean() * 100, sc_overlap_ratio(msk_f, sc) * 100),
            fontsize=8,
        )
        axes[row][3].imshow(masked)
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Spacecraft-Aware Mask Preview", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(project_dir, "mask_preview.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Mask preview saved to {out_path}")
    print(
        "Mask rules: max {}% coverage, max {}% SC overlap, {}-{} patches, max {}px".format(
            int(MASK_MAX_COVERAGE * 100),
            int(SC_OVERLAP_LIMIT * 100),
            MASK_N_MIN,
            MASK_N_MAX,
            MASK_MAX_RECT,
        )
    )


# ----------------------------------------------------------------------------
# Step: build LaMa input directory
# ----------------------------------------------------------------------------
def build_lama_input(speedplus_dir, gt_dir, lama_in, max_images=None):
    all_imgs = sorted(f for f in os.listdir(speedplus_dir) if f.endswith((".jpg", ".png")))
    print(f"Found {len(all_imgs)} SPEED+ images")

    run_imgs = all_imgs[:max_images] if max_images else all_imgs
    skipped = 0

    for fname in run_imgs:
        # Strip any 'Copy of ' prefix some cloud-sync tools add
        stem = os.path.splitext(fname)[0].replace("Copy of ", "").strip()
        bgr = cv2.imread(os.path.join(speedplus_dir, fname))
        if bgr is None:
            skipped += 1
            continue
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        msk = make_smart_mask(bgr)
        cv2.imwrite(os.path.join(gt_dir, stem + ".png"), bgr)
        cv2.imwrite(os.path.join(lama_in, stem + ".png"), bgr)
        cv2.imwrite(os.path.join(lama_in, stem + "_mask001.png"), msk)

    n = len([f for f in os.listdir(lama_in) if "_mask" not in f])
    print(f"Done: {n} image+mask pairs ready | skipped: {skipped}")
    print("Sample files:", sorted(os.listdir(lama_in))[:4])
    return all_imgs


# ----------------------------------------------------------------------------
# Step: patch the local lama repo / imgaug for modern numpy/albumentations/torch
# ----------------------------------------------------------------------------
def patch_lama_repo(lama_repo):
    import importlib.util

    # Locate imgaug.py inside whatever environment is active
    spec = importlib.util.find_spec("imgaug")
    if spec is None or spec.origin is None:
        print("imgaug not found on path; skipping imgaug patch")
    else:
        imgaug_path = spec.origin
        with open(imgaug_path, "r") as f:
            src = f.read()
        replacements = {
            'NP_FLOAT_TYPES = set(np.sctypes["float"])': "NP_FLOAT_TYPES = set([np.float16, np.float32, np.float64])",
            'NP_INT_TYPES = set(np.sctypes["int"])': "NP_INT_TYPES = set([np.int8, np.int16, np.int32, np.int64])",
            'NP_UINT_TYPES = set(np.sctypes["uint"])': "NP_UINT_TYPES = set([np.uint8, np.uint16, np.uint32, np.uint64])",
        }
        changed = False
        for old, new in replacements.items():
            if old in src:
                src = src.replace(old, new)
                changed = True
        if changed:
            with open(imgaug_path, "w") as f:
                f.write(src)
            print(f"Patched {imgaug_path}")
        else:
            print("imgaug.py already patched (or nothing to patch)")

    # Patch lama's aug.py
    aug_path = os.path.join(lama_repo, "saicinpainting", "training", "data", "aug.py")
    if os.path.exists(aug_path):
        with open(aug_path, "r") as f:
            src = f.read()
        if "DualIAATransform" in src:
            src = src.replace(
                "from albumentations import DualIAATransform, to_tuple",
                "from albumentations import DualTransform as DualIAATransform, to_tuple",
            )
            with open(aug_path, "w") as f:
                f.write(src)
            print(f"Patched {aug_path}")
        else:
            print("aug.py already patched (or nothing to patch)")
    else:
        print(f"WARNING: {aug_path} not found; skipping aug.py patch")

    # Patch trainers/__init__.py for torch.load(weights_only=False)
    trainer_path = os.path.join(lama_repo, "saicinpainting", "training", "trainers", "__init__.py")
    if os.path.exists(trainer_path):
        with open(trainer_path, "r") as f:
            src = f.read()
        if "weights_only=False" not in src:
            src = src.replace(
                "state = torch.load(path, map_location=map_location)",
                "state = torch.load(path, map_location=map_location, weights_only=False)",
            )
            with open(trainer_path, "w") as f:
                f.write(src)
        print("Patched:", "weights_only=False" in open(trainer_path).read())
    else:
        print(f"WARNING: {trainer_path} not found; skipping trainers patch")


# ----------------------------------------------------------------------------
# Step: run big-lama inference
# ----------------------------------------------------------------------------
def run_inference(lama_repo, lama_ckpt, lama_in, lama_out, python_exe=sys.executable):
    env = os.environ.copy()
    env["PYTHONPATH"] = lama_repo + os.pathsep + env.get("PYTHONPATH", "")

    predict_py = os.path.join(lama_repo, "bin", "predict.py")
    cmd = [
        python_exe,
        predict_py,
        f"model.path={lama_ckpt}",
        f"indir={lama_in}",
        f"outdir={lama_out}",
        "dataset.img_suffix=.png",
    ]
    print("> Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    out_files = sorted(f for f in os.listdir(lama_out) if f.endswith(".png"))
    print(f"Done: {len(out_files)} output images")
    print("Outputs:", out_files[:3])


# ----------------------------------------------------------------------------
# Step: qualitative results figure
# ----------------------------------------------------------------------------
def qualitative_results(gt_dir, lama_in, lama_out, vis_dir, n_show=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_files = sorted(f for f in os.listdir(lama_out) if f.endswith(".png"))
    if not out_files:
        print("No LaMa outputs found; skipping qualitative figure.")
        return
    print("Sample output filenames:", out_files[:3])

    n_show = min(n_show, len(out_files))
    samples = random.sample(out_files, n_show)

    fig, axes = plt.subplots(n_show, 4, figsize=(18, 4 * n_show))
    if n_show == 1:
        axes = [axes]
    for c, t in enumerate(["GT (original)", "Mask (white=inpainted)", "LaMa Output", "|GT - Output| error"]):
        axes[0][c].set_title(t, fontsize=11, fontweight="bold")

    for row, fname in enumerate(samples):
        stem = os.path.splitext(fname)[0].replace("_mask001", "")
        gt_path = os.path.join(gt_dir, stem + ".png")
        msk_path = os.path.join(lama_in, stem + "_mask001.png")
        out_path = os.path.join(lama_out, fname)
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        out = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
        diff = np.abs(gt.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
        mse = np.mean((gt.astype(np.float32) / 255 - out.astype(np.float32) / 255) ** 2)
        psnr = 10 * math.log10(1 / mse) if mse > 1e-10 else 100
        axes[row][0].imshow(gt)
        axes[row][1].imshow(msk, cmap="gray", vmin=0, vmax=255)
        axes[row][2].imshow(out)
        axes[row][2].set_xlabel("PSNR = {:.2f} dB".format(psnr), fontsize=9)
        axes[row][3].imshow(diff, cmap="hot", vmin=0, vmax=40)
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("big-lama Inpainting - SPEED+ Qualitative Results", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True)
    fig_path = os.path.join(vis_dir, "lama_qualitative.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to " + fig_path)


# ----------------------------------------------------------------------------
# Step: quantitative evaluation (PSNR / SSIM)
# ----------------------------------------------------------------------------
def quantitative_evaluation(gt_dir, lama_in, lama_out, project_dir):
    from skimage.metrics import structural_similarity as skssim

    out_files = sorted(f for f in os.listdir(lama_out) if f.endswith(".png"))
    results = []

    for fname in out_files:
        stem = os.path.splitext(fname)[0].replace("_mask001", "")
        gt_path = os.path.join(gt_dir, stem + ".png")
        msk_path = os.path.join(lama_in, stem + "_mask001.png")
        out_path = os.path.join(lama_out, fname)
        if not all(os.path.exists(p) for p in [gt_path, msk_path, out_path]):
            continue
        gt = cv2.imread(gt_path).astype(np.float32) / 255.0
        out = cv2.imread(out_path).astype(np.float32) / 255.0
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        baseline = gt.copy()
        baseline[msk > 0.5] = 0.0
        mse_out = np.mean((gt - out) ** 2)
        mse_base = np.mean((gt - baseline) ** 2)
        results.append(
            {
                "name": stem,
                "psnr_baseline": 10 * math.log10(1 / mse_base) if mse_base > 1e-10 else 100,
                "psnr_lama": 10 * math.log10(1 / mse_out) if mse_out > 1e-10 else 100,
                "ssim_baseline": skssim(baseline, gt, data_range=1.0, channel_axis=2),
                "ssim_lama": skssim(out, gt, data_range=1.0, channel_axis=2),
                "mask_coverage": float(msk.mean()),
            }
        )

    if not results:
        print("No results to evaluate; skipping quantitative evaluation.")
        return

    avg = {k: np.mean([r[k] for r in results]) for k in results[0] if k != "name"}
    print("=" * 58)
    print("  big-lama - SPEED+ Inpainting Evaluation")
    print("=" * 58)
    print("  {:<28} {:>10} {:>10}".format("Method", "PSNR (dB)", "SSIM"))
    print("-" * 52)
    print("  {:<28} {:>10.2f} {:>10.4f}".format("Masked Input (baseline)", avg["psnr_baseline"], avg["ssim_baseline"]))
    print("  {:<28} {:>10.2f} {:>10.4f}".format("big-lama", avg["psnr_lama"], avg["ssim_lama"]))
    print("=" * 58)
    print("  Avg mask coverage: {:.1f}%".format(avg["mask_coverage"] * 100))
    print("  Images evaluated : {}".format(len(results)))

    csv_path = os.path.join(project_dir, "inpaint_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print("CSV saved to " + csv_path)


# ----------------------------------------------------------------------------
# Step: copy outputs to project directory
# ----------------------------------------------------------------------------
def save_outputs(lama_out, project_dir, vis_dir):
    drive_out = os.path.join(project_dir, "inpaint", "lama_output")
    os.makedirs(drive_out, exist_ok=True)

    out_files = [f for f in os.listdir(lama_out) if f.endswith(".png")]
    for fname in out_files:
        shutil.copy(os.path.join(lama_out, fname), os.path.join(drive_out, fname))

    print("All outputs saved:")
    print(f"  Inpainted images : {drive_out} ({len(out_files)} files)")
    print(f"  Metrics CSV      : {os.path.join(project_dir, 'inpaint_results.csv')}")
    print(f"  Qualitative fig  : {os.path.join(vis_dir, 'lama_qualitative.png')}")
    print(f"  Mask preview     : {os.path.join(project_dir, 'mask_preview.png')}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Local (non-Colab) SPEED+ spacecraft inpainting pipeline using big-lama."
    )
    p.add_argument("--speedplus-dir", required=True, help="Directory with SPEED+ source images")
    p.add_argument("--project-dir", required=True, help="Output/project directory (figures, CSV, results)")
    p.add_argument("--work-dir", required=True, help="Scratch working directory (gt/lama_in/lama_out)")
    p.add_argument("--lama-repo", required=True, help="Path to a local checkout of https://github.com/advimman/lama")
    p.add_argument("--lama-ckpt", default=None, help="Path to big-lama checkpoint dir (default: <lama-repo>/big-lama)")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images processed (for quick tests)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible mask generation")
    p.add_argument("--n-preview", type=int, default=4, help="Number of images in the mask preview figure")
    p.add_argument("--n-qualitative", type=int, default=6, help="Number of images in the qualitative results figure")
    p.add_argument("--python-exe", default=sys.executable, help="Python executable to invoke predict.py with")

    p.add_argument("--patch-lama", action="store_true", help="Apply compatibility patches to the local lama repo before running")
    p.add_argument("--skip-mask-preview", action="store_true")
    p.add_argument("--skip-build", action="store_true", help="Skip building the LaMa input dir (reuse existing)")
    p.add_argument("--skip-inference", action="store_true", help="Skip running big-lama inference (reuse existing outputs)")
    p.add_argument("--skip-qualitative", action="store_true")
    p.add_argument("--skip-quantitative", action="store_true")
    p.add_argument("--skip-save", action="store_true")
    p.add_argument("--run-all", action="store_true", help="Run every step (equivalent to not passing any --skip-* flags)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    lama_ckpt = args.lama_ckpt or os.path.join(args.lama_repo, "big-lama")

    gt_dir = os.path.join(args.work_dir, "gt")
    lama_in = os.path.join(args.work_dir, "lama_in")
    lama_out = os.path.join(args.work_dir, "lama_out")
    vis_dir = os.path.join(args.project_dir, "visuals", "lama")

    for d in [gt_dir, lama_in, lama_out, vis_dir, args.project_dir]:
        os.makedirs(d, exist_ok=True)

    if args.patch_lama:
        patch_lama_repo(args.lama_repo)

    all_imgs = sorted(f for f in os.listdir(args.speedplus_dir) if f.endswith((".jpg", ".png")))
    if not all_imgs:
        print(f"ERROR: no .jpg/.png images found in {args.speedplus_dir}")
        sys.exit(1)

    if not args.skip_mask_preview:
        preview_masks(all_imgs, args.speedplus_dir, args.project_dir, n=args.n_preview)

    if not args.skip_build:
        build_lama_input(args.speedplus_dir, gt_dir, lama_in, max_images=args.max_images)

    if not args.skip_inference:
        run_inference(args.lama_repo, lama_ckpt, lama_in, lama_out, python_exe=args.python_exe)

    if not args.skip_qualitative:
        qualitative_results(gt_dir, lama_in, lama_out, vis_dir, n_show=args.n_qualitative)

    if not args.skip_quantitative:
        quantitative_evaluation(gt_dir, lama_in, lama_out, args.project_dir)

    if not args.skip_save:
        save_outputs(lama_out, args.project_dir, vis_dir)


if __name__ == "__main__":
    main()
