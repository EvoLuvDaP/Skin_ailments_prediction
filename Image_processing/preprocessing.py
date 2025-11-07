#!/usr/bin/env python3

import os, shutil, hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
SRC_DIR = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_split")
DST_DIR = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed")
TARGET_SIZE = 224
DEDUP = True
# ======================

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def resize_image(img, size):
    img = img.convert("RGB")
    img.thumbnail((size, size))
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    bg.paste(img, ((size - img.width)//2, (size - img.height)//2))
    return bg

def preprocess():
    if DST_DIR.exists():
        print(f"Removing old dataset: {DST_DIR}")
        shutil.rmtree(DST_DIR)

    hashes = set()
    for split_dir in sorted(SRC_DIR.iterdir()):
        if not split_dir.is_dir(): continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir(): continue
            outdir = DST_DIR / split_dir.name / cls_dir.name
            outdir.mkdir(parents=True, exist_ok=True)

            files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for f in tqdm(files, desc=f"{split_dir.name}/{cls_dir.name}"):
                try:
                    img = Image.open(f)
                except Exception:
                    continue

                if DEDUP:
                    h = sha256_of_file(f)
                    if h in hashes:
                        continue
                    hashes.add(h)

                out_path = outdir / f.name
                img = resize_image(img, TARGET_SIZE)
                img.save(out_path, format="JPEG", quality=95)

    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
