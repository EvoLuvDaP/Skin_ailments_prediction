#!/usr/bin/env python3

import os, shutil, random
from pathlib import Path

# === CONFIGURATION ===
SRC_DIR = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Raw_Images")
DST_DIR = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_split")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1
SEED = 42
# ======================

def split_dataset():
    random.seed(SEED)
    if DST_DIR.exists():
        print(f"Removing old dataset: {DST_DIR}")
        shutil.rmtree(DST_DIR)
    (DST_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DST_DIR / "val").mkdir(parents=True, exist_ok=True)
    (DST_DIR / "test").mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])
    for cls in classes:
        print(f"Processing class: {cls.name}")
        images = sorted([f for f in cls.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }
        for split, files in splits.items():
            outdir = DST_DIR / split / cls.name
            outdir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, outdir / f.name)
    print("✅ Dataset split complete.")

if __name__ == "__main__":
    split_dataset()
