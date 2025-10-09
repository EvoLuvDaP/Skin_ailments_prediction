# preprocess_minimal_fixed.py

from pathlib import Path
import random, hashlib, json
from math import floor
from PIL import Image
import numpy as np

# -------------------------
# CONFIGURE HERE
# -------------------------
SRC = Path(r"/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset")             # source folder
DST = Path(r"/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed")# destination
TARGET_SIZE = 300
RESIZE_METHOD = "pad"   # "pad" or "center_crop"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42
COPY_FILES = True
DEDUP = True
DRYRUN = False
# -------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path): return p.is_file() and p.suffix.lower() in VALID_EXTS

def safe_open_image(path: Path):
    try:
        with Image.open(path) as im: im.verify()
        im = Image.open(path); im.load(); return im
    except Exception:
        return None

def compute_hash(path: Path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def resize_with_aspect(img: Image.Image, target_size: int, method: str = "pad"):
    if img.mode != "RGB": img = img.convert("RGB")
    w, h = img.size
    if method == "center_crop":
        ratio = max(target_size / w, target_size / h)
        nw, nh = int(w * ratio + 0.5), int(h * ratio + 0.5)
        img_resized = img.resize((nw, nh), Image.LANCZOS)
        left = (nw - target_size) // 2; top = (nh - target_size) // 2
        return img_resized.crop((left, top, left + target_size, top + target_size))
    else:
        img.thumbnail((target_size, target_size), Image.LANCZOS)
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        left = (target_size - img.width) // 2; top = (target_size - img.height) // 2
        new_img.paste(img, (left, top)); return new_img

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def deterministic_split(items, tr, vr, te, seed):
    n = len(items)
    if n == 0: return [], [], []
    rnd = random.Random(seed); items_shuf = items[:]; rnd.shuffle(items_shuf)
    n_train = int(floor(n * tr)); n_val = int(floor(n * vr)); n_test = n - n_train - n_val
    if n_train == 0 and n >= 1:
        n_train = 1
        if n_test > 0: n_test -= 1
        elif n_val > 0: n_val -= 1
    t1 = n_train; t2 = n_train + n_val
    return items_shuf[:t1], items_shuf[t1:t2], items_shuf[t2:]

def discover_layout(src: Path):
    """Detect if source already has train/val/test split."""
    lower_children = {p.name.lower(): p for p in src.iterdir() if p.is_dir()}
    has_splits = all(k in lower_children for k in ("train","val","test"))
    return "split" if has_splits else "unsplit"

def process_existing_splits(src: Path, dst: Path, seen_hashes, copy_files, dedup, dryrun):
    manifest = {"train": [], "val": [], "test": []}
    stats = {}
    for split in ("train","val","test"):
        split_src = src / split
        if not split_src.exists(): continue
        for cls in sorted([p for p in split_src.iterdir() if p.is_dir()], key=lambda x: x.name):
            imgs = [p for p in cls.iterdir() if is_image_file(p)]
            valid = []
            for p in imgs:
                im = safe_open_image(p)
                if im is None:
                    print(f"Skipping corrupted: {p}")
                    continue
                if dedup:
                    h = compute_hash(p)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                valid.append(p)
            stats.setdefault(cls.name, {"total":0,"train":0,"val":0,"test":0})
            stats[cls.name]["total"] += len(valid)
            # process each valid image into dst/<split>/<class>
            out_class = dst / split / cls.name; ensure_dir(out_class)
            for p in valid:
                if dryrun:
                    manifest[split].append({"filepath": str((out_class / p.name).resolve()), "class": cls.name})
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                proc = resize_with_aspect(im, TARGET_SIZE, method=RESIZE_METHOD)
                out_path = out_class / p.name
                if out_path.exists():
                    base = p.stem; ext = p.suffix.lower(); i = 1
                    while (out_class / f"{base}_{i}{ext}").exists(): i+=1
                    out_path = out_class / f"{base}_{i}{ext}"
                proc.save(out_path, quality=95)
                if not copy_files:
                    try: p.unlink()
                    except Exception: pass
                manifest[split].append({"filepath": str(out_path.resolve()), "class": cls.name})
            # count split counts
            stats[cls.name][split] += len(valid)
    return manifest, stats

def process_unsplit(src: Path, dst: Path, seen_hashes, copy_files, dedup, dryrun):
    manifest = {"train": [], "val": [], "test": []}
    stats = {}
    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()], key=lambda x: x.name)
    for cls in class_dirs:
        imgs = [p for p in cls.iterdir() if is_image_file(p)]
        valid = []
        for p in imgs:
            im = safe_open_image(p)
            if im is None:
                print(f"Skipping corrupted: {p}")
                continue
            if dedup:
                h = compute_hash(p)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            valid.append(p)
        tr, va, te = deterministic_split(valid, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED + hash(cls.name) & 0xffffffff)
        stats[cls.name] = {"total": len(valid), "train": len(tr), "val": len(va), "test": len(te)}
        for split_name, lst in (("train",tr),("val",va),("test",te)):
            out_class = dst / split_name / cls.name; ensure_dir(out_class)
            for p in lst:
                if dryrun:
                    manifest[split_name].append({"filepath": str((out_class / p.name).resolve()), "class": cls.name})
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                proc = resize_with_aspect(im, TARGET_SIZE, method=RESIZE_METHOD)
                out_path = out_class / p.name
                if out_path.exists():
                    base = p.stem; ext = p.suffix.lower(); i = 1
                    while (out_class / f"{base}_{i}{ext}").exists(): i+=1
                    out_path = out_class / f"{base}_{i}{ext}"
                proc.save(out_path, quality=95)
                if not copy_files:
                    try: p.unlink()
                    except Exception: pass
                manifest[split_name].append({"filepath": str(out_path.resolve()), "class": cls.name})
    return manifest, stats

def compute_mean_std(manifest_train):
    mean = np.zeros(3, dtype=np.float64); sq = np.zeros(3, dtype=np.float64); count = 0
    for row in manifest_train:
        p = Path(row["filepath"])
        try:
            im = Image.open(p).convert("RGB")
            arr = np.asarray(im, dtype=np.float32)/255.0
            pixels = arr.shape[0]*arr.shape[1]
            mean += arr.reshape(-1,3).sum(axis=0)
            sq += (arr.reshape(-1,3)**2).sum(axis=0)
            count += pixels
        except Exception:
            continue
    if count>0:
        mean = (mean/count).tolist()
        var = (sq/count) - np.array(mean)**2
        std  = np.sqrt(np.maximum(var,1e-12)).tolist()
    else:
        mean = [0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    return mean, std

def write_csvs_and_stats(dst: Path, manifest, stats, mean, std, src):
    import csv
    for split in ("train","val","test"):
        csv_path = dst / f"{split}.csv"
        with open(csv_path,"w",newline="",encoding="utf8") as f:
            writer = csv.writer(f); writer.writerow(["filepath","class"])
            for r in manifest[split]:
                writer.writerow([r["filepath"], r["class"]])
    stats_out = {
        "source": str(src.resolve()), "destination": str(dst.resolve()),
        "target_size": TARGET_SIZE, "resize_method": RESIZE_METHOD,
        "ratios": {"train": TRAIN_RATIO,"val":VAL_RATIO,"test":TEST_RATIO},
        "classes": stats, "train_mean": mean, "train_std": std
    }
    with open(dst / "dataset_stats.json","w",encoding="utf8") as f: json.dump(stats_out,f,indent=2)
    print("Wrote CSVs and dataset_stats.json")

def main():
    print("Start preprocess (fixed). SRC:", SRC, "DST:", DST)
    ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True)
    if not SRC.exists() or not SRC.is_dir():
        print("Invalid SRC"); return
    ensure_dir(DST)
    layout = discover_layout(SRC)
    print("Detected layout:", layout)
    seen_hashes = set()
    if DRYRUN:
        print("DRYRUN: printing planned actions only")
    if layout == "split":
        manifest, stats = process_existing_splits(SRC, DST, seen_hashes, COPY_FILES, DEDUP, DRYRUN)
    else:
        manifest, stats = process_unsplit(SRC, DST, seen_hashes, COPY_FILES, DEDUP, DRYRUN)
    mean, std = compute_mean_std(manifest["train"])
    write_csvs_and_stats(DST, manifest, stats, mean, std, SRC)
    print("Done.")

if __name__ == "__main__":
    main()
