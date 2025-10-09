# Single-cell inference (ResNet50) — robust to missing class_names
# Edit CONFIG below, then run cell.

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
from collections import OrderedDict
import sys

# ---------------- CONFIG (edit only) ----------------
DATA_DIR    = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed")  # root with train/val/test
MODEL_PATH  = Path("weights_resnet50/best.pth")   # your saved checkpoint
MODEL_NAME  = "resnet50"                          # backbone you trained
IMG_SIZE    = 224                                 # image size used in training
IMAGE_PATH  = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed/test/SJS-TEN/Crop-0350_100400sjs-ten-nail-52__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg")  # <-- set this to the image you want to predict
TOPK        = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------

# --- utilities ---
def _show_pil(pil_img, title=None, figsize=(5,5)):
    arr = np.array(pil_img).astype(np.float32)/255.0
    plt.figure(figsize=figsize)
    plt.imshow(arr)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# --- transforms (must match training pipeline) ---
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- determine class_names: try checkpoint -> else ImageFolder test -> else error ---
class_names = None
if MODEL_PATH.exists():
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt:
        class_names = ckpt["classes"]
        print("Loaded class names from checkpoint.")
if class_names is None:
    test_folder = DATA_DIR / "test"
    if test_folder.exists():
        ds_tmp = datasets.ImageFolder(str(test_folder), transform=test_tf)
        class_names = ds_tmp.classes
        print("Loaded class names from test folder (ImageFolder).")
    else:
        raise SystemExit("class_names not provided and test folder not found. Set DATA_DIR or embed class_names manually.")

print("Classes:", class_names)

# --- build model and load checkpoint robustly ---
num_classes = len(class_names)
print("Building model:", MODEL_NAME, "num_classes:", num_classes)
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

def load_checkpoint_adapt(model, ckpt_path, num_classes, dropout_p=0.4, strict=False):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # extract state dict if wrapped
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # normalize keys and collect fc keys
    new_state = OrderedDict()
    for k,v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    # detect sequential fc saved as fc.0..., fc.1...
    saved_fc_keys = [k for k in new_state.keys() if k.startswith("fc.")]
    has_fc1 = any(k.startswith("fc.1") for k in saved_fc_keys)
    has_fc_weight = any(k == "fc.weight" for k in new_state.keys())

    if has_fc1 and not has_fc_weight:
        # adapt model.fc to be Sequential(Dropout, Linear) so keys match fc.0 / fc.1
        print("Checkpoint uses Sequential fc (fc.1.*). Replacing model.fc with Sequential(Dropout, Linear).")
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_f = model.fc.in_features
        else:
            in_f = getattr(model, "num_features", None)
        if in_f is None:
            raise RuntimeError("Cannot determine fc in_features automatically. Construct model head manually.")
        model.fc = torch.nn.Sequential(torch.nn.Dropout(p=dropout_p), torch.nn.Linear(in_f, num_classes))

    # load state dict
    load_result = model.load_state_dict(new_state, strict=strict)
    if not strict:
        missing = load_result.missing_keys
        unexpected = load_result.unexpected_keys
        if missing:
            print("Missing keys after load:", missing)
        if unexpected:
            print("Unexpected keys after load:", unexpected)
    print("Checkpoint loaded:", ckpt_path)
    return model

model = load_checkpoint_adapt(model, MODEL_PATH, num_classes=num_classes, dropout_p=0.4, strict=False)
model = model.to(DEVICE)
model.eval()

# --- sanity check image path ---
if not IMAGE_PATH.exists():
    raise SystemExit(f"Image not found: {IMAGE_PATH}")

# --- inference ---
pil_img = Image.open(str(IMAGE_PATH)).convert("RGB")
input_tensor = test_tf(pil_img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    out = model(input_tensor)
    probs = F.softmax(out, dim=1).cpu().numpy().squeeze(0)

topk_idx = probs.argsort()[::-1][:TOPK]
topk_probs = probs[topk_idx]
topk_names = [class_names[i] for i in topk_idx]

# display
title = f"Pred: {topk_names[0]} ({topk_probs[0]*100:.2f}%)"
if IMAGE_PATH.parts and "test" in IMAGE_PATH.parts:
    # infer GT if image_path is inside .../test/<class>/
    try:
        idx = IMAGE_PATH.parts.index("test")
        if idx+1 < len(IMAGE_PATH.parts):
            gt_name = IMAGE_PATH.parts[idx+1]
            title = f"GT: {gt_name}\n" + title
    except Exception:
        pass

_show_pil(pil_img, title=title)
print("Top-{} predictions:".format(TOPK))
for name, p in zip(topk_names, topk_probs):
    print(f"  {name:25s} {p*100:6.2f}%")
