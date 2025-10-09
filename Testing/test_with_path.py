# Single-cell inference (ResNet50 + Custom Head)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
from collections import OrderedDict

# ---------------- CONFIG ----------------
DATA_DIR    = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed")
MODEL_PATH  = Path("weights_resnet50_head_custom/best.pth")
MODEL_NAME  = "resnet50"
IMG_SIZE    = 224
IMAGE_PATH  = Path("/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset_preprocessed/test/Nail_psoriasis/3241__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg")
TOPK        = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------

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

# --- determine class_names ---
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
        raise SystemExit("class_names not provided and test folder not found.")

print("Classes:", class_names)

# --- build model with SAME custom head ---
num_classes = len(class_names)
print("Building model:", MODEL_NAME, "num_classes:", num_classes)

backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
in_f = backbone.get_classifier().in_features

custom_head = nn.Sequential(
    nn.Linear(in_f, 512),
    nn.ELU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.2),
    nn.Linear(512, 128),
    nn.ELU(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(p=0.2),
    nn.Linear(128, num_classes)
)
backbone.fc = custom_head
model = backbone

# --- load checkpoint ---
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state" in checkpoint:
    state = checkpoint["model_state"]
elif "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
else:
    state = checkpoint
model.load_state_dict(state, strict=False)

model = model.to(DEVICE)
model.eval()
print("Checkpoint loaded successfully.")

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
