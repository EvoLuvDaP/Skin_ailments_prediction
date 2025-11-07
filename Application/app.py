from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


MODEL_PATH = Path("/home/phamtiendat/Documents/ComputerVision/weights_resnet50_head_custom/best.pth")

CLASS_NAMES = ['Nail_psoriasis', 'SJS-TEN', 'Unknown_Normal', 'Vitiligo', 'acne', 'hyperpigmentation']
NUM_CLASSES = len(CLASS_NAMES)

DATA_DIR = Path.cwd()
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3
IMG_SIZE = 224
# -----------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.secret_key = 'replace-me-with-a-very-secure-key'


def load_custom_model(ckpt_path, num_classes):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_f = backbone.fc.in_features
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
    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(DEVICE)
    model.eval()
    return model

CLASS_DESCRIPTIONS = {
    "Nail_psoriasis": "Psoriatic changes in nails — may present with pitting, discoloration, or onycholysis.",
    "SJS-TEN": "Severe skin reaction (Stevens-Johnson / Toxic Epidermal Necrolysis) — urgent clinical evaluation required.",
    "Unknown_Normal": "Appears to be normal skin or a condition not recognized by the model.",
    "Vitiligo": "Loss of pigment in patches on the skin. Consult a dermatologist for treatment options.",
    "acne": "Inflammatory lesions (pimples, pustules). Treatments available; consult clinician for severe cases.",
    "hyperpigmentation": "Areas of increased pigment; causes include post-inflammatory changes, sun exposure, or hormonal factors.",
}

model = None
if MODEL_PATH.exists():
    try:
        model = load_custom_model(MODEL_PATH, NUM_CLASSES)
        print(" Model loaded successfully. Classes:", CLASS_NAMES)
    except Exception as e:
        print(f" Failed to load model: {e}")
        model = None
else:
    print(f" MODEL_PATH {MODEL_PATH} does not exist.")


def preprocess_with_ratio(img_pil: Image.Image):
    # Calculate scale to resize longer side to IMG_SIZE
    width, height = img_pil.size
    if width > height:
        new_width = IMG_SIZE
        new_height = int(IMG_SIZE * height / width)
    else:
        new_height = IMG_SIZE
        new_width = int(IMG_SIZE * width / height)
    
    # Resize preserving aspect ratio
    img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # Pad to square with black (0) padding
    padded_img = ImageOps.pad(img_resized, (IMG_SIZE, IMG_SIZE), color=(0, 0, 0), centering=(0.5, 0.5))
    
    # Transforms: to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(padded_img), padded_img.size, (IMG_SIZE - new_width) // 2, (IMG_SIZE - new_height) // 2  # Return tensor, padded size, and padding offsets

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

def predict_from_image(img_pil: Image.Image):
    if model is None:
        raise RuntimeError("Model not loaded.")
    x, _, _, _ = preprocess_with_ratio(img_pil)  # Get tensor
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    topk_idx = probs.argsort()[::-1][:TOP_K]
    results = [(CLASS_NAMES[i], float(probs[i])) for i in topk_idx]
    return results, int(topk_idx[0])

# Grad-CAM with Alignment Fix 
def gradcam_on_image(model, img_pil, target_class):
    model.eval()
    x, padded_size, pad_left, pad_top = preprocess_with_ratio(img_pil)  # Get padded tensor and padding info
    x = x.unsqueeze(0).to(DEVICE)

    gradients = []
    activations = []

    def save_activation(module, input, output):
        activations.append(output.detach())

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.layer4[-1].conv3
    h1 = target_layer.register_forward_hook(save_activation)
    h2 = target_layer.register_backward_hook(save_gradient)

    out = model(x)
    score = out[0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)

    acts = activations[0].cpu().numpy()[0]
    grads = gradients[0].cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, padded_size)  # Resize to padded size (square)
    
    # Crop padding to match original resized aspect ratio
    cam = cam[pad_top:pad_top + (padded_size[1] - 2*pad_top), pad_left:pad_left + (padded_size[0] - 2*pad_left)]
    
    # Resize cropped CAM to original image size
    cam = cv2.resize(cam, img_pil.size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam  # Avoid division by zero

    img = np.array(img_pil.convert('RGB'))  # Ensure RGB
    heatmap = (255 * cam).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
    
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    h1.remove()
    h2.remove()

    return Image.fromarray(overlay)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', results=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        flash('Unsupported file extension. Allowed: ' + ','.join(ALLOWED_EXT))
        return redirect(url_for('index'))

    try:
        path = UPLOAD_DIR / filename
        file.save(path)
        pil_img = Image.open(path).convert('RGB')

        results, pred_class_idx = predict_from_image(pil_img)
        overlay_img = gradcam_on_image(model, pil_img, pred_class_idx)
        overlay_filename = f"overlay_{filename}"
        overlay_path = UPLOAD_DIR / overlay_filename
        overlay_img.save(overlay_path)

        return render_template(
            'index.html',
            results=results,
            filename=filename,
            overlay_filename=overlay_filename,
            descriptions=CLASS_DESCRIPTIONS
        )
    except Exception as e:
        flash('Failed to process image: ' + str(e))
        return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print('Server running at http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)