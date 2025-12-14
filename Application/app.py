from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from pathlib import Path
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.secret_key = 'super_secret_key'


CLASS_DESCRIPTIONS = {
    "Nail_psoriasis": "Psoriatic changes in nails — may present with pitting, discoloration, or onycholysis.",
    "SJS-TEN": "Severe skin reaction (Stevens-Johnson / Toxic Epidermal Necrolysis) — urgent clinical evaluation required.",
    "Unknown_Normal": "Appears to be normal skin or a condition not recognized by the model.",
    "Vitiligo": "Loss of pigment in patches on the skin. Consult a dermatologist for treatment options.",
    "acne": "Inflammatory lesions (pimples, pustules). Treatments available; consult clinician for severe cases.",
    "hyperpigmentation": "Areas of increased pigment; causes include post-inflammatory changes, sun exposure, or hormonal factors.",
}


def load_custom_model(ckpt_path, num_classes):
    print(f"Loading model from {ckpt_path}...")
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
    try:
        checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
    model = model.to(DEVICE)
    model.eval()
    return model

model = None
if MODEL_PATH.exists():
    model = load_custom_model(MODEL_PATH, NUM_CLASSES)
else:
    print(f"Warning: Model path {MODEL_PATH} not found.")

#PREPROCESSING
def preprocess_image(img_pil):
    w, h = img_pil.size
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    img_padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img_padded).unsqueeze(0).to(DEVICE)
    
    meta = {
        'orig_size': (w, h),
        'valid_size': (new_w, new_h),
        'offset': (paste_x, paste_y)
    }
    return tensor, meta

#GRAD-CAM
def generate_gradcam(model, img_pil, target_class_idx):
    model.eval()
    input_tensor, meta = preprocess_image(img_pil)
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
        
    target_layer = model.layer4[-1] 
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    model.zero_grad()
    output = model(input_tensor)
    
    score = output[0, target_class_idx]
    score.backward()
    
    grads = gradients[0].cpu().data.numpy()[0]
    fmaps = activations[0].cpu().data.numpy()[0]
    
    handle_f.remove()
    handle_b.remove()
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * fmaps[i]
        
    cam = np.maximum(cam, 0)
    
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    valid_w, valid_h = meta['valid_size']
    off_x, off_y = meta['offset']
    cam_cropped = cam[off_y : off_y + valid_h, off_x : off_x + valid_w]
    
    orig_w, orig_h = meta['orig_size']
    cam_final = cv2.resize(cam_cropped, (orig_w, orig_h))
    
    # Normalize về 0-1
    cam_final = cam_final - np.min(cam_final)
    if np.max(cam_final) != 0:
        cam_final = cam_final / np.max(cam_final)
    
    # Không dùng thresholding nữa để giữ màu xanh mượt mà
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_final), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # alpha=0.6 (ảnh gốc), beta=0.4 (heatmap)
    img_orig_np = np.array(img_pil)
    overlay = cv2.addWeighted(img_orig_np, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlay)

def allowed_file(filename):
    return '.' in filename and Path(filename).suffix.lower() in ALLOWED_EXT

# Route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = UPLOAD_DIR / filename
        file.save(filepath)
        
        img_pil = Image.open(filepath).convert('RGB')
        
        input_tensor, _ = preprocess_image(img_pil)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        top_k_idxs = probs.argsort()[::-1][:TOP_K]
        
        results = []
        for i in top_k_idxs:
            class_name = CLASS_NAMES[i]
            probability = float(probs[i])
            results.append((class_name, probability))
            
        top_class_idx = top_k_idxs[0]
        
        overlay_img = generate_gradcam(model, img_pil, top_class_idx)
        overlay_filename = "gradcam_" + filename
        overlay_img.save(UPLOAD_DIR / overlay_filename)
        
        return render_template('index.html', 
                               results=results, 
                               filename=filename, 
                               overlay_filename=overlay_filename,
                               descriptions=CLASS_DESCRIPTIONS)
                               
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)