from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------- CONFIG (edit before running) ----------------
MODEL_PATH = Path("/home/phamtiendat/Documents/ComputerVision/weights_resnet50_head_custom/best.pth")

# --- Manually define your class names in the correct order ---
CLASS_NAMES = ['Nail_psoriasis', 'SJS-TEN', 'Unknown_Normal', 'Vitiligo', 'acne', 'hyperpigmentation']
NUM_CLASSES = len(CLASS_NAMES)

DATA_DIR = Path.cwd()
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3
IMG_SIZE = 224
# ----------------------------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.secret_key = 'replace-me-with-a-very-secure-key'

# ------------------- utilities: model loading -------------------
def load_custom_model(ckpt_path, num_classes):
    """
    Loads the ResNet-50 with the specific custom head architecture.
    """
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

# ------------------ class descriptions (editable) ------------------
CLASS_DESCRIPTIONS = {
    "Nail_psoriasis": "Psoriatic changes in nails — may present with pitting, discoloration, or onycholysis. Clinical exam required for confirmation.",
    "SJS-TEN": "Severe skin reaction (Stevens-Johnson / Toxic Epidermal Necrolysis) — urgent clinical evaluation required.",
    "Unknown_Normal": "Appears to be normal skin or a condition not recognized by the model.",
    "Vitiligo": "Loss of pigment in patches on the skin. Generally non-contagious; consult a dermatologist for treatment options.",
    "acne": "Inflammatory lesions of the pilosebaceous unit (pimples, pustules). Many treatments available; consult clinician for severe cases.",
    "hyperpigmentation": "Areas of increased pigment; causes include post-inflammatory changes, sun exposure, or hormonal factors.",
}

# ------------------ load model at startup ------------------
model = None
if MODEL_PATH.exists():
    try:
        model = load_custom_model(MODEL_PATH, NUM_CLASSES)
        print("✅ Model loaded successfully. Classes:", CLASS_NAMES)
    except Exception as e:
        print(f"❌ Failed to load model at startup: {e}")
        model = None
else:
    print(f"⚠️ Warning: MODEL_PATH {MODEL_PATH} does not exist. The app will not be able to predict.")

# ------------------ preprocessing ------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------ helpers ------------------
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

def predict_from_image(img_pil: Image.Image):
    if model is None:
        raise RuntimeError("Model is not loaded on server. Configure MODEL_PATH and restart.")
    x = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    topk_idx = probs.argsort()[::-1][:TOP_K]
    results = [(CLASS_NAMES[i], float(probs[i])) for i in topk_idx]
    return results

# ------------------ routes ------------------

# ** NO MORE INDEX_HTML VARIABLE HERE **

@app.route('/', methods=['GET'])
def index():
    # Use render_template to load and display the HTML file
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
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        pil_img = Image.open(in_memory_file).convert('RGB')
        results = predict_from_image(pil_img)
        # Pass the results back to the same template
        return render_template('index.html', results=results, filename=filename, descriptions=CLASS_DESCRIPTIONS)
    except Exception as e:
        flash('Failed to process image: ' + str(e))
        return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print('Starting server on http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)