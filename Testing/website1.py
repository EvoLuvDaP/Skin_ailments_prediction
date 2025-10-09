import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = Path("/home/phamtiendat/Documents/ComputerVision/weights_resnet50_head_custom/best.pth")
NUM_CLASSES = 6   # Change according to your dataset
CLASS_NAMES = ["acne","hyperpigmentation", "psoriasis",  "SJS-TEN", "Unknown", "Vitiligo"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# MODEL INIT + SAFE LOADING
# --------------------------
def load_model(model_path: Path, num_classes: int, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    state_dict = torch.load(model_path, map_location=device)

    # handle strict loading issues (fc mismatch)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model_state.update(filtered_dict)
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

# --------------------------
# IMAGE TRANSFORM
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def predict_image(image: Image.Image, model, transform, class_names, device):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze(0)
    top_idx = probs.argmax()
    return class_names[top_idx], float(probs[top_idx])

# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Skin Disease Classifier</title></head>
        <body>
        <h2>Upload a skin image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/png,image/jpeg,image/jpg,image/webp">
            <input type="submit" value="Upload">
        </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        pred_class, confidence = predict_image(image, model, transform, CLASS_NAMES, DEVICE)
        result = {
            "filename": file.filename,
            "predicted_class": pred_class,
            "confidence": confidence
        }
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()  # <-- print full error to terminal
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --------------------------
# ENTRYPOINT
# --------------------------
if __name__ == "__main__":
    uvicorn.run("website:app", host="127.0.0.1", port=8000, reload=True)
