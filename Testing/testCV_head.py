import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "/home/phamtiendat/Documents/ComputerVision/weights_resnet50_head_custom/best.pth"
NUM_CLASSES = 6
CLASS_NAMES = ['Nail_psoriasis', 'SJS-TEN', 'Unknown_Normal' ,'Vitligo', 'acne', 'hyperpigmentation']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LOAD MODEL
# ------------------------
# 1. Load the ResNet-50 backbone WITH pre-trained ImageNet weights.
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Get the number of input features for the classifier.
in_f = backbone.fc.in_features

# 3. Re-create the EXACT same custom head as in training.
custom_head = nn.Sequential(
    nn.Linear(in_f, 512),
    nn.ELU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.2),
    nn.Linear(512, 128),
    nn.ELU(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(p=0.2),
    nn.Linear(128, NUM_CLASSES)
)

# 4. Attach the custom head, replacing the original one.
backbone.fc = custom_head
model = backbone

# 5. Load the checkpoint dictionary from your file.
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 6. <<< --- THE FINAL FIX --- >>>
# Load the state_dict into the ENTIRE model structure.
# Use `strict=False` to ignore the missing backbone keys.
model.load_state_dict(checkpoint['model_state'], strict=False)

model.to(DEVICE)
model.eval()

# ------------------------
# TRANSFORM (same as training)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------
# CAMERA LOOP
# ------------------------
print("🚀 Starting camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)

        label = CLASS_NAMES[pred.item()]
        conf_score = confidence.item()

        display_text = f"Pred: {label} ({conf_score:.2f})"
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()