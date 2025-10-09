import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "/home/phamtiendat/Documents/ComputerVision/weights_resnet50/best.pth"
NUM_CLASSES = 6
CLASS_NAMES = ['Nail_psoriasis', 'SJS-TEN', 'Unknown_Normal' ,'Vitiligo', 'acne', 'hyperpigmentation'] # <-- update with your labels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LOAD MODEL
# ------------------------
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ------------------------
# TRANSFORM (same as training)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------
# CAMERA LOOP
# ------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Convert frame (OpenCV BGR -> PIL RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Preprocess
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Predict
        outputs = model(input_tensor)
        _, pred = outputs.max(1)
        label = CLASS_NAMES[pred.item()]

        # Show prediction on frame
        cv2.putText(frame, f"Pred: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Real-time Prediction", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
