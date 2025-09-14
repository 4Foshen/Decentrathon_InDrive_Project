from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms

# --- твоя модель из проекта ---
class CarStateNet(nn.Module):
    def __init__(self):
        super(CarStateNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 класса
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --- загрузка модели ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarStateNet().to(device)
model.eval()

# --- классы ---
classes = [
    ("Чистый", "Целый"),
    ("Чистый", "Битый"),
    ("Грязный", "Целый"),
    ("Грязный", "Битый")
]

# --- трансформация картинки ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])


def predict_car_state(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]


# --- FastAPI ---
app = FastAPI()

@app.post("/check_car/")
async def check_car(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_car_state(image)
        return JSONResponse(content={
            "cleanliness": result[0],
            "integrity": result[1]
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {e}")
