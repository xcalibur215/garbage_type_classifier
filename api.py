import io
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
import uvicorn

app = FastAPI()

# Model
import torch.nn as nn
class GarbageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.35),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Linear(128, 6)
        )
    def forward(self, x):
        return self.model(x)

model = GarbageClassifier()
try:
    state_dict = torch.load('stat_dict.pt', map_location='cpu')
    model.model.load_state_dict(state_dict)
    model.eval()
except FileNotFoundError:
    print("ERROR: stat_dict.pt not found. Please provide the correct model weights file.")
    raise
except RuntimeError as e:
    print(f"ERROR loading state_dict: {e}")
    raise

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Trained class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.post('/predict/')
async def predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image)
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = T.ToTensor()(input_tensor)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
            results.append({
                'filename': file.filename,
                'class': class_names[pred]
            })
    return JSONResponse(content={'results': results})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
