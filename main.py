import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from torchvision import transforms
from PIL import Image
import io
from model import LeNet

app = FastAPI()
device = torch.device("cpu")

@app.get("/")
def root():
    return FileResponse("index.html")

# تحميل الموديل
checkpoint = torch.load("lenet_mnist.pth", map_location=device)
net = LeNet().to(device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# نفس الـ transform اللي اتدرب عليها
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = net(tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return {
        "digit": predicted.item(),
        "confidence": round(confidence.item() * 100, 2)
    }