from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from train_model import StrokeNet

# Load model
model = StrokeNet()
model.load_state_dict(torch.load("model/stroke_model.pth"))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Load image
img = Image.open("test_image.png").convert("RGB")
img = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img)
    probs = F.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)

classes = ['Normal','Stroke']

print("Prediction:", classes[pred.item()])
print("Confidence:", round(confidence.item()*100,2), "%")