import torch
import torchvision.transforms as transforms

from PIL import Image
from model import CNNTransformer


class ModelInference:
    def __init__(self, model_path="weights/model_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNTransformer()
        # 載入權重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 與訓練時相同的預處理
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),               # → [1, 28, 28] 且自動除以 255
        ])

    def predict(self, img: Image.Image):
        # 影像預處理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.argmax(output, dim=1)
            
        return int(pred.item())