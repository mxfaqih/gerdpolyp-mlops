import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path


class ModelService:
    """
    Singleton-like service for loading and running the ConvNeXt model.
    Production optimized: model loaded once, no repeated weight loading.
    """

    def __init__(self, model_path: str, class_map_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Load class mapping
        with open(class_map_path, "r") as f:
            self.class_map = json.load(f)

        # Build preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load model once
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """
        Load ConvNeXt-Tiny model and modify classification head for 4 classes.
        """
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, len(self.class_map))

        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def preprocess_image(self, image: Image.Image):
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image):
        """
        Perform prediction for a PIL.Image.
        Returns:
            {
               "pred_class": "Gerd",
               "confidence": 0.94
            }
        """
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        pred_class = self.class_map[str(pred_idx.item())]

        return {
            "pred_class": pred_class,
            "confidence": round(conf.item(), 4)
        }
