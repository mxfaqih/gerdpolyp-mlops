import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json


def load_model(model_path, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, image_path, class_map_path):
    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    return class_map[str(idx.item())], round(conf.item(), 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="artifacts/model_best.pth")
    parser.add_argument("--classes", default="app/classes.json")
    args = parser.parse_args()

    model, device = load_model(args.model)
    label, conf = predict_image(model, device, args.image, args.classes)

    print(f"Predicted: {label} ({conf*100:.2f}%)")