import torch
import json
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.load_config import load_config
from src.data.dataset_loader import load_datasets


def load_trained_model(path, num_classes=4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate():
    cfg = load_config("src/config/params.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_ds, val_ds, test_ds = load_datasets(
        root=cfg["data"]["root"],
        original_dir=cfg["data"]["original_dir"],
        augmented_dir=cfg["data"]["augmented_dir"],
        use_augmented=cfg["data"]["use_augmented"],
        img_size=cfg["data"]["img_size"],
        normalize=cfg["data"]["normalize"],
        seed=cfg["training"]["seed"]
    )

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load model
    model = load_trained_model(cfg["paths"]["best_model"], num_classes=cfg["model"]["num_classes"])

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="magma")
    plt.title("Confusion Matrix (Evaluate Script)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("artifacts/eval_confusion_matrix.png")

    # MLflow logging
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_precision", prec)
    mlflow.log_metric("eval_recall", rec)
    mlflow.log_metric("eval_f1", f1)
    mlflow.log_artifact("artifacts/eval_confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
