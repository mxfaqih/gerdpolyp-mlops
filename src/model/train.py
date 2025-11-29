import os
import json
import mlflow
import mlflow.pytorch
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.load_config import load_config, parse_args
from src.data.dataset_loader import load_datasets
from torchvision import models
from torch.utils.data import DataLoader


# ============================================================
#              TRAINING PIPELINE IMPLEMENTATION
# ============================================================

def train_model(config):
    # --------------------------------------------------------
    # Setup MLflow
    # --------------------------------------------------------
    mlflow.set_tracking_uri(config["logging"]["tracking_uri"])
    mlflow.set_experiment(config["logging"]["experiment_name"])

    with mlflow.start_run(run_name=config["logging"]["run_name"]):

        # ----------------------------------------------------
        # Log Params
        # ----------------------------------------------------
        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])
        mlflow.log_params(config["data"])

        # ----------------------------------------------------
        # Load Dataset
        # ----------------------------------------------------
        data_cfg = config["data"]
        train_ds, val_ds, test_ds = load_datasets(
            root=data_cfg["root"],
            original_dir=data_cfg["original_dir"],
            augmented_dir=data_cfg["augmented_dir"],
            use_augmented=data_cfg["use_augmented"],
            img_size=data_cfg["img_size"],
            normalize=data_cfg["normalize"],
            seed=config["training"]["seed"]
        )

        # Build DataLoaders
        batch = config["training"]["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

        # ----------------------------------------------------
        # Build Model
        # ----------------------------------------------------
        model_cfg = config["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.convnext_tiny(weights="IMAGENET1K_V1")
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, model_cfg["num_classes"])
        model = model.to(device)

        # Full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        # ----------------------------------------------------
        # Training Loop
        # ----------------------------------------------------
        EPOCHS = config["training"]["epochs"]
        PATIENCE = config["training"]["early_stopping"]

        best_val_loss = float("inf")
        no_improve = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(EPOCHS):
            # ============================ TRAIN ============================
            model.train()
            train_total, train_correct, train_loss_sum = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_acc = train_correct / train_total

            # ============================ VAL ============================
            model.eval()
            val_total, val_correct, val_loss_sum = 0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total

            # Save metrics
            history["train_loss"].append(train_loss_sum / len(train_loader))
            history["val_loss"].append(val_loss_sum / len(val_loader))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"TrainLoss: {train_loss_sum/len(train_loader):.4f} | "
                  f"ValLoss: {val_loss_sum/len(val_loader):.4f} | "
                  f"ValAcc: {val_acc:.4f}")

            # Log per-epoch to MLflow
            mlflow.log_metric("train_loss", train_loss_sum/len(train_loader), step=epoch)
            mlflow.log_metric("val_loss", val_loss_sum/len(val_loader), step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            # ======================= EARLY STOPPING ========================
            if val_loss_sum < best_val_loss:
                best_val_loss = val_loss_sum
                no_improve = 0

                # Save best model
                os.makedirs("artifacts", exist_ok=True)
                best_model_path = config["paths"]["best_model"]
                torch.save(model.state_dict(), best_model_path)

            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Save history
        history_path = config["paths"]["history"]
        pd.DataFrame(history).to_csv(history_path, index=False)
        mlflow.log_artifact(history_path)

        # ----------------------------------------------------
        # TESTING
        # ----------------------------------------------------
        model.load_state_dict(torch.load(config["paths"]["best_model"]))
        model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Compute metrics
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec  = recall_score(y_true, y_pred, average="macro")
        f1   = f1_score(y_true, y_pred, average="macro")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)

        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------
        if config["evaluation"]["confusion_matrix"]:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            cm_path = config["paths"]["confusion_matrix"]
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

        # Log best model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="model")


# ============================================================
#                   ENTRY POINT
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_model(config)
