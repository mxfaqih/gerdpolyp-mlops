import os
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms


def build_transforms(img_size: int = 224, normalize: bool = True):
    """
    Build PyTorch transforms based on config.
    """
    if normalize:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])


def load_datasets(
    root: str,
    original_dir: str,
    augmented_dir: str,
    use_augmented: bool,
    img_size: int,
    normalize: bool,
    seed: int = 42,
    train_split: float = 0.7,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataset folders (original + optional augmented),
    split into train/val/test loaders, and return dataloaders.
    """

    root = Path(root)

    # ───────────────────────────────
    # 1. BUILD TRANSFORMS
    # ───────────────────────────────
    transform = build_transforms(img_size, normalize)

    # ───────────────────────────────
    # 2. LOAD ORIGINAL DATASET
    # ───────────────────────────────
    original_path = root / original_dir
    print(f"Loading ORIGINAL dataset from: {original_path}")

    original_dataset = datasets.ImageFolder(
        root=str(original_path),
        transform=transform
    )

    datasets_list = [original_dataset]

    # ───────────────────────────────
    # 3. OPTIONAL AUGMENTED DATA
    # ───────────────────────────────
    if use_augmented:
        augmented_path = root / augmented_dir
        print(f"Loading AUGMENTED dataset from: {augmented_path}")

        augmented_dataset = datasets.ImageFolder(
            root=str(augmented_path),
            transform=transform
        )
        datasets_list.append(augmented_dataset)

    # Merge into one
    dataset = ConcatDataset(datasets_list)
    dataset_size = len(dataset)
    print(f"Total samples: {dataset_size}")

    # ───────────────────────────────
    # 4. SPLITTING
    # ───────────────────────────────
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # ───────────────────────────────
    # 5. CREATE LOADERS
    # ───────────────────────────────
    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=32,         # default, overridden later
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    return train_ds, val_ds, test_ds
