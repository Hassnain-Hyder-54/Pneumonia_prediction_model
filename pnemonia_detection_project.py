import os
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(r"D:\Medical_Datasets\ChestMNIST")

IMAGE_SIZE = 128

DATA_FILE = DATA_DIR / f"chestmnist_{IMAGE_SIZE}.npz"

MODEL_PATH = DATA_DIR / "best_chestmnist_model.pth"

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

SEED = 42

DISEASE_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia"
]

NUM_CLASSES = len(DISEASE_NAMES)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CHESTMNIST AI DISEASE CLASSIFICATION")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# DATASET CLASS
# =========================================================

class ChestMNISTDataset(Dataset):

    def __init__(self, images, labels, training=False):

        self.images = images
        self.labels = labels.astype(np.float32)

        if training:

            self.transform = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.fromarray(
            self.images[index]
        ).convert("L")

        image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image, label


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    print("\nLoading ChestMNIST dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with np.load(DATA_FILE) as data:

        train_images = data["train_images"]
        train_labels = data["train_labels"]

        val_images = data["val_images"]
        val_labels = data["val_labels"]

        test_images = data["test_images"]
        test_labels = data["test_labels"]

    print("Training images:", len(train_images))
    print("Validation images:", len(val_images))
    print("Testing images:", len(test_images))

    train_dataset = ChestMNISTDataset(
        train_images,
        train_labels,
        training=True
    )

    val_dataset = ChestMNISTDataset(
        val_images,
        val_labels
    )

    test_dataset = ChestMNISTDataset(
        test_images,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


# =========================================================
# BUILD RESNET18 MODEL
# =========================================================

def build_model():

    print("\nBuilding ResNet18 model...")

    model = models.resnet18(weights=None)

    # ChestMNIST images are grayscale.
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Output 14 independent disease scores.
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model.to(DEVICE)


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / len(loader.dataset)


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_scores = []

    for images, labels in tqdm(
        loader,
        desc="Evaluating",
        leave=False
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probabilities = torch.sigmoid(outputs)

        running_loss += loss.item() * images.size(0)

        all_labels.append(
            labels.cpu().numpy()
        )

        all_scores.append(
            probabilities.cpu().numpy()
        )

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    auc_scores = []

    for i in range(NUM_CLASSES):

        y_true = all_labels[:, i]
        y_score = all_scores[:, i]

        # AUROC requires positive and negative samples.
        if len(np.unique(y_true)) == 2:

            auc = roc_auc_score(
                y_true,
                y_score
            )

            auc_scores.append(auc)

        else:
            auc_scores.append(np.nan)

    mean_auc = np.nanmean(auc_scores)

    average_loss = running_loss / len(loader.dataset)

    return average_loss, mean_auc, auc_scores


# =========================================================
# MAIN TRAINING LOOP
# =========================================================

def main():

    set_seed(SEED)

    train_loader, val_loader, test_loader = load_dataset()

    model = build_model()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_auc = -1.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_auc, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUROC:  {val_auc:.4f}")

        scheduler.step(val_auc)

        if val_auc > best_auc:

            best_auc = val_auc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "disease_names": DISEASE_NAMES,
                    "best_val_auc": best_auc
                },
                MODEL_PATH
            )

            print("Best model saved!")

    print("\nTraining completed!")

    # =====================================================
    # FINAL TEST EVALUATION
    # =====================================================

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_auc, disease_aucs = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean AUROC: {test_auc:.4f}")

    print("\nDisease-wise AUROC:\n")

    for disease, auc in zip(
        DISEASE_NAMES,
        disease_aucs
    ):

        print(f"{disease:25s}: {auc:.4f}")

    print("\nModel saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()