import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# =========================
# SETTINGS
# =========================

torch.set_num_threads(2)

DEVICE = torch.device("cpu")

TRAIN_DIR = r"C:\Users\prith\face-mask-project\dataset_split\train"
VAL_DIR = r"C:\Users\prith\face-mask-project\dataset_split\validation"

MODEL_FILE = "mask_type_model.pth"

EPOCHS = 15
BATCH_SIZE = 8

# =========================
# DATA AUGMENTATION
# =========================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================

print("Device:", DEVICE)
print("Loading datasets...")

train_data = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_data = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_data.classes)
print("Train:", len(train_data))
print("Validation:", len(val_data))

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# MODEL
# =========================

print("Loading MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze everything first
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the LAST few feature layers
for param in model.features[-4:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        model.last_channel,
        len(train_data.classes)
    )
)

model = model.to(DEVICE)

print("Model ready")
print("Fine-tuning last MobileNetV2 layers...")

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# =========================
# TRAINING
# =========================

best_accuracy = 0.0
patience = 4
no_improvement = 0

print("Starting training...")

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy
        no_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes,
                "accuracy": best_accuracy
            },
            MODEL_FILE
        )

        print(
            f"  >>> Best model saved! "
            f"Validation Accuracy: {best_accuracy:.2f}%"
        )

    else:

        no_improvement += 1

    # -------------------------
    # EARLY STOPPING
    # -------------------------

    if no_improvement >= patience:

        print()
        print("Early stopping.")
        break

# =========================
# FINISHED
# =========================

print()
print("================================")
print("Training complete!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print(f"Model saved: {MODEL_FILE}")
print("================================")