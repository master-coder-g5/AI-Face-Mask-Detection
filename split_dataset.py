import os
import random
import shutil

SOURCE = r"C:\Users\prith\face-mask-project\dataset_downloaded"
DEST = r"C:\Users\prith\face-mask-project\dataset_split"

CLASSES = ["CLOTH", "N95", "NO_MASK", "SURGICAL"]

TRAIN_RATIO = 0.80

random.seed(42)

for class_name in CLASSES:

    source_folder = os.path.join(SOURCE, class_name)

    train_folder = os.path.join(DEST, "train", class_name)
    validation_folder = os.path.join(DEST, "validation", class_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    validation_images = images[split_index:]

    for filename in train_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(train_folder, filename)
        )

    for filename in validation_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(validation_folder, filename)
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(validation_images)} validation"
    )

print()
print("=" * 45)
print("DATASET SPLIT COMPLETE")
print("=" * 45)