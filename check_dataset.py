import os
from PIL import Image

DATASET = r"C:\Users\prith\face-mask-project\dataset_downloaded"

valid = 0
bad = 0

for root, dirs, files in os.walk(DATASET):
    for filename in files:
        path = os.path.join(root, filename)

        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1

        except Exception:
            bad += 1
            print("BAD:", path)

print()
print("=" * 40)
print("Valid images:", valid)
print("Corrupted images:", bad)
print("=" * 40)