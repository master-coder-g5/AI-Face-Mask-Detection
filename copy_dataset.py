import os
import shutil

SOURCE = r"C:\Users\prith\Downloads\deep-mask-detect-master\deep-mask-detect-master\data\phase1"
DEST = r"C:\Users\prith\face-mask-project\dataset_downloaded"

mapping = {
    "cloth": "CLOTH",
    "n95": "N95",
    "nfm": "NO_MASK",
    "surgical": "SURGICAL"
}

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total = 0

for source_folder, destination_folder in mapping.items():

    source_path = os.path.join(SOURCE, source_folder)
    destination_path = os.path.join(DEST, destination_folder)

    os.makedirs(destination_path, exist_ok=True)

    count = 0

    for root, dirs, files in os.walk(source_path):

        for filename in files:

            if filename.lower().endswith(extensions):

                source_file = os.path.join(root, filename)

                # Add class name to avoid duplicate filenames
                new_name = f"{destination_folder}_{count + 1:05d}" + os.path.splitext(filename)[1].lower()

                destination_file = os.path.join(destination_path, new_name)

                shutil.copy2(source_file, destination_file)

                count += 1
                total += 1

    print(f"{source_folder} -> {destination_folder}: {count} images")

print()
print("=" * 45)
print(f"TOTAL IMAGES COPIED: {total}")
print("=" * 45)
print("Dataset copying completed!")