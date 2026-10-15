import cv2
import os
import time

# =========================
# SETTINGS
# =========================

SAVE_DIR = r"C:\Users\prith\face-mask-project\cloth_extra"

TOTAL_IMAGES = 100

# Save one image every 0.3 seconds
SAVE_INTERVAL = 0.3

# =========================
# CREATE FOLDER
# =========================

os.makedirs(SAVE_DIR, exist_ok=True)

# Find existing images so we don't overwrite them
existing = [
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

count = len(existing)

print("========================================")
print(" CLOTH / HANDKERCHIEF IMAGE COLLECTOR")
print("========================================")
print()
print("Images already saved:", count)
print("Images to collect:", TOTAL_IMAGES)
print()
print("Instructions:")
print("1. Wear the handkerchief/cloth properly.")
print("2. Look directly at the camera.")
print("3. Slowly move your head left/right/up/down.")
print("4. Change your distance from the camera.")
print("5. Use different lighting/positions.")
print("6. Press Q to stop.")
print()
print("Starting camera...")

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

last_save_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    current_time = time.time()

    # =========================
    # SAVE IMAGE
    # =========================

    if (
        current_time - last_save_time >= SAVE_INTERVAL
        and count < TOTAL_IMAGES
    ):

        filename = os.path.join(
            SAVE_DIR,
            f"cloth_{count + 1:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        count += 1
        last_save_time = current_time

        print(f"Saved: {filename}")

    # =========================
    # DISPLAY
    # =========================

    cv2.putText(
        frame,
        f"CLOTH images: {count}/{TOTAL_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = Stop",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "CLOTH - Handkerchief Collection",
        frame
    )

    # Stop with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Automatically stop after 100 images
    if count >= TOTAL_IMAGES:
        print()
        print("========================================")
        print("Collection complete!")
        print(f"Total images: {count}")
        print(f"Saved in: {SAVE_DIR}")
        print("========================================")
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print()
print("Camera stopped.")
print(f"Total CLOTH images collected: {count}")