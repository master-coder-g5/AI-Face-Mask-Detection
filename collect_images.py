import cv2
import time
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

TARGET_IMAGES = 400
CAPTURE_INTERVAL = 0.20   # seconds between images

classes = {
    ord("1"): "NO_MASK",
    ord("2"): "SURGICAL",
    ord("3"): "N95",
    ord("4"): "CLOTH"
}

base_dir = Path("dataset")

for name in classes.values():
    (base_dir / name).mkdir(parents=True, exist_ok=True)

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print()
print("==========================================")
print("       AUTOMATIC MASK DATASET COLLECTOR")
print("==========================================")
print()
print("1 = NO MASK")
print("2 = SURGICAL")
print("3 = N95")
print("4 = CLOTH")
print("Q = QUIT")
print()
print(f"Target: {TARGET_IMAGES} images per class")
print()

selected_class = None
last_capture_time = 0

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Cannot read webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    display = frame.copy()

    # --------------------------------------
    # Draw faces
    # --------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # --------------------------------------
    # No class selected
    # --------------------------------------

    if selected_class is None:

        cv2.putText(
            display,
            "SELECT MASK TYPE",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "1:No Mask  2:Surgical  3:N95  4:Cloth",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # --------------------------------------
    # Class selected
    # --------------------------------------

    else:

        folder = base_dir / selected_class

        image_count = len(
            list(folder.glob("*.jpg"))
        )

        cv2.putText(
            display,
            f"COLLECTING: {selected_class}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Images: {image_count}/{TARGET_IMAGES}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Move your face slowly",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------
        # Capture
        # ----------------------------------

        current_time = time.time()

        if (
            len(faces) > 0
            and current_time - last_capture_time >= CAPTURE_INTERVAL
            and image_count < TARGET_IMAGES
        ):

            # Largest face
            face = max(
                faces,
                key=lambda r: r[2] * r[3]
            )

            x, y, w, h = face

            # Margin around face
            margin = int(0.20 * w)

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)

            x2 = min(
                frame.shape[1],
                x + w + margin
            )

            y2 = min(
                frame.shape[0],
                y + h + margin
            )

            crop = frame[y1:y2, x1:x2]

            crop = cv2.resize(
                crop,
                (224, 224)
            )

            number = image_count + 1

            filepath = (
                folder /
                f"{selected_class}_{number:04d}.jpg"
            )

            if cv2.imwrite(
                str(filepath),
                crop
            ):

                print(
                    f"Saved {selected_class}: "
                    f"{number}/{TARGET_IMAGES}"
                )

            last_capture_time = current_time

        # ----------------------------------
        # Finished
        # ----------------------------------

        if image_count >= TARGET_IMAGES:

            print(
                f"\n{selected_class} dataset complete!"
            )

            selected_class = None

    cv2.imshow(
        "Automatic Mask Dataset Collector",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Select class
    if key in classes:

        selected_class = classes[key]

        print()
        print(
            f"Starting collection: "
            f"{selected_class}"
        )
        print(
            "Move your head slowly in different directions."
        )

        last_capture_time = 0

camera.release()
cv2.destroyAllWindows()

print()
print("Dataset collection stopped.")