import cv2
import torch
import torch.nn as nn
import socket
import json

from torchvision import models, transforms
from PIL import Image


# =========================
# SETTINGS
# =========================

MODEL_FILE = "mask_type_model.pth"
DEVICE = torch.device("cpu")

NODE_RED_HOST = "127.0.0.1"
NODE_RED_PORT = 5005


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]

print("Classes:", classes)
print("Validation Accuracy:", checkpoint["accuracy"])


# =========================
# CREATE MODEL
# =========================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        model.last_channel,
        len(classes)
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================
# CONNECT TO NODE-RED
# =========================

node_red = None

try:

    node_red = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    node_red.connect(
        (NODE_RED_HOST, NODE_RED_PORT)
    )

    print("Connected to Node-RED!")

except Exception as e:

    print("Node-RED connection failed:", e)

    if node_red is not None:
        node_red.close()

    node_red = None


# =========================
# START CAMERA
# =========================

print("Starting webcam...")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    if node_red is not None:
        node_red.close()

    raise SystemExit


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Add margin around face
        margin = int(0.25 * w)

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

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # BGR -> RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            face_rgb
        )

        input_tensor = transform(
            pil_image
        ).unsqueeze(0)

        input_tensor = input_tensor.to(DEVICE)


        # =========================
        # PREDICTION
        # =========================

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        label = classes[
            predicted.item()
        ]

        confidence_percent = (
            confidence.item() * 100
        )


        # =========================
        # SEND TO NODE-RED
        # =========================

        if node_red is not None:

            data = {
                "mask": label,
                "confidence": round(
                    confidence_percent,
                    2
                )
            }

            try:

                message = (
                    json.dumps(data) + "\n"
                )

                node_red.sendall(
                    message.encode()
                )

            except Exception as e:

                print(
                    "Node-RED connection lost:",
                    e
                )

                node_red.close()
                node_red = None


        # =========================
        # DISPLAY LABEL
        # =========================

        if label == "NO_MASK":
            display_label = "NO MASK"

        elif label == "N95":
            display_label = "N95 MASK"

        elif label == "SURGICAL":
            display_label = "SURGICAL MASK"

        elif label == "CLOTH":
            display_label = "CLOTH MASK"

        else:
            display_label = label


        text = (
            f"{display_label} "
            f"{confidence_percent:.1f}%"
        )


        # =========================
        # DRAW FACE BOX
        # =========================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Text size
        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )


        # Text background
        cv2.rectangle(
            frame,
            (x, y - text_height - 15),
            (
                x + text_width + 10,
                y
            ),
            (0, 255, 0),
            -1
        )


        # Text
        cv2.putText(
            frame,
            text,
            (x + 5, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )


    # =========================
    # SHOW CAMERA
    # =========================

    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    # =========================
    # QUIT
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()

if node_red is not None:
    node_red.close()

print("Webcam stopped.")
print("Node-RED connection closed.")