
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam is working! Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not capture a frame.")
        break

    cv2.imshow("Face Mask Detection - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()