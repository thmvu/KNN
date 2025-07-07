from ultralytics import YOLO
import cv2

model = YOLO("weights/traffic_light_best.pt")

cap = cv2.VideoCapture("input/videos/test_red_light.mp4")
while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Traffic Light Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
