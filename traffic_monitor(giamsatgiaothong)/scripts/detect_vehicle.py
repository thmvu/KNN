from ultralytics import YOLO
import cv2

# Load model đã train từ Roboflow
model = YOLO("weights/best.pt")  # ← sau khi tải xong bạn thay path vào đây

# Dùng webcam hoặc video test
cap = cv2.VideoCapture(0)  # hoặc thay bằng đường dẫn file video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Vẽ kết quả
    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
