from ultralytics import YOLO
import cv2

def load_model():
    return YOLO("weights/best_2.pt")  # model nhận diện đèn

def detect(model, frame):
    results = model(frame, conf=0.5, iou=0.5)[0]

    detections = []
    boxes = results.boxes
    print(f"[🚦 DETECT_LIGHT] Số lượng object phát hiện: {len(boxes)}")

    status_map = {0: "red", 1: "green", 2: "yellow"}  # ánh xạ class_id -> trạng thái đèn

    for idx, (box, cls_id, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_id = int(cls_id.item())
        confidence = float(conf.item())
        status = status_map.get(class_id, "unknown")

        detections.append({
            "id": str(idx),
            "box": [x1, y1, x2, y2],
            "class_id": class_id,
            "confidence": confidence,
            "status": status
        })

    return detections

def draw_lights(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        status = det["status"]
        conf = det["confidence"]

        # Màu tương ứng
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "unknown": (128, 128, 128)
        }
        color = color_map.get(status, (255, 255, 255))

        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Vẽ trạng thái và độ tin cậy
        label = f"{status} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame
