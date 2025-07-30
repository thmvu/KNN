from ultralytics import YOLO
import cv2

def load_model():
    return YOLO("weights/light1.pt")  # Đường dẫn đến model YOLO đèn giao thông

def detect(model, frame, conf_thresh=0.2, iou_thresh=0.2):
    results = model(frame, conf=conf_thresh, iou=iou_thresh)[0]
    detections = []

    status_map = {
        0: "green",
        1: "red",
        2: "yellow"
    }

    if results.boxes is None:
        return []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        status = status_map.get(class_id, "unknown")

        light_id = str(i)

        detections.append({
            "box": [x1, y1, x2, y2],        # ✅ Sửa "x" → "box"
            "id": light_id,                 # ✅ Sửa "claid" → "id"
            "class_id": class_id,           # ✅ Sửa "boss_id" → "class_id"
            "confidence": conf,
            "status": status
        })

    return detections

def draw_lights(frame, detections):
    color_map = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "unknown": (128, 128, 128)
    }

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        status = det["status"]
        conf = det["confidence"]
        color = color_map.get(status, (255, 255, 255))

        label = f"{det['id']} - {status} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
