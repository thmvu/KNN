from ultralytics import YOLO

def load_model():
    return YOLO("weights/best_1.pt")  # model nhận diện xe

def detect(model, frame):
    results = model(frame, conf=0.5, iou=0.5)

    num_detections = len(results[0].boxes)
    print(f"[🚗 DETECT_VEHICLE] Số lượng object phát hiện: {num_detections}")

    return results[0].plot(conf=True)
